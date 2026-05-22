#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from horrible_js_interfaces.msg import PowerSpectogram
from matplotlib.animation import FuncAnimation
from rclpy.node import Node
from sensor_msgs.msg import Joy


HOLE_LABELS = ["T", "1", "2", "3", "4", "5", "6", "7"]

RECORDER_C_FINGERINGS = [
    {
        "note": "C5",
        "frequency_hz": 523.25,
        "covered": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    },
    {
        "note": "D5",
        "frequency_hz": 587.33,
        "covered": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
    },
    {
        "note": "E5",
        "frequency_hz": 659.25,
        "covered": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
    },
    {
        "note": "F5",
        "frequency_hz": 698.46,
        "covered": [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    },
    {
        "note": "G5",
        "frequency_hz": 783.99,
        "covered": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    },
    {
        "note": "A5",
        "frequency_hz": 880.00,
        "covered": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    {
        "note": "B5",
        "frequency_hz": 987.77,
        "covered": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    {
        "note": "C6",
        "frequency_hz": 1046.50,
        "covered": [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    {
        "note": "D6",
        "frequency_hz": 1174.66,
        "covered": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    {
        "note": "E6",
        "frequency_hz": 1318.51,
        "covered": [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
    },
    {
        "note": "F6",
        "frequency_hz": 1396.91,
        "covered": [0.5, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    },
    {
        "note": "G6",
        "frequency_hz": 1567.98,
        "covered": [0.5, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    },
    {
        "note": "A6",
        "frequency_hz": 1760.00,
        "covered": [0.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
]


@dataclass(frozen=True)
class RecorderDetection:
    note: str
    target_hz: float
    strength: str
    level_dbfs: float
    blow_power: float
    note_margin_db: float
    peak_margin_db: float
    raw_confident: bool
    confident: bool


@dataclass(frozen=True)
class RecorderAnalysis:
    detection: RecorderDetection
    frequencies_hz: np.ndarray
    spectrum_dbfs: np.ndarray
    note_names: list[str]
    note_db: np.ndarray
    hole_power: np.ndarray
    joystick_axes: np.ndarray


def package_config_path(filename: str) -> Path:
    share_dir = Path(get_package_share_directory("horrible_joysticks"))
    return share_dir / "config" / filename


def resolve_fingering_path(fingering_path: str) -> Path:
    path = Path(fingering_path).expanduser()
    if path.is_absolute():
        return path
    return package_config_path(fingering_path)


def load_fingering_map(fingering_path: Path) -> tuple[list[str], list[dict[str, object]]]:
    if not fingering_path.exists():
        print(
            f"Fingering file not found at {fingering_path}. "
            "Using built-in soprano C recorder map."
        )
        return HOLE_LABELS, RECORDER_C_FINGERINGS

    with fingering_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    hole_labels = payload.get("hole_labels", HOLE_LABELS)
    notes = payload.get("notes")
    if not isinstance(hole_labels, list) or len(hole_labels) == 0:
        raise ValueError("Fingering JSON must contain a non-empty 'hole_labels' list.")
    if not isinstance(notes, list) or len(notes) == 0:
        raise ValueError("Fingering JSON must contain a non-empty 'notes' list.")

    normalized = []
    for row in notes:
        covered = [float(value) for value in row["covered"]]
        if len(covered) != len(hole_labels):
            raise ValueError(
                f"Note {row['note']} has {len(covered)} cover values, "
                f"expected {len(hole_labels)}."
            )
        normalized.append(
            {
                "note": str(row["note"]),
                "frequency_hz": float(row["frequency_hz"]),
                "covered": covered,
            }
        )

    normalized.sort(key=lambda x: x["frequency_hz"])
    print(f"Loaded recorder fingering map from {fingering_path}.")
    return [str(label) for label in hole_labels], normalized


def tone_energy(
    freqs_hz: np.ndarray,
    magnitudes: np.ndarray,
    fundamental_hz: float,
    max_harmonics: int = 4,
    cents_width: float = 22.0,
    noise_inner_cents: float = 80.0,
    noise_outer_cents: float = 420.0,
) -> float:
    score = 0.0
    nyquist = float(freqs_hz[-1])

    for harmonic in range(1, max_harmonics + 1):
        center_hz = fundamental_hz * harmonic
        if center_hz >= nyquist:
            break

        cents = 1200.0 * np.log2(np.maximum(freqs_hz, 1e-9) / center_hz)
        weights = np.exp(-0.5 * (cents / cents_width) ** 2)
        weighted = np.sum(magnitudes * weights)
        norm = np.sum(weights) + 1e-12
        signal = weighted / norm

        abs_cents = np.abs(cents)
        noise_mask = (
            (abs_cents >= noise_inner_cents)
            & (abs_cents <= noise_outer_cents)
        )
        noise_floor = 0.0
        if np.any(noise_mask):
            noise_floor = float(np.percentile(magnitudes[noise_mask], 70.0))

        score += max(signal - noise_floor, 0.0) / harmonic

    return float(score)


def frequency_band_peak_dbfs(
    freqs_hz: np.ndarray,
    spectrum_dbfs: np.ndarray,
    center_hz: float,
    cents_width: float,
) -> float:
    cents = 1200.0 * np.log2(np.maximum(freqs_hz, 1e-9) / center_hz)
    mask = np.abs(cents) <= cents_width
    if np.any(mask):
        return float(np.max(spectrum_dbfs[mask]))

    freq_idx = int(np.argmin(np.abs(freqs_hz - center_hz)))
    return float(spectrum_dbfs[freq_idx])


def local_noise_floor_dbfs(
    freqs_hz: np.ndarray,
    spectrum_dbfs: np.ndarray,
    center_hz: float,
    inner_cents: float,
    outer_cents: float,
) -> float:
    cents = 1200.0 * np.log2(np.maximum(freqs_hz, 1e-9) / center_hz)
    abs_cents = np.abs(cents)
    mask = (abs_cents >= inner_cents) & (abs_cents <= outer_cents)
    if np.any(mask):
        return float(np.percentile(spectrum_dbfs[mask], 70.0))
    return float(np.percentile(spectrum_dbfs, 70.0))


def best_note_margin_db(note_db: np.ndarray, best_idx: int) -> float:
    if note_db.size < 2:
        return float("inf")

    others = np.delete(note_db, best_idx)
    return float(note_db[best_idx] - np.max(others))


def classify_strength(level_dbfs: float) -> str:
    if level_dbfs < -45.0:
        return "very light"
    if level_dbfs < -35.0:
        return "light"
    if level_dbfs < -25.0:
        return "medium"
    if level_dbfs < -15.0:
        return "strong"
    return "very strong"


class RecorderJoystickNode(Node):
    def __init__(self, enable_gui_override: Optional[bool] = None) -> None:
        super().__init__("recorder_joystick")

        self.declare_parameter("input_topic", "power_spectrum")
        self.declare_parameter("output_topic", "joy")
        self.declare_parameter("detected_output_topic", "detected_joy")
        self.declare_parameter("fingering_file", "recorder-fingerings.json")
        self.declare_parameter("max_harmonics", 4)
        self.declare_parameter("detect_threshold_dbfs", -34.0)
        self.declare_parameter("print_interval_s", 0.12)
        self.declare_parameter("enable_gui", True)
        self.declare_parameter("joystick_zero_dbfs", -36.0)
        self.declare_parameter("joystick_full_scale_dbfs", -10.0)
        self.declare_parameter("frequency_window_cents", 22.0)
        self.declare_parameter("local_noise_inner_cents", 80.0)
        self.declare_parameter("local_noise_outer_cents", 420.0)
        self.declare_parameter("minimum_peak_margin_db", 9.0)
        self.declare_parameter("minimum_note_margin_db", 2.5)
        self.declare_parameter("confirmation_frames", 2)

        self.input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self.output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self.detected_output_topic = (
            self.get_parameter("detected_output_topic").get_parameter_value().string_value
        )
        fingering_file = self.get_parameter("fingering_file").get_parameter_value().string_value
        self.max_harmonics = (
            self.get_parameter("max_harmonics").get_parameter_value().integer_value
        )
        self.detect_threshold_dbfs = (
            self.get_parameter("detect_threshold_dbfs").get_parameter_value().double_value
        )
        self.print_interval_s = (
            self.get_parameter("print_interval_s").get_parameter_value().double_value
        )
        self.enable_gui = self.get_parameter("enable_gui").get_parameter_value().bool_value
        self.joystick_zero_dbfs = (
            self.get_parameter("joystick_zero_dbfs").get_parameter_value().double_value
        )
        self.joystick_full_scale_dbfs = (
            self.get_parameter("joystick_full_scale_dbfs").get_parameter_value().double_value
        )
        self.frequency_window_cents = (
            self.get_parameter("frequency_window_cents").get_parameter_value().double_value
        )
        self.local_noise_inner_cents = (
            self.get_parameter("local_noise_inner_cents").get_parameter_value().double_value
        )
        self.local_noise_outer_cents = (
            self.get_parameter("local_noise_outer_cents").get_parameter_value().double_value
        )
        self.minimum_peak_margin_db = (
            self.get_parameter("minimum_peak_margin_db").get_parameter_value().double_value
        )
        self.minimum_note_margin_db = (
            self.get_parameter("minimum_note_margin_db").get_parameter_value().double_value
        )
        self.confirmation_frames = (
            self.get_parameter("confirmation_frames").get_parameter_value().integer_value
        )
        if enable_gui_override is not None:
            self.enable_gui = enable_gui_override

        if self.max_harmonics < 1:
            raise ValueError("max_harmonics must be at least 1")
        if self.print_interval_s < 0.0:
            raise ValueError("print_interval_s must be >= 0")
        if self.joystick_zero_dbfs >= self.joystick_full_scale_dbfs:
            raise ValueError("joystick_zero_dbfs must be lower than joystick_full_scale_dbfs")
        if self.frequency_window_cents <= 0.0:
            raise ValueError("frequency_window_cents must be greater than 0")
        if self.local_noise_inner_cents <= self.frequency_window_cents:
            raise ValueError("local_noise_inner_cents must be greater than frequency_window_cents")
        if self.local_noise_outer_cents <= self.local_noise_inner_cents:
            raise ValueError(
                "local_noise_outer_cents must be greater than local_noise_inner_cents"
            )
        if self.minimum_peak_margin_db < 0.0:
            raise ValueError("minimum_peak_margin_db must be >= 0")
        if self.minimum_note_margin_db < 0.0:
            raise ValueError("minimum_note_margin_db must be >= 0")
        if self.confirmation_frames < 1:
            raise ValueError("confirmation_frames must be at least 1")

        self.fingering_path = resolve_fingering_path(fingering_file)
        self.hole_labels, note_map = load_fingering_map(self.fingering_path)
        self.note_names = [str(item["note"]) for item in note_map]
        self.note_targets = np.array(
            [float(item["frequency_hz"]) for item in note_map],
            dtype=np.float64,
        )
        self.fingerings = np.array([item["covered"] for item in note_map], dtype=np.float32)

        self.analysis_queue: Queue[RecorderAnalysis] = Queue(maxsize=1)
        self._last_print_time = 0.0
        self._candidate_note: Optional[str] = None
        self._candidate_count = 0

        self.joy_publisher = self.create_publisher(Joy, self.output_topic, 1)
        self.detected_joy_publisher = self.create_publisher(Joy, self.detected_output_topic, 1)
        self.create_subscription(PowerSpectogram, self.input_topic, self._on_spectrum, 1)

        self.get_logger().info(
            "Analyzing recorder notes from '%s', publishing Joy to '%s' and detected Joy to "
            "'%s' using fingering map '%s'"
            % (
                self.input_topic,
                self.output_topic,
                self.detected_output_topic,
                self.fingering_path,
            )
        )

    def _on_spectrum(self, msg: PowerSpectogram) -> None:
        power = np.asarray(msg.powers, dtype=np.float64)
        if power.size == 0:
            return

        freqs = np.linspace(float(msg.minfreq), float(msg.maxfreq), power.size)
        magnitudes = np.sqrt(np.maximum(power, 0.0))
        analysis = self.analyze_spectrum(freqs, magnitudes)

        self.output_analysis(analysis)

        if self.enable_gui:
            if self.analysis_queue.full():
                try:
                    self.analysis_queue.get_nowait()
                except Empty:
                    pass
            self.analysis_queue.put_nowait(analysis)

    def analyze_spectrum(self, freqs: np.ndarray, magnitudes: np.ndarray) -> RecorderAnalysis:
        spec_db = 20.0 * np.log10(np.maximum(magnitudes, 1e-12))

        note_scores = np.array(
            [
                tone_energy(
                    freqs,
                    magnitudes,
                    hz,
                    max_harmonics=self.max_harmonics,
                    cents_width=self.frequency_window_cents,
                    noise_inner_cents=self.local_noise_inner_cents,
                    noise_outer_cents=self.local_noise_outer_cents,
                )
                for hz in self.note_targets
            ]
        )
        note_db = 20.0 * np.log10(np.maximum(note_scores, 1e-12))
        note_power = self._scores_to_joystick_axes(note_db)

        best_idx = int(np.argmax(note_scores))
        target_hz = float(self.note_targets[best_idx])
        level_dbfs = frequency_band_peak_dbfs(
            freqs,
            spec_db,
            target_hz,
            self.frequency_window_cents,
        )
        noise_floor_dbfs = local_noise_floor_dbfs(
            freqs,
            spec_db,
            target_hz,
            self.local_noise_inner_cents,
            self.local_noise_outer_cents,
        )
        peak_margin_db = level_dbfs - noise_floor_dbfs
        note_margin_db = best_note_margin_db(note_db, best_idx)
        blow_power = float(note_power[best_idx])
        raw_confident = (
            level_dbfs >= self.detect_threshold_dbfs
            and peak_margin_db >= self.minimum_peak_margin_db
            and note_margin_db >= self.minimum_note_margin_db
            and blow_power > 0.0
        )
        confident = self._confirm_candidate(self.note_names[best_idx], raw_confident)
        joystick_axes = self._fingering_to_joystick_axes(best_idx, blow_power, confident)

        detection = RecorderDetection(
            note=self.note_names[best_idx],
            target_hz=target_hz,
            strength=classify_strength(level_dbfs),
            level_dbfs=level_dbfs,
            blow_power=blow_power,
            note_margin_db=note_margin_db,
            peak_margin_db=peak_margin_db,
            raw_confident=raw_confident,
            confident=confident,
        )

        return RecorderAnalysis(
            detection=detection,
            frequencies_hz=freqs,
            spectrum_dbfs=spec_db,
            note_names=self.note_names,
            note_db=note_db,
            hole_power=joystick_axes,
            joystick_axes=joystick_axes,
        )

    def _scores_to_joystick_axes(self, scores_db: np.ndarray) -> np.ndarray:
        span = self.joystick_full_scale_dbfs - self.joystick_zero_dbfs
        normalized = (scores_db - self.joystick_zero_dbfs) / span
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)

    def _confirm_candidate(self, note_name: str, raw_confident: bool) -> bool:
        if not raw_confident:
            self._candidate_note = None
            self._candidate_count = 0
            return False

        if note_name == self._candidate_note:
            self._candidate_count += 1
        else:
            self._candidate_note = note_name
            self._candidate_count = 1

        return self._candidate_count >= self.confirmation_frames

    def _fingering_to_joystick_axes(
        self,
        note_idx: int,
        blow_power: float,
        confident: bool,
    ) -> np.ndarray:
        if not confident:
            return np.zeros(len(self.hole_labels), dtype=np.float32)
        return (self.fingerings[note_idx] * blow_power).astype(np.float32)

    def _make_joy_message(self, axes: np.ndarray) -> Joy:
        joy_msg = Joy()
        joy_msg.header.stamp = self.get_clock().now().to_msg()
        joy_msg.header.frame_id = "recorder"
        joy_msg.axes = axes.astype(np.float32).tolist()
        joy_msg.buttons = []
        return joy_msg

    def _detected_joystick_axes(self, analysis: RecorderAnalysis) -> np.ndarray:
        return analysis.joystick_axes.astype(np.float32)

    def output_analysis(self, analysis: RecorderAnalysis) -> None:
        self.joy_publisher.publish(self._make_joy_message(analysis.joystick_axes))
        self.detected_joy_publisher.publish(
            self._make_joy_message(self._detected_joystick_axes(analysis))
        )

        detection = analysis.detection
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if (now_s - self._last_print_time) < self.print_interval_s:
            return

        if detection.confident:
            status = (
                f"Note {detection.note} | {detection.strength} "
                f"({detection.level_dbfs:.1f} dBFS, power {detection.blow_power:.2f}, "
                f"margins {detection.peak_margin_db:.1f}/{detection.note_margin_db:.1f} dB) | "
                f"axes {np.array2string(analysis.joystick_axes, precision=2, suppress_small=True)}"
            )
        elif detection.raw_confident:
            status = (
                f"Confirming {detection.note} | "
                f"margins {detection.peak_margin_db:.1f}/{detection.note_margin_db:.1f} dB"
            )
        else:
            status = (
                f"No confident recorder note detected | best {detection.note} "
                f"margins {detection.peak_margin_db:.1f}/{detection.note_margin_db:.1f} dB"
            )

        sys.stdout.write("\r" + status + " " * 12)
        sys.stdout.flush()
        self._last_print_time = now_s


class RecorderJoystickGui:
    def __init__(self, node: RecorderJoystickNode) -> None:
        self.node = node
        self.fig, (self.ax_holes, self.ax_notes, self.ax_spec) = plt.subplots(
            3,
            1,
            figsize=(11, 8),
            gridspec_kw={"height_ratios": [1.5, 1.6, 1.2]},
        )

        hole_x = np.arange(len(node.hole_labels))
        self.bars_holes = self.ax_holes.bar(
            hole_x,
            np.zeros(len(node.hole_labels)),
            width=0.55,
            label="Hole power",
            color="#457b9d",
        )
        self.ax_holes.set_xticks(hole_x)
        self.ax_holes.set_xticklabels(node.hole_labels)
        self.ax_holes.set_ylabel("Joy axis")
        self.ax_holes.set_xlabel("Recorder hole")
        self.ax_holes.set_ylim(0.0, 1.0)
        self.ax_holes.set_title("Recorder Hole Power")
        self.ax_holes.legend(loc="upper right")
        self.status_text = self.ax_holes.text(
            0.01,
            0.96,
            "Waiting for signal...",
            transform=self.ax_holes.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )

        note_x = np.arange(len(node.note_names))
        self.bars_notes = self.ax_notes.bar(
            note_x,
            np.full(len(node.note_names), -120.0),
            width=0.65,
            label="Note energy",
            color="#2a9d8f",
        )
        self.ax_notes.set_xticks(note_x)
        self.ax_notes.set_xticklabels(node.note_names)
        self.ax_notes.set_ylabel("Estimated note energy (dB)")
        self.ax_notes.set_xlabel("Recorder note")
        self.ax_notes.set_ylim(-90.0, 0.0)
        self.ax_notes.set_title("Recorder Note Histogram")
        self.ax_notes.legend(loc="upper right")

        (self.spec_line,) = self.ax_spec.plot([], [], color="#264653", linewidth=1.1)
        self.ax_spec.set_ylim(-110, 5)
        self.ax_spec.set_xlabel("Frequency (Hz)")
        self.ax_spec.set_ylabel("Magnitude (dBFS)")
        self.ax_spec.set_title("Current Spectrum")

    def update(self, _frame):
        rclpy.spin_once(self.node, timeout_sec=0.0)

        latest = None
        while True:
            try:
                latest = self.node.analysis_queue.get_nowait()
            except Empty:
                break

        if latest is None:
            return (
                *self.bars_holes,
                *self.bars_notes,
                self.spec_line,
                self.status_text,
            )

        self.spec_line.set_data(latest.frequencies_hz, latest.spectrum_dbfs)
        self.ax_spec.set_xlim(float(latest.frequencies_hz[0]), float(latest.frequencies_hz[-1]))

        for i in range(len(self.node.hole_labels)):
            self.bars_holes[i].set_height(float(latest.hole_power[i]))
        for i in range(len(self.node.note_names)):
            self.bars_notes[i].set_height(float(latest.note_db[i]))

        detection = latest.detection
        if detection.confident:
            status = (
                f"Note {detection.note} | {detection.strength} "
                f"({detection.level_dbfs:.1f} dBFS) | power {detection.blow_power:.2f} | "
                f"margins {detection.peak_margin_db:.1f}/{detection.note_margin_db:.1f} dB"
            )
        elif detection.raw_confident:
            status = (
                f"Confirming {detection.note} | "
                f"margins {detection.peak_margin_db:.1f}/{detection.note_margin_db:.1f} dB"
            )
        else:
            status = (
                f"No confident recorder note detected | best {detection.note} "
                f"margins {detection.peak_margin_db:.1f}/{detection.note_margin_db:.1f} dB"
            )
        self.status_text.set_text(status)

        return (
            *self.bars_holes,
            *self.bars_notes,
            self.spec_line,
            self.status_text,
        )


def main(args: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gui", action="store_true", dest="enable_gui")
    group.add_argument("--no-gui", action="store_false", dest="enable_gui")
    parser.set_defaults(enable_gui=None)
    parsed_args, ros_args = parser.parse_known_args(args)

    rclpy.init(args=ros_args)
    node = RecorderJoystickNode(enable_gui_override=parsed_args.enable_gui)

    try:
        if node.enable_gui:
            gui = RecorderJoystickGui(node)
            print("Listening... close the plot window to stop.")
            anim = FuncAnimation(
                gui.fig,
                gui.update,
                interval=45,
                blit=True,
                cache_frame_data=False,
            )
            _ = anim
            plt.tight_layout()
            plt.show()
        else:
            rclpy.spin(node)
    finally:
        print()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
