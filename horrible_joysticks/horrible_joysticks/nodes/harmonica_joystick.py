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


HOLES_C = [
    {"hole": 1, "blow_hz": 261.63, "draw_hz": 293.66},
    {"hole": 2, "blow_hz": 329.63, "draw_hz": 392.00},
    {"hole": 3, "blow_hz": 392.00, "draw_hz": 493.88},
    {"hole": 4, "blow_hz": 523.25, "draw_hz": 587.33},
    {"hole": 5, "blow_hz": 659.25, "draw_hz": 698.46},
    {"hole": 6, "blow_hz": 783.99, "draw_hz": 880.00},
    {"hole": 7, "blow_hz": 1046.50, "draw_hz": 987.77},
    {"hole": 8, "blow_hz": 1318.51, "draw_hz": 1174.66},
    {"hole": 9, "blow_hz": 1567.98, "draw_hz": 1396.91},
    {"hole": 10, "blow_hz": 2093.00, "draw_hz": 1760.00},
]


@dataclass(frozen=True)
class HarmonicaDetection:
    hole: int
    direction: str
    strength: str
    target_hz: float
    level_dbfs: float
    confident: bool


@dataclass(frozen=True)
class HarmonicaAnalysis:
    detection: HarmonicaDetection
    frequencies_hz: np.ndarray
    spectrum_dbfs: np.ndarray
    blow_db: np.ndarray
    draw_db: np.ndarray
    joystick_axes: np.ndarray
    joystick_direction: str


def package_config_path(filename: str) -> Path:
    share_dir = Path(get_package_share_directory("horrible_joysticks"))
    return share_dir / "config" / filename


def resolve_calibration_path(calibration_path: str) -> Path:
    path = Path(calibration_path).expanduser()
    if path.is_absolute():
        return path
    return package_config_path(calibration_path)


def load_hole_map(calibration_path: Path) -> list[dict[str, float]]:
    if not calibration_path.exists():
        print(f"Calibration file not found at {calibration_path}. Using built-in C harmonica map.")
        return HOLES_C

    with calibration_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    holes = payload.get("holes")
    if not isinstance(holes, list) or len(holes) == 0:
        raise ValueError("Calibration JSON must contain a non-empty 'holes' list.")

    normalized = []
    for row in holes:
        normalized.append(
            {
                "hole": int(row["hole"]),
                "blow_hz": float(row["blow_hz"]),
                "draw_hz": float(row["draw_hz"]),
            }
        )

    normalized.sort(key=lambda x: x["hole"])
    print(f"Loaded harmonica calibration from {calibration_path}.")
    return normalized


def tone_energy(
    freqs_hz: np.ndarray,
    magnitudes: np.ndarray,
    fundamental_hz: float,
    max_harmonics: int = 4,
    cents_width: float = 38.0,
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
        score += (weighted / norm) / harmonic

    return float(score)


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


class HarmonicaHoleAnalyzerNode(Node):
    def __init__(self, enable_gui_override: Optional[bool] = None) -> None:
        super().__init__("harmonica_hole_analyzer")

        self.declare_parameter("input_topic", "power_spectrum")
        self.declare_parameter("output_topic", "joy")
        self.declare_parameter("detected_output_topic", "detected_joy")
        self.declare_parameter("calibration_file", "harmonica-calibration.json")
        self.declare_parameter("max_harmonics", 4)
        self.declare_parameter("detect_threshold_dbfs", -40.0)
        self.declare_parameter("print_interval_s", 0.12)
        self.declare_parameter("enable_gui", True)
        self.declare_parameter("joystick_zero_dbfs", -40.0)
        self.declare_parameter("joystick_full_scale_dbfs", -10.0)

        self.input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self.output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self.detected_output_topic = (
            self.get_parameter("detected_output_topic").get_parameter_value().string_value
        )
        calibration_file = (
            self.get_parameter("calibration_file").get_parameter_value().string_value
        )
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
        if enable_gui_override is not None:
            self.enable_gui = enable_gui_override

        if self.max_harmonics < 1:
            raise ValueError("max_harmonics must be at least 1")
        if self.print_interval_s < 0.0:
            raise ValueError("print_interval_s must be >= 0")
        if self.joystick_zero_dbfs >= self.joystick_full_scale_dbfs:
            raise ValueError("joystick_zero_dbfs must be lower than joystick_full_scale_dbfs")

        self.calibration_path = resolve_calibration_path(calibration_file)
        hole_map = load_hole_map(self.calibration_path)
        self.holes = np.array([item["hole"] for item in hole_map], dtype=np.int32)
        self.blow_targets = np.array([item["blow_hz"] for item in hole_map], dtype=np.float64)
        self.draw_targets = np.array([item["draw_hz"] for item in hole_map], dtype=np.float64)

        self.analysis_queue: Queue[HarmonicaAnalysis] = Queue(maxsize=1)
        self._last_print_time = 0.0

        self.joy_publisher = self.create_publisher(Joy, self.output_topic, 1)
        self.detected_joy_publisher = self.create_publisher(Joy, self.detected_output_topic, 1)
        self.create_subscription(PowerSpectogram, self.input_topic, self._on_spectrum, 1)

        self.get_logger().info(
            "Analyzing harmonica holes from '%s', publishing Joy to '%s' and detected Joy to "
            "'%s' using calibration '%s'"
            % (
                self.input_topic,
                self.output_topic,
                self.detected_output_topic,
                self.calibration_path,
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

    def analyze_spectrum(self, freqs: np.ndarray, magnitudes: np.ndarray) -> HarmonicaAnalysis:
        spec_db = 20.0 * np.log10(np.maximum(magnitudes, 1e-12))

        blow_scores = np.array(
            [
                tone_energy(freqs, magnitudes, hz, max_harmonics=self.max_harmonics)
                for hz in self.blow_targets
            ]
        )
        draw_scores = np.array(
            [
                tone_energy(freqs, magnitudes, hz, max_harmonics=self.max_harmonics)
                for hz in self.draw_targets
            ]
        )

        blow_db = 20.0 * np.log10(np.maximum(blow_scores, 1e-12))
        draw_db = 20.0 * np.log10(np.maximum(draw_scores, 1e-12))

        blow_total = float(np.sum(blow_scores))
        draw_total = float(np.sum(draw_scores))
        is_blow = blow_total >= draw_total
        joystick_direction = "blow" if is_blow else "draw"
        direction_scores = blow_scores if is_blow else draw_scores
        direction_db = blow_db if is_blow else draw_db
        joystick_axes = self._scores_to_joystick_axes(direction_db, is_blow=is_blow)

        best_idx = int(np.argmax(direction_scores))
        best_hole = int(self.holes[best_idx])
        target_hz = float(self.blow_targets[best_idx] if is_blow else self.draw_targets[best_idx])

        freq_idx = int(np.argmin(np.abs(freqs - target_hz)))
        level_dbfs = float(spec_db[freq_idx])
        detection = HarmonicaDetection(
            hole=best_hole,
            direction=joystick_direction,
            strength=classify_strength(level_dbfs),
            target_hz=target_hz,
            level_dbfs=level_dbfs,
            confident=level_dbfs >= self.detect_threshold_dbfs,
        )

        return HarmonicaAnalysis(
            detection=detection,
            frequencies_hz=freqs,
            spectrum_dbfs=spec_db,
            blow_db=blow_db,
            draw_db=draw_db,
            joystick_axes=joystick_axes,
            joystick_direction=joystick_direction,
        )

    def _scores_to_joystick_axes(self, scores_db: np.ndarray, is_blow: bool) -> np.ndarray:
        span = self.joystick_full_scale_dbfs - self.joystick_zero_dbfs
        normalized = (scores_db - self.joystick_zero_dbfs) / span
        axes = np.clip(normalized, 0.0, 1.0).astype(np.float32)
        if not is_blow:
            axes *= -1.0
        return axes

    def _make_joy_message(self, axes: np.ndarray) -> Joy:
        joy_msg = Joy()
        joy_msg.header.stamp = self.get_clock().now().to_msg()
        joy_msg.header.frame_id = "harmonica"
        joy_msg.axes = axes.astype(np.float32).tolist()
        joy_msg.buttons = []
        return joy_msg

    def _detected_joystick_axes(self, analysis: HarmonicaAnalysis) -> np.ndarray:
        axes = np.zeros_like(analysis.joystick_axes, dtype=np.float32)
        detection = analysis.detection
        if detection.confident:
            matches = np.flatnonzero(self.holes == detection.hole)
            if matches.size > 0:
                idx = int(matches[0])
                axes[idx] = analysis.joystick_axes[idx]
        return axes

    def output_analysis(self, analysis: HarmonicaAnalysis) -> None:
        """Final analysis output hook.

        This is the final output point. Add any extra publishers here.
        """
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
                f"Hole {detection.hole} | {detection.direction} | {detection.strength} "
                f"({detection.level_dbfs:.1f} dBFS) | axes "
                f"{np.array2string(analysis.joystick_axes, precision=2, suppress_small=True)}"
            )
        else:
            status = "No confident note detected"

        sys.stdout.write("\r" + status + " " * 12)
        sys.stdout.flush()
        self._last_print_time = now_s


class HarmonicaAnalyzerGui:
    def __init__(self, node: HarmonicaHoleAnalyzerNode) -> None:
        self.node = node
        self.fig, (self.ax_hist, self.ax_spec) = plt.subplots(
            2,
            1,
            figsize=(11, 7),
            gridspec_kw={"height_ratios": [2.1, 1.2]},
        )

        x = np.arange(len(node.holes))
        width = 0.25
        initial_db = np.full(len(node.holes), -120.0)
        self.bars_blow = self.ax_hist.bar(
            x - width,
            initial_db,
            width=width,
            label="Blow",
            color="#2a9d8f",
        )
        self.bars_draw = self.ax_hist.bar(
            x,
            initial_db,
            width=width,
            label="Draw",
            color="#e76f51",
        )
        self.ax_joy = self.ax_hist.twinx()
        self.bars_joy = self.ax_joy.bar(
            x + width,
            np.zeros(len(node.holes)),
            width=width,
            label="Joy axis",
            color="#457b9d",
            alpha=0.75,
        )

        self.ax_hist.set_xticks(x)
        self.ax_hist.set_xticklabels([str(h) for h in node.holes])
        self.ax_hist.set_ylabel("Estimated note energy (dB)")
        self.ax_hist.set_xlabel("Harmonica hole")
        self.ax_hist.set_ylim(-90, 0)
        self.ax_hist.set_title("Harmonica Hole Histogram and Joystick Axes")
        self.ax_joy.set_ylabel("Joy axis")
        self.ax_joy.set_ylim(-1.0, 1.0)
        hist_handles, hist_labels = self.ax_hist.get_legend_handles_labels()
        joy_handles, joy_labels = self.ax_joy.get_legend_handles_labels()
        self.ax_hist.legend(hist_handles + joy_handles, hist_labels + joy_labels, loc="upper right")
        self.status_text = self.ax_hist.text(
            0.01,
            0.97,
            "Waiting for signal...",
            transform=self.ax_hist.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )

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
                *self.bars_blow,
                *self.bars_draw,
                *self.bars_joy,
                self.spec_line,
                self.status_text,
            )

        self.spec_line.set_data(latest.frequencies_hz, latest.spectrum_dbfs)
        self.ax_spec.set_xlim(float(latest.frequencies_hz[0]), float(latest.frequencies_hz[-1]))

        for i in range(len(self.node.holes)):
            self.bars_blow[i].set_height(float(latest.blow_db[i]))
            self.bars_draw[i].set_height(float(latest.draw_db[i]))
            self.bars_joy[i].set_height(float(latest.joystick_axes[i]))

        detection = latest.detection
        if detection.confident:
            status = (
                f"Hole {detection.hole} | {detection.direction} | {detection.strength} "
                f"({detection.level_dbfs:.1f} dBFS) | joy {latest.joystick_direction}"
            )
        else:
            status = "No confident note detected"
        self.status_text.set_text(status)

        return (
            *self.bars_blow,
            *self.bars_draw,
            *self.bars_joy,
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
    node = HarmonicaHoleAnalyzerNode(enable_gui_override=parsed_args.enable_gui)

    try:
        if node.enable_gui:
            gui = HarmonicaAnalyzerGui(node)
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
