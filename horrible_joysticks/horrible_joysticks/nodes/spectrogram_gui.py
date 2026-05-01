#!/usr/bin/env python3
from __future__ import annotations

from queue import Empty, Queue
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from horrible_js_interfaces.msg import PowerSpectogram
from matplotlib.animation import FuncAnimation
from rclpy.node import Node


class MicrophoneSpectrogramGuiNode(Node):
    """Plot PowerSpectogram messages with the same Matplotlib style as sound-2-control."""

    def __init__(self) -> None:
        super().__init__("microphone_spectrogram_gui")

        self.declare_parameter("input_topic", "power_spectrum")
        self.declare_parameter("history_seconds", 8.0)
        self.declare_parameter("history_frames", 200)
        self.declare_parameter("db_min", -110.0)
        self.declare_parameter("db_max", 0.0)
        self.declare_parameter("auto_scale", True)

        self.input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self.history_seconds = (
            self.get_parameter("history_seconds").get_parameter_value().double_value
        )
        self.history_frames = (
            self.get_parameter("history_frames").get_parameter_value().integer_value
        )
        self.db_min = self.get_parameter("db_min").get_parameter_value().double_value
        self.db_max = self.get_parameter("db_max").get_parameter_value().double_value
        self.auto_scale = self.get_parameter("auto_scale").get_parameter_value().bool_value

        if self.history_seconds <= 0.0:
            raise ValueError("history_seconds must be greater than 0")
        if self.history_frames < 10:
            raise ValueError("history_frames must be at least 10")
        if self.db_min >= self.db_max:
            raise ValueError("db_min must be lower than db_max")

        self.spectrum_queue: Queue[tuple[float, float, np.ndarray]] = Queue(maxsize=1)
        self.create_subscription(
            PowerSpectogram,
            self.input_topic,
            self._on_spectrum,
            1,
        )

        self.get_logger().info(
            "Plotting microphone spectrogram from '%s' over %.1f seconds"
            % (self.input_topic, self.history_seconds)
        )

    def _on_spectrum(self, msg: PowerSpectogram) -> None:
        powers = np.asarray(msg.powers, dtype=np.float32)
        if powers.size == 0:
            return

        powers_db = np.multiply(10.0, np.log10(np.maximum(powers, 1e-24)))
        latest = (float(msg.minfreq), float(msg.maxfreq), powers_db.astype(np.float32))

        if self.spectrum_queue.full():
            try:
                self.spectrum_queue.get_nowait()
            except Empty:
                pass
        self.spectrum_queue.put_nowait(latest)


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = MicrophoneSpectrogramGuiNode()

    spectrogram: Optional[np.ndarray] = None
    img = None
    cbar = None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Live Microphone Spectrogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    def reset_spectrogram(min_freq: float, max_freq: float, bins: int):
        nonlocal spectrogram, img, cbar

        spectrogram = np.full((bins, node.history_frames), -120.0, dtype=np.float32)
        if img is not None:
            img.remove()
        if cbar is not None:
            cbar.remove()

        img = ax.imshow(
            spectrogram,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(-float(node.history_seconds), 0.0, float(min_freq), float(max_freq)),
            cmap="magma",
            vmin=node.db_min,
            vmax=node.db_max,
        )
        cbar = plt.colorbar(img, ax=ax)
        cbar.set_label("Power (dB)")
        return img

    def update(_frame):
        nonlocal spectrogram, img

        rclpy.spin_once(node, timeout_sec=0.0)

        latest_spectrum = None
        while True:
            try:
                latest_spectrum = node.spectrum_queue.get_nowait()
            except Empty:
                break

        updated = latest_spectrum is not None
        if latest_spectrum is not None:
            min_freq, max_freq, latest = latest_spectrum
            if spectrogram is None or latest.size != spectrogram.shape[0]:
                reset_spectrogram(min_freq, max_freq, int(latest.size))

            spectrogram[:, :-1] = spectrogram[:, 1:]
            spectrogram[:, -1] = latest

        if updated and img is not None:
            img.set_data(spectrogram)
            if node.auto_scale:
                hi = float(np.percentile(spectrogram, 99.5))
                lo = hi - 90.0
                img.set_clim(max(node.db_min, lo), max(node.db_max, hi))

        if img is None:
            return ()
        return (img,)

    try:
        print("Listening... close the plot window to stop.")
        animation = FuncAnimation(fig, update, interval=40, blit=True, cache_frame_data=False)
        _ = animation
        plt.tight_layout()
        plt.show()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
