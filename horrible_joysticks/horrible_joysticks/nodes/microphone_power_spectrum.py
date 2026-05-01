#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional

import rclpy
from horrible_js_interfaces.msg import PowerSpectogram
from rclpy.node import Node

from horrible_joysticks.utils.frequency_power_analyzer import (
    FrequencyPowerAnalyzer,
    MicrophonePowerStream,
)


class MicrophonePowerSpectrumNode(Node):
    """Publish microphone FFT power bins as a PowerSpectogram message."""

    def __init__(self) -> None:
        super().__init__("microphone_power_spectrum")

        self.declare_parameter("min_freq", 80.0)
        self.declare_parameter("max_freq", 5000.0)
        self.declare_parameter("number_of_freq_bins", 8192)
        self.declare_parameter("sample_rate", 44100)
        self.declare_parameter("block_size", 4096)
        self.declare_parameter("output_topic", "power_spectrum")

        self._min_freq = self.get_parameter("min_freq").get_parameter_value().double_value
        self._max_freq = self.get_parameter("max_freq").get_parameter_value().double_value
        self._number_of_freq_bins = (
            self.get_parameter("number_of_freq_bins").get_parameter_value().integer_value
        )
        sample_rate = self.get_parameter("sample_rate").get_parameter_value().integer_value
        block_size = self.get_parameter("block_size").get_parameter_value().integer_value
        output_topic = self.get_parameter("output_topic").get_parameter_value().string_value

        if self._min_freq >= self._max_freq:
            raise ValueError("min_freq must be lower than max_freq")
        if self._number_of_freq_bins < 1:
            raise ValueError("number_of_freq_bins must be at least 1")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be greater than 0")
        if block_size <= 0:
            raise ValueError("block_size must be greater than 0")
        if self._number_of_freq_bins * 2 < block_size:
            raise ValueError("number_of_freq_bins * 2 must be >= block_size")

        analyzer = FrequencyPowerAnalyzer(
            sample_rate=sample_rate,
            block_size=block_size,
            fft_size=self._number_of_freq_bins * 2,
            min_freq=self._min_freq,
            max_freq=self._max_freq,
        )
        self._stream = MicrophonePowerStream(analyzer)
        self._stream.start()

        self._publisher = self.create_publisher(PowerSpectogram, output_topic, 1)

        self.get_logger().info(
            "Publishing microphone power spectrum to '%s' from %.1f Hz to %.1f Hz "
            "(sample_rate=%d, block_size=%d, fft_size=%d)"
            % (
                output_topic,
                self._min_freq,
                self._max_freq,
                sample_rate,
                block_size,
                self._number_of_freq_bins * 2,
            )
        )

    def publish_next_spectrum(self, timeout: Optional[float] = None) -> None:
        latest = self._stream.wait_latest(timeout=timeout)
        if latest is None:
            return

        _, _, power = latest
        msg = PowerSpectogram()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "microphone"
        msg.minfreq = float(self._min_freq)
        msg.maxfreq = float(self._max_freq)
        msg.size = len(power)
        msg.powers = power.astype("float32").tolist()

        self._publisher.publish(msg)

    def destroy_node(self) -> bool:
        self._stream.stop()
        return super().destroy_node()


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = MicrophonePowerSpectrumNode()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            node.publish_next_spectrum(timeout=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
