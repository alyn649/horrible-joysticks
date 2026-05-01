from __future__ import annotations

import argparse
from queue import Empty, Queue
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd


class FrequencyPowerAnalyzer:
    """Compute per-bin frequency power using the same FFT pipeline as harmonica analysis.

    Processing steps match the harmonica script:
    - Hann window
    - rFFT (optionally zero-padded to fft_size)
    - Window-amplitude normalization: sum(window) / 2

    Returned "power" is linear power per frequency bin: magnitude^2.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        block_size: int = 4096,
        fft_size: int = 16384,
        min_freq: Optional[float] = None,
        max_freq: Optional[float] = None,
    ) -> None:
        if fft_size < block_size:
            raise ValueError("fft_size must be >= block_size")

        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)
        self.fft_size = int(fft_size)

        self._freqs_all = np.fft.rfftfreq(self.fft_size, d=1.0 / self.sample_rate)
        if min_freq is None:
            min_freq = float(self._freqs_all[0])
        if max_freq is None:
            max_freq = float(self._freqs_all[-1])

        self.min_freq = float(min_freq)
        self.max_freq = float(max_freq)

        self._mask = (self._freqs_all >= self.min_freq) & (self._freqs_all <= self.max_freq)
        if not np.any(self._mask):
            raise ValueError("No FFT bins inside selected frequency range.")

        self.frequencies_hz = self._freqs_all[self._mask]

    def analyze_block(self, audio_block: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (frequencies_hz, linear_power) for one mono audio block."""
        audio = np.asarray(audio_block, dtype=np.float64).reshape(-1)
        if audio.size == 0:
            raise ValueError("audio_block is empty")

        window = np.hanning(audio.size)
        windowed = audio * window

        spectrum = np.fft.rfft(windowed, n=self.fft_size)
        scale = max(np.sum(window) / 2.0, 1e-12)
        magnitudes = np.abs(spectrum) / scale

        power = np.square(magnitudes)[self._mask]
        return self.frequencies_hz, power

    def analyze_block_dbfs(self, audio_block: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (frequencies_hz, power_dbfs) for one mono audio block."""
        freqs, power = self.analyze_block(audio_block)
        power_dbfs = 10.0 * np.log10(np.maximum(power, 1e-24))
        return freqs, power_dbfs


class MicrophonePowerStream:
    """Small helper around sounddevice to fetch latest spectrum power from the mic."""

    def __init__(
        self,
        analyzer: FrequencyPowerAnalyzer,
        channels: int = 1,
        queue_size: int = 8,
    ) -> None:
        self.analyzer = analyzer
        self.channels = int(channels)
        self._queue: Queue = Queue(maxsize=max(1, int(queue_size)))
        self._stream: Optional[sd.InputStream] = None

    def _callback(self, indata, frames, stream_time, status) -> None:
        if status:
            print(status)

        audio = indata[:, 0]
        freqs, power = self.analyzer.analyze_block(audio)
        timestamp_s = float(stream_time.inputBufferAdcTime)

        if self._queue.full():
            try:
                self._queue.get_nowait()
            except Empty:
                pass
        self._queue.put_nowait((timestamp_s, freqs, power))

    def start(self) -> None:
        if self._stream is not None:
            return

        self._stream = sd.InputStream(
            channels=self.channels,
            samplerate=self.analyzer.sample_rate,
            blocksize=self.analyzer.block_size,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return

        self._stream.stop()
        self._stream.close()
        self._stream = None

    def get_latest(self) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
        """Return newest (timestamp_s, frequencies_hz, linear_power), or None."""
        latest = None
        while True:
            try:
                latest = self._queue.get_nowait()
            except Empty:
                break
        return latest

    def wait_latest(
        self,
        timeout: Optional[float] = None,
    ) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
        """Wait for a spectrum, then return the newest queued result.

        Args:
            timeout: Seconds to wait for the first queued spectrum. None waits forever.
        """
        try:
            if timeout is None:
                latest = self._queue.get()
            else:
                latest = self._queue.get(timeout=timeout)
        except Empty:
            return None

        while True:
            try:
                latest = self._queue.get_nowait()
            except Empty:
                break
        return latest

    def __enter__(self) -> "MicrophonePowerStream":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def _run_cli() -> None:
    parser = argparse.ArgumentParser(description="Live microphone frequency-power monitor")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--fft-size", type=int, default=16384)
    parser.add_argument("--min-freq", type=float, default=80.0)
    parser.add_argument("--max-freq", type=float, default=5000.0)
    parser.add_argument("--print-interval", type=float, default=0.12)
    args = parser.parse_args()

    analyzer = FrequencyPowerAnalyzer(
        sample_rate=args.sample_rate,
        block_size=args.block_size,
        fft_size=args.fft_size,
        min_freq=args.min_freq,
        max_freq=args.max_freq,
    )

    print("Listening... press Ctrl+C to stop.")

    try:
        with MicrophonePowerStream(analyzer) as stream:
            while True:
                sd.sleep(max(10, int(args.print_interval * 1000)))
                latest = stream.get_latest()
                if latest is None:
                    continue

                _, freqs, power = latest
                idx = int(np.argmax(power))
                peak_hz = float(freqs[idx])
                peak_power = float(power[idx])
                peak_db = float(10.0 * np.log10(max(peak_power, 1e-24)))
                print(f"Peak: {peak_hz:8.1f} Hz | power {peak_power:.6e} | {peak_db:6.1f} dB")
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    _run_cli()
