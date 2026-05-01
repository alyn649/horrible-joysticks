import sys
from queue import Empty, Queue
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from matplotlib.animation import FuncAnimation

# Standard 10-hole Richter-tuned C diatonic harmonica.
HOLES_C = [
    {"hole": 1, "blow_hz": 261.63, "draw_hz": 293.66},  # C4, D4
    {"hole": 2, "blow_hz": 329.63, "draw_hz": 392.00},  # E4, G4
    {"hole": 3, "blow_hz": 392.00, "draw_hz": 493.88},  # G4, B4
    {"hole": 4, "blow_hz": 523.25, "draw_hz": 587.33},  # C5, D5
    {"hole": 5, "blow_hz": 659.25, "draw_hz": 698.46},  # E5, F5
    {"hole": 6, "blow_hz": 783.99, "draw_hz": 880.00},  # G5, A5
    {"hole": 7, "blow_hz": 1046.50, "draw_hz": 987.77},  # C6, B5
    {"hole": 8, "blow_hz": 1318.51, "draw_hz": 1174.66},  # E6, D6
    {"hole": 9, "blow_hz": 1567.98, "draw_hz": 1396.91},  # G6, F6
    {"hole": 10, "blow_hz": 2093.00, "draw_hz": 1760.00},  # C7, A6
]

CALIBRATION_PATH = Path("harmonica-calibration.json")


def load_hole_map(calibration_path=CALIBRATION_PATH):
    path = Path(calibration_path)
    if not path.exists():
        print(f"Calibration file not found at {path}. Using built-in C harmonica map.")
        return HOLES_C

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    holes = payload.get("holes")
    if not isinstance(holes, list) or len(holes) == 0:
        raise ValueError("Calibration JSON must contain a non-empty 'holes' list.")

    normalized = []
    for row in holes:
        hole = int(row["hole"])
        blow_hz = float(row["blow_hz"])
        draw_hz = float(row["draw_hz"])
        normalized.append({"hole": hole, "blow_hz": blow_hz, "draw_hz": draw_hz})

    normalized.sort(key=lambda x: x["hole"])
    print(f"Loaded calibration from {path}.")
    return normalized


def _tone_energy(freqs_hz, magnitudes, fundamental_hz, max_harmonics=4, cents_width=38.0):
    """Estimate how much spectral energy belongs to one note (fundamental + harmonics)."""
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


def _classify_strength(level_dbfs):
    if level_dbfs < -45.0:
        return "very light"
    if level_dbfs < -35.0:
        return "light"
    if level_dbfs < -25.0:
        return "medium"
    if level_dbfs < -15.0:
        return "strong"
    return "very strong"


def run_harmonica_analyzer(
    sample_rate=44100,
    block_size=4096,
    fft_size=16384,
    min_freq=80,
    max_freq=3200,
    max_harmonics=4,
    print_interval_s=0.12,
    detect_threshold_dbfs=-50,
    calibration_path=CALIBRATION_PATH,
):
    if fft_size < block_size:
        raise ValueError("fft_size must be >= block_size")

    all_freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    analysis_mask = (all_freqs >= min_freq) & (all_freqs <= max_freq)
    freqs = all_freqs[analysis_mask]
    if freqs.size == 0:
        raise ValueError("No FFT bins in selected analysis range.")

    q = Queue(maxsize=6)
    last_print_time = 0.0

    hole_map = load_hole_map(calibration_path=calibration_path)
    holes = np.array([item["hole"] for item in hole_map])
    blow_targets = np.array([item["blow_hz"] for item in hole_map], dtype=np.float64)
    draw_targets = np.array([item["draw_hz"] for item in hole_map], dtype=np.float64)

    def callback(indata, frames, stream_time, status):
        if status:
            print(status)

        audio = indata[:, 0]
        window = np.hanning(len(audio))
        windowed = audio * window

        spectrum = np.fft.rfft(windowed, n=fft_size)
        # Window-amplitude normalization keeps dBFS estimates stable.
        scale = max(np.sum(window) / 2.0, 1e-12)
        magnitudes = (np.abs(spectrum) / scale)[analysis_mask]

        if q.full():
            try:
                q.get_nowait()
            except Empty:
                pass
        q.put_nowait((float(stream_time.inputBufferAdcTime), magnitudes))

    fig, (ax_hist, ax_spec) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [2.1, 1.2]})
    x = np.arange(len(holes))
    width = 0.38

    blow_db = np.full(len(holes), -120.0)
    draw_db = np.full(len(holes), -120.0)

    bars_blow = ax_hist.bar(x - width / 2, blow_db, width=width, label="Blow", color="#2a9d8f")
    bars_draw = ax_hist.bar(x + width / 2, draw_db, width=width, label="Draw (Suck)", color="#e76f51")

    ax_hist.set_xticks(x)
    ax_hist.set_xticklabels([str(h) for h in holes])
    ax_hist.set_ylabel("Estimated note energy (dB)")
    ax_hist.set_xlabel("Harmonica hole")
    ax_hist.set_ylim(-90, 0)
    ax_hist.set_title("C Harmonica Hole Histogram")
    ax_hist.legend(loc="upper right")
    status_text = ax_hist.text(
        0.01,
        0.97,
        "Waiting for signal...",
        transform=ax_hist.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )

    spec_line, = ax_spec.plot(freqs, np.full(freqs.shape, -120.0), color="#264653", linewidth=1.1)
    ax_spec.set_xlim(min_freq, max_freq)
    ax_spec.set_ylim(-110, 5)
    ax_spec.set_xlabel("Frequency (Hz)")
    ax_spec.set_ylabel("Magnitude (dBFS)")
    ax_spec.set_title("Current Spectrum")

    def update(_frame):
        nonlocal last_print_time

        latest = None
        while True:
            try:
                latest = q.get_nowait()
            except Empty:
                break

        if latest is None:
            return (*bars_blow, *bars_draw, spec_line, status_text)

        now_s, magnitudes = latest
        spec_db = 20.0 * np.log10(np.maximum(magnitudes, 1e-12))

        blow_scores = np.array([
            _tone_energy(freqs, magnitudes, hz, max_harmonics=max_harmonics) for hz in blow_targets
        ])
        draw_scores = np.array([
            _tone_energy(freqs, magnitudes, hz, max_harmonics=max_harmonics) for hz in draw_targets
        ])

        blow_db_vals = 20.0 * np.log10(np.maximum(blow_scores, 1e-12))
        draw_db_vals = 20.0 * np.log10(np.maximum(draw_scores, 1e-12))

        for i in range(len(holes)):
            bars_blow[i].set_height(float(blow_db_vals[i]))
            bars_draw[i].set_height(float(draw_db_vals[i]))

        dominant_per_hole = np.maximum(blow_scores, draw_scores)
        best_idx = int(np.argmax(dominant_per_hole))
        best_hole = int(holes[best_idx])
        is_blow = bool(blow_scores[best_idx] >= draw_scores[best_idx])
        direction = "blow" if is_blow else "draw (suck)"
        target_hz = float(blow_targets[best_idx] if is_blow else draw_targets[best_idx])

        # Fundamental-level estimate near the selected note frequency.
        freq_idx = int(np.argmin(np.abs(freqs - target_hz)))
        level_dbfs = float(spec_db[freq_idx])
        strength = _classify_strength(level_dbfs)

        confident = level_dbfs >= detect_threshold_dbfs
        if confident:
            status = f"Hole {best_hole} | {direction} | {strength} ({level_dbfs:.1f} dBFS)"
        else:
            status = "No confident note detected"

        status_text.set_text(status)
        spec_line.set_ydata(spec_db)

        if (now_s - last_print_time) >= print_interval_s:
            if confident:
                sys.stdout.write("\r" + status + " " * 12)
            else:
                sys.stdout.write("\rNo confident note detected" + " " * 12)
            sys.stdout.flush()
            last_print_time = now_s

        return (*bars_blow, *bars_draw, spec_line, status_text)

    with sd.InputStream(
        channels=1,
        samplerate=sample_rate,
        blocksize=block_size,
        callback=callback,
    ):
        print("Listening... close the plot window to stop.")
        anim = FuncAnimation(fig, update, interval=45, blit=True, cache_frame_data=False)
        _ = anim
        plt.tight_layout()
        plt.show()

    print()


if __name__ == "__main__":
    run_harmonica_analyzer()
