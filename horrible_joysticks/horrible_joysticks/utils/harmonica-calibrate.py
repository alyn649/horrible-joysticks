import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sounddevice as sd

DEFAULT_HOLES_C = [
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

OUTPUT_PATH = Path("harmonica-calibration.json")


def load_existing_results(path):
    path = Path(path)
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

    holes = payload.get("holes", [])
    results = {}
    for row in holes:
        try:
            hole = int(row["hole"])
            results[hole] = {
                "hole": hole,
                "blow_hz": float(row["blow_hz"]),
                "draw_hz": float(row["draw_hz"]),
                "blow_dbfs": float(row.get("blow_dbfs", -120.0)),
                "draw_dbfs": float(row.get("draw_dbfs", -120.0)),
            }
        except (KeyError, TypeError, ValueError):
            continue
    return results


def save_results(
    results_by_hole,
    sample_rate,
    duration_s,
    fft_size,
    harmonics,
    output_path,
):
    ordered_holes = sorted(results_by_hole.keys())
    rows = [results_by_hole[h] for h in ordered_holes]

    payload = {
        "instrument": "10-hole diatonic harmonica",
        "key": "C",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sample_rate": sample_rate,
        "duration_s": duration_s,
        "fft_size": fft_size,
        "harmonics": harmonics,
        "holes": rows,
    }

    output_path = Path(output_path)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Calibration written to {output_path.resolve()} ({len(rows)} hole(s) saved)")


def detect_fundamental_hz(
    audio,
    sample_rate,
    fft_size,
    min_hz,
    max_hz,
    harmonics=4,
):
    window = np.hanning(len(audio))
    windowed = audio * window

    spectrum = np.fft.rfft(windowed, n=fft_size)
    scale = max(np.sum(window) / 2.0, 1e-12)
    magnitudes = np.abs(spectrum) / scale
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)

    band = (freqs >= min_hz) & (freqs <= max_hz)
    if not np.any(band):
        return None, None

    hps = magnitudes.copy()
    for harmonic in range(2, max(2, harmonics) + 1):
        downsampled = magnitudes[::harmonic]
        hps[: len(downsampled)] *= downsampled

    band_hps = hps[band]
    idx_local = int(np.argmax(band_hps))
    freq_band = freqs[band]
    mag_band = magnitudes[band]

    fundamental_hz = float(freq_band[idx_local])
    level_dbfs = float(20.0 * np.log10(max(mag_band[idx_local], 1e-12)))
    return fundamental_hz, level_dbfs


def record_note_frequency(
    expected_hz,
    direction,
    hole,
    sample_rate=44100,
    duration_s=0.6,
    fft_size=16384,
    harmonics=4,
):
    margin = 0.45
    min_hz = max(70.0, expected_hz * (1.0 - margin))
    max_hz = expected_hz * (1.0 + margin)

    print(f"Prepare hole {hole} {direction}. Press Enter to record {duration_s:.1f}s...")
    input()

    recording = sd.rec(int(duration_s * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()

    audio = recording[:, 0]
    return detect_fundamental_hz(
        audio=audio,
        sample_rate=sample_rate,
        fft_size=fft_size,
        min_hz=min_hz,
        max_hz=max_hz,
        harmonics=harmonics,
    )


def run_calibration(
    sample_rate=44100,
    duration_s=0.6,
    fft_size=16384,
    harmonics=4,
    min_accept_dbfs=-55.0,
    output_path=OUTPUT_PATH,
):
    output_path = Path(output_path)
    existing = load_existing_results(output_path)

    if existing:
        completed = sorted(existing.keys())
        next_hole_default = min(10, max(completed) + 1)
        print(f"Found existing calibration with holes: {completed}")
    else:
        next_hole_default = 1

    start_raw = input(f"Start from hole (1-10) [{next_hole_default}]: ").strip()
    try:
        start_hole = int(start_raw) if start_raw else next_hole_default
    except ValueError:
        start_hole = next_hole_default
    start_hole = min(10, max(1, start_hole))

    results_by_hole = dict(existing)

    print("C harmonica calibration")
    print("For each prompt, play a clean single hole note and keep it steady.")
    print("You can press Ctrl+C any time to cancel.\n")

    try:
        for row in DEFAULT_HOLES_C:
            hole = int(row["hole"])
            if hole < start_hole:
                continue

            prev = results_by_hole.get(hole, {})

            blow_hz, blow_db = record_note_frequency(
                expected_hz=float(row["blow_hz"]),
                direction="BLOW",
                hole=hole,
                sample_rate=sample_rate,
                duration_s=duration_s,
                fft_size=fft_size,
                harmonics=harmonics,
            )
            if blow_hz is None or blow_db is None or blow_db < min_accept_dbfs:
                print(f"Hole {hole} blow too weak/noisy ({blow_db}). Keeping fallback value.")

            blow_hz_val = round(float(blow_hz if blow_hz is not None else prev.get("blow_hz", row["blow_hz"])), 2)
            blow_db_val = round(float(blow_db if blow_db is not None else prev.get("blow_dbfs", -120.0)), 1)

            draw_hz, draw_db = record_note_frequency(
                expected_hz=float(row["draw_hz"]),
                direction="DRAW",
                hole=hole,
                sample_rate=sample_rate,
                duration_s=duration_s,
                fft_size=fft_size,
                harmonics=harmonics,
            )
            if draw_hz is None or draw_db is None or draw_db < min_accept_dbfs:
                print(f"Hole {hole} draw too weak/noisy ({draw_db}). Keeping fallback value.")

            draw_hz_val = round(float(draw_hz if draw_hz is not None else prev.get("draw_hz", row["draw_hz"])), 2)
            draw_db_val = round(float(draw_db if draw_db is not None else prev.get("draw_dbfs", -120.0)), 1)

            results_by_hole[hole] = {
                "hole": hole,
                "blow_hz": blow_hz_val,
                "draw_hz": draw_hz_val,
                "blow_dbfs": blow_db_val,
                "draw_dbfs": draw_db_val,
            }

            save_results(
                results_by_hole=results_by_hole,
                sample_rate=sample_rate,
                duration_s=duration_s,
                fft_size=fft_size,
                harmonics=harmonics,
                output_path=output_path,
            )
            print(
                f"Saved hole {hole}: blow {blow_hz_val:.2f} Hz ({blow_db_val:.1f} dBFS), "
                f"draw {draw_hz_val:.2f} Hz ({draw_db_val:.1f} dBFS)\n"
            )
    except KeyboardInterrupt:
        print("\nInterrupted. Saving calibration progress...")
        save_results(
            results_by_hole=results_by_hole,
            sample_rate=sample_rate,
            duration_s=duration_s,
            fft_size=fft_size,
            harmonics=harmonics,
            output_path=output_path,
        )
        print("Partial calibration saved.")
        return

    print("Calibration complete.")


if __name__ == "__main__":
    run_calibration()
