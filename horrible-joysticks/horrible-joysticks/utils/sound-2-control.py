import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from queue import Queue, Empty


NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def hz_to_note(hz):
    midi = int(np.round(69 + 12 * np.log2(max(hz, 1e-12) / 440.0)))
    note_name = NOTE_NAMES[midi % 12]
    octave = (midi // 12) - 1
    nearest_hz = 440.0 * (2.0 ** ((midi - 69) / 12.0))
    cents = 1200.0 * np.log2(max(hz, 1e-12) / nearest_hz)
    return f"{note_name}{octave}", float(cents)


def show_mic_spectrogram(
    sample_rate=44100,
    block_size=4096,
    fft_size=16384,
    min_freq=20,
    max_freq=5000,
    history_seconds=8,
    db_min=-110,
    db_max=0,
    auto_scale=True,
    note_min_hz=220,
    note_max_hz=1800,
    note_harmonics=4,
    note_stable_frames=3,
    note_print_threshold_db=-30,
    note_print_interval_s=0.12,
):
    """
    Opens the default microphone and displays a live spectrogram.
    
    Install dependencies:
        pip install sounddevice numpy matplotlib
    """

    if fft_size < block_size:
        raise ValueError("fft_size must be >= block_size.")

    freq_bins = np.fft.rfftfreq(fft_size, d=1 / sample_rate)
    freq_mask = (freq_bins >= min_freq) & (freq_bins <= max_freq)
    display_freqs = freq_bins[freq_mask]
    note_mask = (freq_bins >= note_min_hz) & (freq_bins <= note_max_hz)
    note_freqs = freq_bins[note_mask]
    if display_freqs.size == 0:
        raise ValueError("No FFT bins in selected frequency range.")
    if note_freqs.size == 0:
        raise ValueError("No FFT bins in note detection frequency range.")

    history_frames = max(10, int(history_seconds * sample_rate / block_size))
    spectrogram = np.full((display_freqs.size, history_frames), -120.0, dtype=np.float32)
    mag_queue = Queue(maxsize=8)
    note_candidate = None
    note_candidate_count = 0
    last_note_print_time = 0.0

    def callback(indata, frames, stream_time, status):
        nonlocal note_candidate, note_candidate_count, last_note_print_time

        if status:
            print(status)

        # Use mono channel from input.
        audio = indata[:, 0]

        window = np.hanning(len(audio))
        windowed = audio * window

        spectrum = np.fft.rfft(windowed, n=fft_size)
        # Normalize to an amplitude-like scale so dB values stay in a useful range.
        scale = max(np.sum(window) / 2.0, 1e-12)
        full_magnitudes = np.abs(spectrum) / scale
        magnitudes = full_magnitudes[freq_mask]
        magnitudes_db = np.multiply(20.0, np.log10(np.maximum(magnitudes, 1e-12)))

        now_s = float(stream_time.inputBufferAdcTime)

        # HPS favors the fundamental over stronger upper harmonics.
        hps = full_magnitudes.copy()
        for harmonic in range(2, max(2, note_harmonics) + 1):
            downsampled = full_magnitudes[::harmonic]
            hps[:len(downsampled)] *= downsampled

        note_hps = hps[note_mask]
        note_idx = int(np.argmax(note_hps))
        fundamental_hz = float(note_freqs[note_idx])
        note_signal_db = float(20.0 * np.log10(max(full_magnitudes[note_mask][note_idx], 1e-12)))

        detected_note = None
        if note_signal_db >= note_print_threshold_db:
            note_name, cents = hz_to_note(fundamental_hz)
            detected_note = (note_name, round(cents, 1), round(fundamental_hz, 1), round(note_signal_db, 1))

        if detected_note is None:
            note_candidate = None
            note_candidate_count = 0
        else:
            if detected_note[0] == note_candidate:
                note_candidate_count += 1
            else:
                note_candidate = detected_note[0]
                note_candidate_count = 1

            if note_candidate_count >= max(1, note_stable_frames) and (now_s - last_note_print_time) >= note_print_interval_s:
                print(
                    f"Note: {detected_note[0]:>3} | {detected_note[2]:7.1f} Hz | "
                    f"{detected_note[1]:+5.1f} cents | {detected_note[3]:6.1f} dB"
                )
                last_note_print_time = now_s

        if mag_queue.full():
            try:
                mag_queue.get_nowait()
            except Empty:
                pass
        mag_queue.put_nowait(magnitudes_db)

    fig, ax = plt.subplots(figsize=(10, 5))
    img = ax.imshow(
        spectrogram,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(-float(history_seconds), 0.0, float(display_freqs[0]), float(display_freqs[-1])),
        cmap="magma",
        vmin=db_min,
        vmax=db_max,
    )
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("Magnitude (dB)")
    ax.set_title("Live Microphone Spectrogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    def update(_frame):
        updated = False
        while True:
            try:
                latest = mag_queue.get_nowait()
            except Empty:
                break

            spectrogram[:, :-1] = spectrogram[:, 1:]
            spectrogram[:, -1] = latest
            updated = True

        if updated:
            img.set_data(spectrogram)
            if auto_scale:
                hi = float(np.percentile(spectrogram, 99.5))
                lo = hi - 90.0
                img.set_clim(max(db_min, lo), max(db_max, hi))
        return (img,)

    with sd.InputStream(
        channels=1,
        samplerate=sample_rate,
        blocksize=block_size,
        callback=callback
    ):
        print("Listening... close the plot window to stop.")
        animation = FuncAnimation(fig, update, interval=40, blit=True, cache_frame_data=False)
        # Keep a reference so the animation is not garbage-collected.
        _ = animation
        plt.tight_layout()
        plt.show()


# Run it
show_mic_spectrogram()