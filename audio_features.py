"""
Extract rolling acoustic features used to flag silence / music-like / sudden
loudness-change regions, independent of the transcript.

Usage:
    python audio_features.py data/raw/episode1.mp3 [--out data/features/episode1.json]

These are deliberately simple, explainable heuristics (not a trained
classifier) so behavior can be reasoned about and tuned via config.py:

- silence: rolling RMS loudness below SILENCE_RMS_DB
- music_score: a smoothed combination of harmonic energy ratio (from
  harmonic/percussive source separation) and spectral flatness -- sustained
  tonal content scores higher, speech-like formant structure scores lower.
  This alone is NOT a reliable music/speech split for every genre (tested:
  misses percussive/guitar-heavy songs) -- detect_breaks.py combines it with
  transcript word-rate (speech density) before thresholding into runs.
- loudness jump: large frame-to-frame change in RMS, e.g. a jingle starting
  abruptly after speech

Processes audio in chunks (FEATURE_CHUNK_SECONDS) rather than loading a
whole multi-hour episode into memory at once -- on 8GB RAM, loading +
harmonic/percussive separation on a 2-hour file caused heavy swapping and
near-total stalls. Each chunk is loaded with a few seconds of padding on
either side so the rolling smoothing/frame analysis doesn't have edge
artifacts at chunk boundaries; the padding is discarded after processing.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

import config


def rms_to_db(rms: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(rms, 1e-10))


def _process_chunk(y: np.ndarray, sr: int, hop_seconds: float, t0: float) -> list:
    """Compute per-window features for one chunk of audio, returning windows
    with absolute timestamps starting at t0."""
    import librosa

    hop_length = int(hop_seconds * sr)
    frame_length = hop_length * 2

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = rms_to_db(rms)

    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=frame_length, hop_length=hop_length)[0]

    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harm_rms = librosa.feature.rms(y=y_harmonic, frame_length=frame_length, hop_length=hop_length)[0]
    perc_rms = librosa.feature.rms(y=y_percussive, frame_length=frame_length, hop_length=hop_length)[0]
    harmonic_ratio = harm_rms / np.maximum(harm_rms + perc_rms, 1e-10)

    n = min(len(rms_db), len(zcr), len(flatness), len(harmonic_ratio))
    rms_db, zcr, flatness, harmonic_ratio = rms_db[:n], zcr[:n], flatness[:n], harmonic_ratio[:n]

    # Heuristic music-likeness score in [0, 1]: high harmonic ratio + low
    # spectral flatness (tonal, not noise-like) + low zero-crossing rate
    # (smoother waveform than typical fricative-heavy speech).
    flatness_norm = np.clip(flatness / (flatness.max() + 1e-10), 0, 1)
    zcr_norm = np.clip(zcr / (zcr.max() + 1e-10), 0, 1)
    music_score_raw = np.clip(harmonic_ratio * (1 - flatness_norm) * (1 - 0.5 * zcr_norm), 0, 1)

    # Smooth over a rolling window: the raw per-second score is noisy enough
    # (fluctuates above/below any fixed threshold within a single song) that
    # thresholding it directly fragments real music segments into tiny slivers.
    smooth_k = max(1, round(config.MUSIC_SMOOTHING_SECONDS / hop_seconds))
    kernel = np.ones(2 * smooth_k + 1) / (2 * smooth_k + 1)
    music_score = np.convolve(music_score_raw, kernel, mode="same")

    windows = []
    for i in range(n):
        windows.append(
            {
                "t": round(float(t0 + i * hop_seconds), 2),
                "rms_db": round(float(rms_db[i]), 2),
                "zcr": round(float(zcr[i]), 4),
                "flatness": round(float(flatness[i]), 4),
                "harmonic_ratio": round(float(harmonic_ratio[i]), 4),
                "music_score": round(float(music_score[i]), 4),
            }
        )
    return windows


def extract_features(audio_path: Path, hop_seconds: float) -> dict:
    import librosa

    total_duration = librosa.get_duration(path=str(audio_path))
    chunk_seconds = config.FEATURE_CHUNK_SECONDS
    pad_seconds = config.MUSIC_SMOOTHING_SECONDS * 3

    all_windows = []
    chunk_start = 0.0
    while chunk_start < total_duration - 1e-6:
        chunk_end = min(chunk_start + chunk_seconds, total_duration)
        load_start = max(0.0, chunk_start - pad_seconds)
        load_end = min(total_duration, chunk_end + pad_seconds)

        y, sr = librosa.load(
            str(audio_path), sr=22050, mono=True, offset=load_start, duration=load_end - load_start
        )
        windows = _process_chunk(y, sr, hop_seconds, t0=load_start)
        all_windows.extend(w for w in windows if chunk_start <= w["t"] < chunk_end)

        print(f"  processed {chunk_start:.0f}s - {chunk_end:.0f}s / {total_duration:.0f}s", file=sys.stderr)
        chunk_start = chunk_end

    n = len(all_windows)
    silence_runs = _merge_runs(
        [w["t"] for w in all_windows if w["rms_db"] < config.SILENCE_RMS_DB],
        hop_seconds,
        min_duration=config.SILENCE_MIN_DURATION,
    )
    # Note: no standalone music_runs here -- music_score alone isn't a
    # reliable speech/music split (see module docstring). detect_breaks.py
    # combines it with transcript word-rate before extracting music runs.

    loudness_jumps = []
    for i in range(1, n):
        delta = abs(all_windows[i]["rms_db"] - all_windows[i - 1]["rms_db"])
        if delta >= config.LOUDNESS_JUMP_DB:
            loudness_jumps.append({"t": all_windows[i]["t"], "delta_db": round(float(delta), 2)})

    return {
        "audio_path": str(audio_path),
        "hop_seconds": hop_seconds,
        "duration": round(float(total_duration), 2),
        "windows": all_windows,
        "silence_runs": silence_runs,
        "loudness_jumps": loudness_jumps,
    }


def _merge_runs(timestamps: list, hop_seconds: float, min_duration: float) -> list:
    """Collapse a sorted list of window start-times (where some condition held)
    into contiguous [start, end) runs, dropping runs shorter than min_duration."""
    if not timestamps:
        return []
    runs = []
    run_start = timestamps[0]
    prev = timestamps[0]
    for t in timestamps[1:]:
        if t - prev > hop_seconds * 1.5:
            runs.append((run_start, prev + hop_seconds))
            run_start = t
        prev = t
    runs.append((run_start, prev + hop_seconds))

    return [
        {"start": round(s, 2), "end": round(e, 2), "duration": round(e - s, 2)}
        for s, e in runs
        if (e - s) >= min_duration
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--hop", type=float, default=config.FEATURE_HOP_SECONDS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_path = args.out or Path("data/features") / (args.audio_path.stem + ".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = extract_features(args.audio_path, args.hop)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"Wrote features to {out_path} "
        f"({len(result['silence_runs'])} silence runs, {len(result['loudness_jumps'])} loudness jumps)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
