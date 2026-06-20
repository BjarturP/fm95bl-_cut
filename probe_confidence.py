"""
One-off research script: test whether Whisper's per-segment avg_logprob /
no_speech_prob / compression_ratio can distinguish real host talk from
music/ad breaks better than transcript word-rate alone.

Stratified sample: takes a clip from the middle of every labeled music/ad
segment AND every real talk gap between segments in a calibration episode,
transcribes each clip standalone, and aggregates the per-segment confidence
stats by clip type (music vs talk). This is deliberately NOT limited to the
two contested regions under discussion -- if a threshold only separates
those two spots it's not a generalizable rule, it's overfitting.

Usage:
    python probe_confidence.py --audio data/raw/episode1.mp3 \\
        --labels data/labels/episode1.csv --out data/confidence_probe/episode1.json
"""
import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import config


def parse_hms(s: str) -> float:
    parts = [float(p) for p in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, sec = parts
    return h * 3600 + m * 60 + sec


def load_labels(path: Path) -> list:
    labels = []
    with path.open() as f:
        for row in csv.DictReader(f):
            labels.append((parse_hms(row["start"]), parse_hms(row["end"])))
    return sorted(labels)


def build_clip_plan(labels: list, max_len: float = 90.0, min_gap: float = 20.0) -> list:
    clips = []
    for s, e in labels:
        mid = (s + e) / 2
        cs, ce = max(s, mid - max_len / 2), min(e, mid + max_len / 2)
        clips.append({"type": "music", "label_start": s, "label_end": e, "clip_start": round(cs, 1), "clip_end": round(ce, 1)})
    for i in range(len(labels) - 1):
        gap_s, gap_e = labels[i][1], labels[i + 1][0]
        if gap_e - gap_s < min_gap:
            continue
        mid = (gap_s + gap_e) / 2
        cs, ce = max(gap_s, mid - max_len / 2), min(gap_e, mid + max_len / 2)
        clips.append({"type": "talk", "label_start": gap_s, "label_end": gap_e, "clip_start": round(cs, 1), "clip_end": round(ce, 1)})
    return clips


def extract_clip(audio_path: Path, start: float, end: float, out_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(start), "-to", str(end),
            "-i", str(audio_path),
            "-ar", "16000", "-ac", "1",
            str(out_path),
        ],
        check=True,
    )


def transcribe_clip(model, clip_path: Path, language: str) -> dict:
    segments_iter, _info = model.transcribe(str(clip_path), language=language, word_timestamps=True, vad_filter=False)
    segments = []
    for seg in segments_iter:
        segments.append(
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "n_words": len(seg.words or []),
                "avg_logprob": seg.avg_logprob,
                "no_speech_prob": seg.no_speech_prob,
                "compression_ratio": seg.compression_ratio,
            }
        )
    return {"segments": segments}


def aggregate(segments: list) -> dict:
    if not segments:
        return {"n_segments": 0, "avg_logprob": None, "no_speech_prob": None, "compression_ratio": None, "n_words": 0}
    n = len(segments)
    return {
        "n_segments": n,
        "avg_logprob": round(sum(s["avg_logprob"] for s in segments) / n, 4),
        "no_speech_prob": round(sum(s["no_speech_prob"] for s in segments) / n, 4),
        "compression_ratio": round(sum(s["compression_ratio"] for s in segments) / n, 4),
        "n_words": sum(s["n_words"] for s in segments),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-clip-len", type=float, default=90.0)
    args = parser.parse_args()

    from faster_whisper import WhisperModel

    labels = load_labels(args.labels)
    clips = build_clip_plan(labels, max_len=args.max_clip_len)
    print(f"Planned {len(clips)} clips ({sum(c['clip_end'] - c['clip_start'] for c in clips):.0f}s total)", file=sys.stderr)

    model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=4)

    with tempfile.TemporaryDirectory() as tmp:
        for i, clip in enumerate(clips):
            clip_path = Path(tmp) / f"clip_{i}.wav"
            extract_clip(args.audio, clip["clip_start"], clip["clip_end"], clip_path)
            result = transcribe_clip(model, clip_path, config.WHISPER_LANGUAGE)
            clip["segments"] = result["segments"]
            clip["agg"] = aggregate(result["segments"])
            print(
                f"  [{i+1}/{len(clips)}] {clip['type']:5s} "
                f"{clip['clip_start']:7.1f}-{clip['clip_end']:7.1f}s  "
                f"agg={clip['agg']}",
                file=sys.stderr,
            )

    out_path = args.out or Path("data/confidence_probe") / (args.audio.stem + ".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(clips, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
