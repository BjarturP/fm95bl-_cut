"""
Transcribe an episode with word-level timestamps.

Usage:
    python transcribe.py data/raw/episode1.mp3 [--out data/transcripts/episode1.json]

Requires `faster-whisper` (pip install -r requirements.txt). The first run
will download the model weights, which can take a while for large models.
"""
import argparse
import json
import sys
from pathlib import Path

import config


def transcribe(audio_path: Path, model_name: str, language: str) -> dict:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=4)
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        vad_filter=False,
    )

    segments = []
    for seg in segments_iter:
        words = [
            {"start": w.start, "end": w.end, "word": w.word.strip()}
            for w in (seg.words or [])
        ]
        segments.append(
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "words": words,
                "avg_logprob": seg.avg_logprob,
                "no_speech_prob": seg.no_speech_prob,
                "compression_ratio": seg.compression_ratio,
            }
        )
        print(f"[{seg.start:7.1f}s] {seg.text.strip()}", file=sys.stderr)

    return {
        "audio_path": str(audio_path),
        "language": info.language,
        "duration": info.duration,
        "segments": segments,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--model", default=config.WHISPER_MODEL)
    parser.add_argument("--language", default=config.WHISPER_LANGUAGE)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_path = args.out or Path("data/transcripts") / (args.audio_path.stem + ".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = transcribe(args.audio_path, args.model, args.language)

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote transcript to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
