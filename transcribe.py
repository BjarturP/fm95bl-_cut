"""
Transcribe an episode with word-level timestamps.

Usage:
    python transcribe.py data/raw/episode1.mp3 [--out data/transcripts/episode1.json]

Requires `faster-whisper` (pip install -r requirements.txt). The first run
will download the model weights, which can take a while for large models.

Crash-safe: every finished segment is appended to <out>.partial.jsonl as it
is produced. If the process is killed (macOS memory pressure has done this
twice on multi-hour episodes), rerunning the same command resumes from the
last checkpointed segment instead of restarting the whole file: the remaining
audio is trimmed out with ffmpeg, transcribed, and timestamps are shifted
back to episode time. On success the sidecar is deleted. Use --no-resume to
force a fresh start.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import config


def _partial_path(out_path: Path) -> Path:
    return out_path.with_suffix(out_path.suffix + ".partial.jsonl")


def _load_partial(partial: Path, audio_path: Path) -> tuple[list[dict], float, str | None]:
    """Return (segments, resume_t, language) from a checkpoint sidecar."""
    segments: list[dict] = []
    language = None
    for ln in partial.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        rec = json.loads(ln)
        if rec.get("_meta"):
            if rec.get("audio_path") != str(audio_path):
                print(f"checkpoint is for {rec.get('audio_path')}, not {audio_path} "
                      "— ignoring it", file=sys.stderr)
                return [], 0.0, None
            language = rec.get("language")
        else:
            segments.append(rec)
    resume_t = segments[-1]["end"] if segments else 0.0
    return segments, resume_t, language


def _trim_audio(audio_path: Path, start: float) -> Path:
    tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start),
         "-i", str(audio_path), "-ar", "16000", "-ac", "1", str(tmp)],
        check=True,
    )
    return tmp


def transcribe(audio_path: Path, model_name: str, language: str,
               out_path: Path, resume: bool = True) -> dict:
    from faster_whisper import WhisperModel

    partial = _partial_path(out_path)
    segments: list[dict] = []
    resume_t = 0.0
    if resume and partial.exists():
        segments, resume_t, ckpt_lang = _load_partial(partial, audio_path)
        if segments:
            language = ckpt_lang or language
            print(f"Resuming from checkpoint: {len(segments)} segments, "
                  f"t={resume_t:.1f}s", file=sys.stderr)

    model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=4)

    work_audio = audio_path
    trimmed = None
    if resume_t > 0:
        trimmed = _trim_audio(audio_path, resume_t)
        work_audio = trimmed

    try:
        segments_iter, info = model.transcribe(
            str(work_audio),
            language=language,
            word_timestamps=True,
            vad_filter=False,
        )
        total_duration = resume_t + info.duration

        with partial.open("a", encoding="utf-8") as ckpt:
            if resume_t == 0:
                ckpt.write(json.dumps({"_meta": True, "audio_path": str(audio_path),
                                       "language": info.language},
                                      ensure_ascii=False) + "\n")
                ckpt.flush()
            for seg in segments_iter:
                words = [
                    {"start": w.start + resume_t, "end": w.end + resume_t,
                     "word": w.word.strip()}
                    for w in (seg.words or [])
                ]
                rec = {
                    "start": seg.start + resume_t,
                    "end": seg.end + resume_t,
                    "text": seg.text.strip(),
                    "words": words,
                    "avg_logprob": seg.avg_logprob,
                    "no_speech_prob": seg.no_speech_prob,
                    "compression_ratio": seg.compression_ratio,
                }
                segments.append(rec)
                ckpt.write(json.dumps(rec, ensure_ascii=False) + "\n")
                ckpt.flush()
                print(f"[{rec['start']:7.1f}s] {rec['text']}", file=sys.stderr)
    finally:
        if trimmed is not None:
            trimmed.unlink(missing_ok=True)

    return {
        "audio_path": str(audio_path),
        "language": info.language if resume_t == 0 else (language or info.language),
        "duration": total_duration,
        "segments": segments,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--model", default=config.WHISPER_MODEL)
    parser.add_argument("--language", default=config.WHISPER_LANGUAGE)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore any checkpoint and transcribe from the start")
    args = parser.parse_args()

    out_path = args.out or Path("data/transcripts") / (args.audio_path.stem + ".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = transcribe(args.audio_path, args.model, args.language,
                        out_path, resume=not args.no_resume)

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _partial_path(out_path).unlink(missing_ok=True)
    print(f"Wrote transcript to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
