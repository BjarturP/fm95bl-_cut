"""
Cut the approved ad/music segments out of the original audio and export the
cleaned episode.

Accepts either:
  - the reviewed CSV from review_export.py (rows with decision=="remove" are
    cut; edit that column by hand first)
  - an Audacity label file you re-exported after adjusting boundaries
    (every row in the file is cut -- delete rows for segments you want to
    keep before re-exporting from Audacity)

Usage:
    python export.py --audio data/raw/episode1.mp3 \\
        --cuts data/labels/episode1_review.csv \\
        --out output/episode1_clean.mp3
"""
import argparse
import csv
from pathlib import Path

import config


def load_cuts_csv(path: Path) -> list:
    cuts = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("decision", "remove").strip().lower() == "remove":
                cuts.append((float(row["start"]), float(row["end"])))
    return cuts


def load_cuts_audacity(path: Path) -> list:
    cuts = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            start, end, *_ = line.rstrip("\n").split("\t")
            cuts.append((float(start), float(end)))
    return cuts


def merge_intervals(intervals: list) -> list:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def invert_intervals(remove_intervals: list, total_duration: float) -> list:
    keep = []
    cursor = 0.0
    for start, end in remove_intervals:
        if start > cursor:
            keep.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < total_duration:
        keep.append((cursor, total_duration))
    return keep


def export(audio_path: Path, cuts: list, out_path: Path) -> None:
    from pydub import AudioSegment

    audio = AudioSegment.from_file(str(audio_path))
    total_duration = len(audio) / 1000.0

    remove_intervals = merge_intervals(cuts)
    keep_intervals = invert_intervals(remove_intervals, total_duration)

    if not keep_intervals:
        raise ValueError("Nothing left to export -- all audio would be removed. Check your cut list.")

    result = None
    for start, end in keep_intervals:
        segment = audio[start * 1000 : end * 1000]
        if result is None:
            result = segment
        else:
            crossfade = min(config.CROSSFADE_MS, len(result), len(segment))
            result = result.append(segment, crossfade=crossfade)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = out_path.suffix.lstrip(".") or "mp3"
    result.export(str(out_path), format=fmt)

    removed = sum(e - s for s, e in remove_intervals)
    print(f"Original duration: {total_duration:.1f}s")
    print(f"Removed:           {removed:.1f}s across {len(remove_intervals)} segment(s)")
    print(f"Final duration:    {len(result) / 1000.0:.1f}s")
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--cuts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.cuts.suffix.lower() == ".csv":
        cuts = load_cuts_csv(args.cuts)
    else:
        cuts = load_cuts_audacity(args.cuts)

    if not cuts:
        raise ValueError(f"No segments to remove found in {args.cuts}")

    export(args.audio, cuts, args.out)


if __name__ == "__main__":
    main()
