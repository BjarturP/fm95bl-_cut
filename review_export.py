"""
Turn detected candidates into human-reviewable artifacts:

1. A CSV report with a `decision` column (defaults to "remove") that you can
   edit directly -- set a row to "keep" to override a false positive.
2. An Audacity label track (.txt) you can import (File > Import > Labels) to
   see and adjust the exact cut boundaries against the waveform.

Usage:
    python review_export.py --candidates data/candidates/episode1.json \\
        --out-csv data/labels/episode1_review.csv \\
        --out-audacity data/labels/episode1_audacity.txt
"""
import argparse
import csv
import json
from pathlib import Path

import config


def to_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def write_csv(candidates: list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "start", "end", "start_hms", "end_hms", "duration", "confidence", "reasons", "decision"])
        for i, c in enumerate(candidates):
            default_decision = "remove" if c["confidence"] >= config.AUTO_REMOVE_CONFIDENCE else "review"
            writer.writerow(
                [
                    i,
                    c["start"],
                    c["end"],
                    to_hms(c["start"]),
                    to_hms(c["end"]),
                    c["duration"],
                    c["confidence"],
                    "; ".join(c["reasons"]),
                    default_decision,
                ]
            )


def write_audacity_labels(candidates: list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, c in enumerate(candidates):
            label = f"#{i} conf={c['confidence']:.2f} {'; '.join(c['reasons'])}"
            f.write(f"{c['start']:.3f}\t{c['end']:.3f}\t{label}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-audacity", type=Path, required=True)
    args = parser.parse_args()

    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))

    write_csv(candidates, args.out_csv)
    write_audacity_labels(candidates, args.out_audacity)

    print(f"Wrote {len(candidates)} candidates to:")
    print(f"  {args.out_csv}  (edit the 'decision' column: remove / keep)")
    print(f"  {args.out_audacity}  (import into Audacity: File > Import > Labels)")


if __name__ == "__main__":
    main()
