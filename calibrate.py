"""
Compare detected candidates against your manually-marked ground-truth
timestamps for one episode, to see what to tune in config.py.

Ground truth CSV format (data/labels/<episode>.csv):
    start,end,label
    12:34,14:10,ad
    25:00,26:45,music
(start/end accept "SS", "MM:SS", or "HH:MM:SS"; label is free text)

Usage:
    python calibrate.py --candidates data/candidates/episode1.json \\
        --labels data/labels/episode1.csv

This is a diagnostic tool, not an auto-tuner: with only one or two labeled
episodes, automatically fitting thresholds risks overfitting to that
episode. Read the false-positive/false-negative list it prints, then adjust
SILENCE_*, MUSIC_*, and BREAK_KEYWORDS in config.py by hand and re-run
detect_breaks.py + calibrate.py until false positives are gone.
"""
import argparse
import csv
import json
import sys
from pathlib import Path


def parse_time(value: str) -> float:
    value = value.strip()
    parts = value.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    raise ValueError(f"Unrecognized time format: {value!r}")


def load_labels(path: Path) -> list:
    labels = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels.append(
                {
                    "start": parse_time(row["start"]),
                    "end": parse_time(row["end"]),
                    "label": row.get("label", "").strip(),
                }
            )
    return labels


def overlap(a_start, a_end, b_start, b_end) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def evaluate(candidates: list, labels: list, coverage_threshold: float = 0.5):
    matched_labels = []
    missed_labels = []
    for gt in labels:
        gt_duration = gt["end"] - gt["start"]
        best_coverage = 0.0
        best_candidate = None
        for c in candidates:
            ov = overlap(gt["start"], gt["end"], c["start"], c["end"])
            coverage = ov / gt_duration if gt_duration > 0 else 0.0
            if coverage > best_coverage:
                best_coverage = coverage
                best_candidate = c
        if best_coverage >= coverage_threshold:
            matched_labels.append({**gt, "coverage": round(best_coverage, 2), "matched_candidate": best_candidate})
        else:
            missed_labels.append({**gt, "best_coverage": round(best_coverage, 2)})

    false_positives = []
    for c in candidates:
        hits_any_gt = any(overlap(gt["start"], gt["end"], c["start"], c["end"]) > 0 for gt in labels)
        if not hits_any_gt:
            false_positives.append(c)

    return matched_labels, missed_labels, false_positives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--coverage-threshold", type=float, default=0.5)
    args = parser.parse_args()

    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    labels = load_labels(args.labels)

    matched, missed, false_positives = evaluate(candidates, labels, args.coverage_threshold)

    recall = len(matched) / len(labels) if labels else float("nan")
    precision = (len(candidates) - len(false_positives)) / len(candidates) if candidates else float("nan")

    print(f"Ground truth segments: {len(labels)}")
    print(f"Candidates detected:   {len(candidates)}")
    print(f"Recall:    {recall:.2%}  ({len(matched)}/{len(labels)} ground-truth breaks found)")
    print(f"Precision: {precision:.2%}  ({len(false_positives)} false positives)")
    print()

    if missed:
        print(f"MISSED ground-truth breaks (false negatives) -- {len(missed)}:")
        for m in missed:
            print(f"  [{m['start']:7.1f}s - {m['end']:7.1f}s] label={m['label']!r} best_coverage={m['best_coverage']:.0%}")
        print()

    if false_positives:
        print(f"FALSE POSITIVES (flagged but no overlap with any ground-truth break) -- {len(false_positives)}:")
        for c in false_positives:
            print(f"  [{c['start']:7.1f}s - {c['end']:7.1f}s] conf={c['confidence']:.2f} reasons={c['reasons']}")
        print()
        print("These are the ones to fix first -- they'd wrongly cut real content.")
    else:
        print("No false positives on this episode.")


if __name__ == "__main__":
    main()
