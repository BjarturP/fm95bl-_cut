"""
Boundary-accuracy diagnostic — step 6 of the boundary-precision plan.

Reads a hybrid_review_<ep>.csv (from hybrid_review.py) and a ground-truth cut
file, matches each predicted region to the ground-truth music cut it best
overlaps, and reports SIGNED boundary error (predicted - actual) split BY
CATEGORY (AUTO / REVIEW-S / REVIEW-B / REVIEW-H / DROP?).

Why signed + median + per-category:
  - signed  → tells direction, not just size. A systematically negative
              REVIEW-H Δstart means stage-2 expand_boundaries is over-reaching
              upstream (fix belongs there), not that a downstream clawback is
              missing.
  - median  → one 15-min closing block or a merged 3-cut region can't dominate
              the summary the way a mean lets it.
  - by cat  → says whether the next effort belongs in the Shazam pipeline
              (AUTO/REVIEW-S) or in finalize_cuts.py (REVIEW-H).

EXT rows are excluded (they are alternate ends of a primary row, not their own
cut). Rows that overlap no GT cut are reported separately as "no-GT" (potential
false positives — not boundary-scored, since there is no truth boundary).

Usage (repeat --ep per episode; aggregates REVIEW-H across all of them):
    python experiments/boundary_eval.py \\
        --ep fm95blo-2011-11-09 data/labels/hybrid_review_fm95blo-2011-11-09.csv data/labels/fm95blo_2011_11_09.csv \\
        --ep fm95blo-2012-02-28 data/labels/hybrid_review_fm95blo-2012-02-28.csv data/labels/2012-02-08_actual_cuts.txt \\
        --ep fm95blo-2012-03-09 data/labels/hybrid_review_fm95blo-2012-03-09.csv data/labels/2012-03-09_actual_cuts.txt
"""

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

MIN_OVERLAP = 30.0          # s, same convention as make_review's GT eval
MUSIC_LABELS = {"music", "music/ads"}   # music-break scope; pure "ads" excluded
CATEGORY_ORDER = ["AUTO", "REVIEW-S", "REVIEW-B", "REVIEW-H", "DROP?"]


def to_s(t: str) -> float:
    p = [float(x) for x in t.strip().split(":")]
    return p[0] * 3600 + p[1] * 60 + p[2] if len(p) == 3 else p[0] * 60 + p[1]


def hms(s: float) -> str:
    h, rem = divmod(abs(s), 3600)
    m, sec = divmod(rem, 60)
    sign = "-" if s < 0 else ""
    return f"{sign}{int(h):02d}:{int(m):02d}:{sec:04.1f}"


def overlap_s(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def load_gt(path: Path) -> list[tuple[float, float]]:
    raw = path.read_text(encoding="utf-8")
    gt = []
    if "music - " in raw or "music-" in raw:
        for line in raw.splitlines():
            m = re.match(r"music\s*-\s*(.+?)\s*-\s*(.+)", line.strip())
            if m:
                gt.append((to_s(m.group(1)), to_s(m.group(2))))
    else:
        for row in csv.DictReader(raw.splitlines()):
            if (row.get("label") or "").strip() in MUSIC_LABELS:
                gt.append((to_s(row["start"]), to_s(row["end"])))
    return gt


def load_rows(path: Path) -> list[dict]:
    out = []
    for r in csv.DictReader(path.open(encoding="utf-8")):
        if r["label"] == "EXT":
            continue
        out.append({
            "cat": r["label"],
            "start": float(r["start"]),
            "end": float(r["end"]),
            "song": r["song"] or r.get("gap_shazam_song", "") or "(heuristic-only)",
        })
    return out


def summarize(pairs: list[tuple[float, float]], label: str) -> None:
    """pairs = list of (signed Δstart, signed Δend)."""
    if not pairs:
        print(f"  {label:<10} n=0")
        return
    ds = [p[0] for p in pairs]
    de = [p[1] for p in pairs]
    n = len(pairs)
    print(f"  {label:<10} n={n:<3}"
          f"  Δstart med {statistics.median(ds):>+6.0f}s [{min(ds):>+5.0f},{max(ds):>+5.0f}]"
          f"  |med| {statistics.median(map(abs, ds)):>4.0f}s"
          f"   Δend med {statistics.median(de):>+6.0f}s [{min(de):>+5.0f},{max(de):>+5.0f}]"
          f"  |med| {statistics.median(map(abs, de)):>4.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ep", nargs=3, action="append", metavar=("NAME", "HYBRID_CSV", "GT_FILE"),
                    required=True, help="episode name, its hybrid_review CSV, its GT file")
    ap.add_argument("--min-overlap", type=float, default=MIN_OVERLAP)
    args = ap.parse_args()

    agg = defaultdict(list)      # category -> [(Δstart, Δend), ...] across all episodes

    for name, hybrid_csv, gt_file in args.ep:
        gt = load_gt(Path(gt_file))
        rows = load_rows(Path(hybrid_csv))

        print(f"\n{'='*104}\n  {name}   ({len(gt)} GT music cuts, {len(rows)} predicted regions)\n{'='*104}")

        per_cat = defaultdict(list)
        no_gt = []
        matched_gt = set()

        for r in rows:
            best, bo = None, 0.0
            for gi, (g0, g1) in enumerate(gt):
                o = overlap_s(r["start"], r["end"], g0, g1)
                if o > bo:
                    bo, best = o, gi
            if best is not None and bo >= args.min_overlap:
                g0, g1 = gt[best]
                d = (r["start"] - g0, r["end"] - g1)
                per_cat[r["cat"]].append(d)
                agg[r["cat"]].append(d)
                matched_gt.add(best)
            else:
                no_gt.append(r)

        print("  Signed boundary error (predicted - actual), by category:")
        for c in CATEGORY_ORDER:
            if per_cat[c]:
                summarize(per_cat[c], c)

        # REVIEW-H detail — the load-bearing question (direction of error)
        if per_cat["REVIEW-H"]:
            print("\n  REVIEW-H rows (detail — direction is what decides upstream vs downstream fix):")
            for r in rows:
                if r["cat"] != "REVIEW-H":
                    continue
                best, bo = None, 0.0
                for gi, (g0, g1) in enumerate(gt):
                    o = overlap_s(r["start"], r["end"], g0, g1)
                    if o > bo:
                        bo, best = o, gi
                if best is not None and bo >= args.min_overlap:
                    g0, g1 = gt[best]
                    print(f"    {hms(r['start']):>9}-{hms(r['end']):<9}  "
                          f"Δstart {r['start']-g0:>+5.0f}s  Δend {r['end']-g1:>+5.0f}s   {r['song'][:38]}")

        recall = len(matched_gt)
        print(f"\n  Combined recall (any category): {recall}/{len(gt)}")
        if no_gt:
            print(f"  Regions with NO GT overlap ({len(no_gt)}, potential FPs — not boundary-scored):")
            for r in no_gt:
                print(f"    [{r['cat']:<8}] {hms(r['start']):>9}-{hms(r['end']):<9}  {r['song'][:38]}")

    # ── Aggregate across all episodes ─────────────────────────────────────────
    print(f"\n{'='*104}\n  AGGREGATE across {len(args.ep)} episode(s) — signed error by category\n{'='*104}")
    for c in CATEGORY_ORDER:
        if agg[c]:
            summarize(agg[c], c)
    print()


if __name__ == "__main__":
    main()
