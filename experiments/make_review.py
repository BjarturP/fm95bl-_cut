"""
Generate a human-review package for one episode.

Three detection sources are combined into a single Audacity label file
and a unified review CSV:

  [H]                — heuristic finalcuts pipeline
  [S ✓/⚠]           — Shazam recognition on heuristic candidate regions
  [GAP-SHAZAM ✓/⚠]  — Shazam gap-scan (songs in gaps the heuristic missed)

Merging rules (in order):
  1. Start from heuristic finalcuts as the primary list.
  2. Annotate each heuristic candidate with the best overlapping normal-Shazam
     match (≥5s overlap).
  3. Add normal-Shazam-only rows for Shazam matches not overlapping any
     heuristic candidate.
  4. Annotate each row with the best overlapping gap-shazam match (≥30s).
  5. Add gap-shazam-only rows for gap matches not overlapping any row above.
  6. Sort everything by suggested_start.

Region type values:
  heuristic_only              — heuristic only
  heuristic+shazam            — heuristic + normal Shazam
  shazam_only                 — normal Shazam only (heuristic missed it)
  gap_shazam_only             — gap scan only (both others missed it)
  heuristic_only+gap_shazam   — heuristic + gap scan (no normal Shazam)
  heuristic+shazam+gap_shazam — all three agree
  shazam_only+gap_shazam      — both Shazam methods, no heuristic

Usage:
    python experiments/make_review.py \\
        --heuristic-csv    data/labels/<episode>_finalcuts_review.csv \\
        --shazam-json      data/labels/song_matches_<episode>.json \\
        --shazam-unmatched data/labels/song_unmatched_<episode>.csv \\
        --episode-name     <episode> \\
        [--gap-shazam-json data/labels/gap_shazam_matches_<episode>.json] \\
        [--out-dir         data/labels]
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


# Minimum overlap (seconds) to associate a gap-shazam match with an existing row
GAP_MERGE_OVERLAP = 30.0


def hms(s: float) -> str:
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{sec:05.2f}"


def overlap_s(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--heuristic-csv",     type=Path, required=True,
                        help="*_finalcuts_review.csv from review_export.py")
    parser.add_argument("--shazam-json",        type=Path, required=True,
                        help="song_matches_*.json from shazam_detect.py")
    parser.add_argument("--shazam-unmatched",   type=Path, required=True,
                        help="song_unmatched_*.csv from shazam_detect.py")
    parser.add_argument("--gap-shazam-json",    type=Path, default=None,
                        help="gap_shazam_matches_*.json from shazam_gap_scan.py (optional)")
    parser.add_argument("--episode-name",       required=True,
                        help="short name used in output filenames")
    parser.add_argument("--out-dir",            type=Path, default=Path("data/labels"))
    args = parser.parse_args()

    # ── Load all sources ──────────────────────────────────────────────────────

    heuristic: list[dict] = []
    with args.heuristic_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            heuristic.append({
                "start":      float(row["start"]),
                "end":        float(row["end"]),
                "confidence": float(row["confidence"]),
                "reasons":    row["reasons"],
            })

    shazam_matched: list[dict] = json.loads(
        args.shazam_json.read_text(encoding="utf-8")
    )

    shazam_unmatched: list[dict] = []
    with args.shazam_unmatched.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            shazam_unmatched.append({
                "start": float(row["region_start"]),
                "end":   float(row["region_end"]),
            })

    gap_shazam: list[dict] = []
    if args.gap_shazam_json and args.gap_shazam_json.exists():
        gap_shazam = json.loads(args.gap_shazam_json.read_text(encoding="utf-8"))
        print(f"Gap-shazam: {len(gap_shazam)} matches from {args.gap_shazam_json.name}")
    else:
        print("Gap-shazam: no file provided — running without gap-scan layer")

    # ── Stage 1: build rows from heuristic + normal Shazam ───────────────────

    used_shazam: set[int] = set()
    rows: list[dict] = []

    for h in heuristic:
        best_sm_i, best_sm_ov = None, 0.0
        for i, sm in enumerate(shazam_matched):
            ov = overlap_s(h["start"], h["end"], sm["start"], sm["end"])
            if ov > best_sm_ov:
                best_sm_ov = ov
                best_sm_i  = i

        if best_sm_i is not None and best_sm_ov >= 5.0:
            sm = shazam_matched[best_sm_i]
            used_shazam.add(best_sm_i)
            sm_status = "UNCERTAIN" if sm["uncertain"] else "ok"
            rows.append({
                "region_type":       "heuristic+shazam",
                "heuristic_start":   h["start"],
                "heuristic_end":     h["end"],
                "heuristic_conf":    h["confidence"],
                "matched_song":      sm.get("matched_song", ""),
                "shazam_offset":     sm.get("shazam_offset", ""),
                "duration_s":        sm.get("duration_s", ""),
                "duration_source":   sm.get("duration_source", ""),
                "shazam_status":     sm_status,
                "uncertain_reasons": "; ".join(sm.get("uncertain_reasons", [])),
                "suggested_start":   sm["suggested_start"] if not sm["uncertain"] else h["start"],
                "suggested_end":     sm["suggested_end"]   if not sm["uncertain"] else h["end"],
            })
        else:
            rows.append({
                "region_type":       "heuristic_only",
                "heuristic_start":   h["start"],
                "heuristic_end":     h["end"],
                "heuristic_conf":    h["confidence"],
                "matched_song":      "",
                "shazam_offset":     "",
                "duration_s":        "",
                "duration_source":   "",
                "shazam_status":     "unmatched",
                "uncertain_reasons": "",
                "suggested_start":   h["start"],
                "suggested_end":     h["end"],
            })

    # Shazam-only rows (no heuristic overlap)
    for i, sm in enumerate(shazam_matched):
        if i in used_shazam:
            continue
        sm_status = "UNCERTAIN" if sm["uncertain"] else "ok"
        rows.append({
            "region_type":       "shazam_only",
            "heuristic_start":   "",
            "heuristic_end":     "",
            "heuristic_conf":    "",
            "matched_song":      sm.get("matched_song", ""),
            "shazam_offset":     sm.get("shazam_offset", ""),
            "duration_s":        sm.get("duration_s", ""),
            "duration_source":   sm.get("duration_source", ""),
            "shazam_status":     sm_status,
            "uncertain_reasons": "; ".join(sm.get("uncertain_reasons", [])),
            "suggested_start":   sm["suggested_start"],
            "suggested_end":     sm["suggested_end"],
        })

    # ── Stage 2: annotate rows with gap-shazam matches ────────────────────────

    for row in rows:
        row.setdefault("gap_shazam_song",              "")
        row.setdefault("gap_shazam_start",             "")
        row.setdefault("gap_shazam_end",               "")
        row.setdefault("gap_shazam_status",            "")
        row.setdefault("gap_shazam_uncertain_reasons", "")

    used_gs: set[int] = set()

    for row in rows:
        r_s = float(row["suggested_start"]) if row["suggested_start"] != "" else 0.0
        r_e = float(row["suggested_end"])   if row["suggested_end"]   != "" else 0.0

        best_gs_i, best_gs_ov = None, 0.0
        for i, gs in enumerate(gap_shazam):
            if i in used_gs:
                continue
            ov = overlap_s(r_s, r_e, gs["suggested_start"], gs["suggested_end"])
            if ov > best_gs_ov:
                best_gs_ov = ov
                best_gs_i  = i

        if best_gs_i is not None and best_gs_ov >= GAP_MERGE_OVERLAP:
            gs = gap_shazam[best_gs_i]
            used_gs.add(best_gs_i)
            row["gap_shazam_song"]              = gs["matched_song"]
            row["gap_shazam_start"]             = gs["suggested_start"]
            row["gap_shazam_end"]               = gs["suggested_end"]
            row["gap_shazam_status"]            = "UNCERTAIN" if gs["uncertain"] else "ok"
            row["gap_shazam_uncertain_reasons"] = "; ".join(gs.get("uncertain_reasons", []))
            row["region_type"]                  = row["region_type"] + "+gap_shazam"

    # Stage 2b: standalone gap-shazam rows (not overlapping anything above)
    for i, gs in enumerate(gap_shazam):
        if i in used_gs:
            continue
        rows.append({
            "region_type":                 "gap_shazam_only",
            "heuristic_start":             "",
            "heuristic_end":               "",
            "heuristic_conf":              "",
            "matched_song":                "",
            "shazam_offset":               "",
            "duration_s":                  "",
            "duration_source":             "",
            "shazam_status":               "unmatched",
            "uncertain_reasons":           "",
            "suggested_start":             gs["suggested_start"],
            "suggested_end":               gs["suggested_end"],
            "gap_shazam_song":             gs["matched_song"],
            "gap_shazam_start":            gs["suggested_start"],
            "gap_shazam_end":              gs["suggested_end"],
            "gap_shazam_status":           "UNCERTAIN" if gs["uncertain"] else "ok",
            "gap_shazam_uncertain_reasons": "; ".join(gs.get("uncertain_reasons", [])),
        })

    rows.sort(key=lambda r: (
        float(r["suggested_start"]) if r["suggested_start"] != "" else 0.0
    ))

    # ── Write review sheet CSV ────────────────────────────────────────────────

    args.out_dir.mkdir(parents=True, exist_ok=True)
    review_csv = args.out_dir / f"review_sheet_{args.episode_name}.csv"

    fieldnames = [
        "row", "region_type",
        "suggested_start", "suggested_end",
        "suggested_start_hms", "suggested_end_hms",
        # Normal Shazam columns
        "matched_song", "shazam_status", "uncertain_reasons",
        "shazam_offset_s", "duration_s", "duration_source",
        # Heuristic columns
        "heuristic_start", "heuristic_end", "heuristic_conf",
        # Gap-shazam columns
        "gap_shazam_song", "gap_shazam_start", "gap_shazam_end",
        "gap_shazam_status", "gap_shazam_uncertain_reasons",
        # Fill these in during review
        "actual_start", "actual_end",
        "actual_start_hms", "actual_end_hms",
        "label", "notes",
    ]

    with review_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for idx, r in enumerate(rows):
            ss = float(r["suggested_start"]) if r["suggested_start"] != "" else 0.0
            se = float(r["suggested_end"])   if r["suggested_end"]   != "" else 0.0
            writer.writerow({
                "row":                         idx + 1,
                "region_type":                 r["region_type"],
                "suggested_start":             r["suggested_start"],
                "suggested_end":               r["suggested_end"],
                "suggested_start_hms":         hms(ss),
                "suggested_end_hms":           hms(se),
                "matched_song":                r.get("matched_song", ""),
                "shazam_status":               r.get("shazam_status", ""),
                "uncertain_reasons":           r.get("uncertain_reasons", ""),
                "shazam_offset_s":             r.get("shazam_offset", ""),
                "duration_s":                  r.get("duration_s", ""),
                "duration_source":             r.get("duration_source", ""),
                "heuristic_start":             r.get("heuristic_start", ""),
                "heuristic_end":               r.get("heuristic_end", ""),
                "heuristic_conf":              r.get("heuristic_conf", ""),
                "gap_shazam_song":             r.get("gap_shazam_song", ""),
                "gap_shazam_start":            r.get("gap_shazam_start", ""),
                "gap_shazam_end":              r.get("gap_shazam_end", ""),
                "gap_shazam_status":           r.get("gap_shazam_status", ""),
                "gap_shazam_uncertain_reasons": r.get("gap_shazam_uncertain_reasons", ""),
                "actual_start":    "", "actual_end":     "",
                "actual_start_hms":"","actual_end_hms": "",
                "label": "", "notes": "",
            })

    # ── Write combined Audacity label file ────────────────────────────────────

    aud = args.out_dir / f"review_audacity_{args.episode_name}.txt"
    lines: list[tuple] = []

    for h in heuristic:
        lines.append((h["start"], h["end"], f"[H conf={h['confidence']:.1f}]"))

    for sm in shazam_matched:
        flag  = " ⚠" if sm["uncertain"] else " ✓"
        lines.append((
            sm["suggested_start"], sm["suggested_end"],
            f"[S{flag}] {sm.get('matched_song', '')}",
        ))

    for gs in gap_shazam:
        flag  = " ⚠" if gs["uncertain"] else " ✓"
        lines.append((
            gs["suggested_start"], gs["suggested_end"],
            f"[GAP-SHAZAM{flag}] {gs.get('matched_song', '')}",
        ))

    lines.sort(key=lambda x: x[0])
    with aud.open("w", encoding="utf-8") as f:
        for start, end, label in lines:
            f.write(f"{start:.3f}\t{end:.3f}\t{label}\n")

    # ── Print summary ─────────────────────────────────────────────────────────

    n_gs_merged     = len(used_gs)
    n_gs_standalone = len(gap_shazam) - n_gs_merged
    rt_counts       = Counter(r["region_type"] for r in rows)

    print(f"\nReview package: {args.episode_name}")
    print(f"  Heuristic candidates :  {len(heuristic)}")
    print(f"  Normal Shazam matches:  {len(shazam_matched)}")
    if gap_shazam:
        print(f"  Gap-Shazam matches  :  {len(gap_shazam)}")
        print(f"    merged with existing row : {n_gs_merged}")
        print(f"    new standalone row       : {n_gs_standalone}")
    print(f"  Total review rows    :  {len(rows)}")
    print(f"\n  Region types:")
    for rt, cnt in sorted(rt_counts.items()):
        print(f"    {rt:<40} {cnt}")

    print(f"\nOutputs:")
    print(f"  {review_csv}")
    print(f"  {aud}")

    print(f"""
Label guide for Audacity:
  [H ...]                = heuristic pipeline candidate
  [S ✓ ...]             = Shazam-matched (boundaries ok)
  [S ⚠ ...]             = Shazam-matched (boundary uncertain)
  [GAP-SHAZAM ✓ ...]    = gap-scan match (ok)
  [GAP-SHAZAM ⚠ ...]    = gap-scan match (uncertain)

region_type in CSV:
  heuristic_only              — pipeline only, no Shazam
  heuristic+shazam            — pipeline + normal Shazam
  shazam_only                 — Shazam found it, pipeline missed
  gap_shazam_only             — gap scan only (review carefully — new find)
  *+gap_shazam                — gap scan corroborates/extends an existing row
""")


if __name__ == "__main__":
    main()
