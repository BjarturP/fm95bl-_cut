"""
Generate a human-review package for one episode.

Three detection sources are combined into a single Audacity label file
and a unified review CSV:

  [H conf=X.X]       — heuristic finalcuts pipeline
  [S ✓/⚠] Song       — Shazam recognition on heuristic candidate regions
  [EXT ✓/⚠] Song     — gap-scan match that is a tail of an existing candidate
                         (song started before the gap, boundary ended too early)
  [GAP ✓/⚠] Song     — gap-scan match that is entirely new (heuristic missed it)

EXT vs GAP classification: if song_start_in_episode < gap_start → EXT (tail);
otherwise → GAP (new find).

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
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


# Minimum overlap (seconds) to associate a gap-shazam match with an existing row
GAP_MERGE_OVERLAP = 30.0
# Maximum gap between same-song regions to consider merging them
SAME_SONG_MAX_GAP = 120.0
# Minimum uncovered chunk of a Shazam-narrowed heuristic region to surface
# as a region_remainder review row (may hide a second, unmatched song)
REMAINDER_MIN_DUR = 180.0


def hms(s: float) -> str:
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{sec:05.2f}"


def overlap_s(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _song_name(row: dict) -> str:
    return (row.get("matched_song") or "").lower().strip()


def gap_last_confirmed(gs: dict) -> float | str:
    """
    Last episode time at which the gap scan positively heard the song:
    song_start + last matched window offset + clip length. Unlike
    suggested_end (song_start + DB duration), this cannot overshoot into
    host talk when the DB duration is the album edit — it errs short.
    """
    offs = gs.get("all_offsets") or []
    if not offs or gs.get("song_start_in_episode") in (None, ""):
        return ""
    clip_len = 15.0
    try:
        cl = float(gs.get("clip_end", 0)) - float(gs.get("clip_start", 0))
        if cl > 0:
            clip_len = cl
    except (TypeError, ValueError):
        pass
    return round(float(gs["song_start_in_episode"]) + max(offs) + clip_len, 1)


def merge_same_song_rows(rows: list[dict], max_gap_s: float = SAME_SONG_MAX_GAP) -> list[dict]:
    """
    Merge adjacent rows that Shazam identified as the same song.
    Requires: same matched_song, gap between suggested_end of row-i and
    suggested_start of row-j ≤ max_gap_s.

    The merged row keeps the earliest start and latest end, combines
    heuristic regions, and adds a merge_note.
    """
    if not rows:
        return rows

    sorted_rows = sorted(
        rows,
        key=lambda r: float(r["suggested_start"]) if r["suggested_start"] != "" else 0.0,
    )

    result: list[dict] = []
    skip: set[int] = set()

    for i, r1 in enumerate(sorted_rows):
        if i in skip:
            continue

        merged = {**r1, "merge_note": ""}
        song1  = _song_name(r1)

        if not song1:
            result.append(merged)
            continue

        partners: list[int] = []
        merged_end = float(merged["suggested_end"]) if merged["suggested_end"] != "" else 0.0

        for j in range(i + 1, len(sorted_rows)):
            if j in skip:
                continue
            r2 = sorted_rows[j]
            if _song_name(r2) != song1:
                continue

            r2_start = float(r2["suggested_start"]) if r2["suggested_start"] != "" else 0.0
            gap = r2_start - merged_end
            if gap > max_gap_s:
                continue

            # Same song within gap threshold — merge
            partners.append(j)
            skip.add(j)

            r1_h_start = float(merged["heuristic_start"]) if merged.get("heuristic_start") else float(merged["suggested_start"])
            r2_h_start = float(r2.get("heuristic_start") or r2["suggested_start"])
            r1_h_end   = float(merged["heuristic_end"])   if merged.get("heuristic_end")   else float(merged["suggested_end"])
            r2_h_end   = float(r2.get("heuristic_end") or r2["suggested_end"])

            merged["suggested_start"] = str(min(float(merged["suggested_start"]), float(r2["suggested_start"])))
            merged["suggested_end"]   = str(max(float(merged["suggested_end"]),   float(r2["suggested_end"])))
            merged["heuristic_start"] = str(min(r1_h_start, r2_h_start))
            merged["heuristic_end"]   = str(max(r1_h_end,   r2_h_end))
            merged["region_type"]     = "merged_same_song"

            # Escalate uncertainty if either is uncertain
            if r2.get("shazam_status") == "UNCERTAIN":
                merged["shazam_status"] = "UNCERTAIN"
            extra_reasons = r2.get("uncertain_reasons", "")
            if extra_reasons:
                existing = merged.get("uncertain_reasons", "")
                merged["uncertain_reasons"] = (existing + "; " + extra_reasons).strip("; ")
            # Aggregate hits; take best confidence (HIGH > MEDIUM > LOW)
            conf_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "": 0}
            c1 = merged.get("boundary_confidence", "") or ""
            c2 = r2.get("boundary_confidence", "") or ""
            merged["boundary_confidence"] = c1 if conf_rank.get(c1, 0) >= conf_rank.get(c2, 0) else c2
            try:
                merged["n_samples_matched"] = int(float(merged.get("n_samples_matched") or 0)) + int(float(r2.get("n_samples_matched") or 0))
            except (TypeError, ValueError):
                pass

            merged_end = float(merged["suggested_end"])

        if partners:
            gap_between = float(sorted_rows[partners[0]]["suggested_start"]) - float(r1["suggested_end"])
            merged["merge_note"] = (
                f"merged {1 + len(partners)} adjacent same-song regions "
                f"(gap {gap_between:.0f}s — possible DJ talk)"
            )

        result.append(merged)

    return result


def evaluate_against_gt(
    rows: list[dict],
    gt_file: Path,
    heuristic: list[dict],
) -> None:
    """Compare suggested cuts against known ground-truth cuts."""

    def _to_s(t: str) -> float:
        parts = t.strip().split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        return int(parts[0]) * 60 + float(parts[1])

    gt: list[dict] = []
    with gt_file.open(encoding="utf-8") as f:
        raw = f.read()

    # Format A: "music - MM:SS - MM:SS" (one per line)
    if "music - " in raw:
        for line in raw.splitlines():
            m = re.match(r"music - (.+?) - (.+)", line.strip())
            if m:
                gt.append({"start": _to_s(m.group(1)), "end": _to_s(m.group(2))})
    # Format B: CSV with header "start,end,label"
    else:
        import csv as _csv
        for row in _csv.DictReader(raw.splitlines()):
            lbl = (row.get("label") or "").strip()
            if lbl in ("music", "music/ads"):
                try:
                    gt.append({"start": _to_s(row["start"]), "end": _to_s(row["end"])})
                except (KeyError, ValueError):
                    pass

    if not gt:
        print("  (no GT cuts parsed — check file format)")
        return

    suggested = [
        {
            "start": float(r["suggested_start"]),
            "end":   float(r["suggested_end"]),
            "song":  r.get("matched_song", "") or "",
            "type":  r.get("region_type", ""),
        }
        for r in rows
        if r.get("suggested_start") not in ("", None)
    ]

    print(f"\n{'='*88}")
    print(f"Ground-truth evaluation: {gt_file.name}")
    print(f"{'='*88}")
    print(
        f"  {'#':>2}  {'GT start':>8} {'GT end':>6} {'dur':>4}   "
        f"{'S start':>8} {'S end':>6} {'dur':>4}   {'Δstart':>7} {'Δend':>7}  Song / Note"
    )
    print(f"  {'─'*86}")

    total_se = total_ee = n_matched = 0

    for gi, g in enumerate(gt):
        gdur = g["end"] - g["start"]

        best, best_ov = None, 0.0
        for s in suggested:
            ov = overlap_s(g["start"], g["end"], s["start"], s["end"])
            if ov > best_ov:
                best_ov, best = ov, s

        gs = hms(g["start"])
        ge = hms(g["end"])

        if best and best_ov >= 30.0:
            ds   = best["start"] - g["start"]
            de   = best["end"]   - g["end"]
            sdur = best["end"]   - best["start"]
            flag = " ⚠" if abs(ds) > 30 or abs(de) > 30 else "  "
            song = (best["song"] or best["type"])[:36]
            print(
                f"  {gi+1:>2}  {gs:>8} {ge:>6} {gdur:>4.0f}s  "
                f"{hms(best['start']):>8} {hms(best['end']):>6} {sdur:>4.0f}s  "
                f"{ds:>+7.0f}s {de:>+7.0f}s  {song}{flag}"
            )
            total_se += abs(ds)
            total_ee += abs(de)
            n_matched += 1
        else:
            # Shazam missed it — check heuristic fallback
            h_note = ""
            for h in heuristic:
                if overlap_s(g["start"], g["end"], h["start"], h["end"]) >= 30.0:
                    h_note = f"[H conf={h['confidence']:.1f}] {hms(h['start'])}–{hms(h['end'])} — Shazam unmatched (Icelandic?)"
                    break
            if not h_note:
                h_note = "BLIND SPOT — not detected by heuristic or Shazam"
            print(
                f"  {gi+1:>2}  {gs:>8} {ge:>6} {gdur:>4.0f}s  "
                f"  {'—':>8} {'—':>6} {'—':>4}  {'':>7}  {'':>7}  {h_note}"
            )

    print(f"\n  Matched {n_matched}/{len(gt)} GT cuts")
    if n_matched:
        print(f"  Mean |Δstart| = {total_se/n_matched:.0f}s")
        print(f"  Mean |Δend|   = {total_ee/n_matched:.0f}s")
    print(f"{'='*88}\n")


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
    parser.add_argument("--gt-file",            type=Path, default=None,
                        help="ground-truth cut file (format: 'music - MM:SS - MM:SS')")
    parser.add_argument("--same-song-max-gap",  type=float, default=SAME_SONG_MAX_GAP,
                        help="max gap (s) between adjacent same-song regions to merge (default 120)")
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
                "region_type":          "heuristic+shazam",
                "heuristic_start":      h["start"],
                "heuristic_end":        h["end"],
                "heuristic_conf":       h["confidence"],
                "matched_song":         sm.get("matched_song", ""),
                "shazam_offset":        sm.get("shazam_offset", ""),
                "duration_s":           sm.get("duration_s", ""),
                "duration_source":      sm.get("duration_source", ""),
                "shazam_status":        sm_status,
                "uncertain_reasons":    "; ".join(sm.get("uncertain_reasons", [])),
                "n_samples_matched":    sm.get("n_samples_matched", ""),
                "boundary_confidence":  sm.get("boundary_confidence", ""),
                "start_confidence":     sm.get("start_confidence", ""),
                "end_confidence":       sm.get("end_confidence", ""),
                "start_uncertain":      sm.get("start_uncertain", ""),
                "end_uncertain":        sm.get("end_uncertain", ""),
                "end_info":             sm.get("end_info", ""),
                "suggested_start":      sm["suggested_start"],
                "suggested_end":        sm["suggested_end"],
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
            "region_type":         "shazam_only",
            "heuristic_start":     "",
            "heuristic_end":       "",
            "heuristic_conf":      "",
            "matched_song":        sm.get("matched_song", ""),
            "shazam_offset":       sm.get("shazam_offset", ""),
            "duration_s":          sm.get("duration_s", ""),
            "duration_source":     sm.get("duration_source", ""),
            "shazam_status":       sm_status,
            "uncertain_reasons":   "; ".join(sm.get("uncertain_reasons", [])),
            "n_samples_matched":   sm.get("n_samples_matched", ""),
            "boundary_confidence": sm.get("boundary_confidence", ""),
            "start_confidence":    sm.get("start_confidence", ""),
            "end_confidence":      sm.get("end_confidence", ""),
            "start_uncertain":     sm.get("start_uncertain", ""),
            "end_uncertain":       sm.get("end_uncertain", ""),
            "end_info":            sm.get("end_info", ""),
            "suggested_start":     sm["suggested_start"],
            "suggested_end":       sm["suggested_end"],
        })

    # ── Stage 2: annotate rows with gap-shazam matches ────────────────────────

    for row in rows:
        row.setdefault("gap_shazam_song",              "")
        row.setdefault("gap_shazam_start",             "")
        row.setdefault("gap_shazam_end",               "")
        row.setdefault("gap_shazam_status",            "")
        row.setdefault("gap_shazam_type",              "")
        row.setdefault("gap_shazam_n_hits",            "")
        row.setdefault("gap_shazam_uncertain_reasons", "")
        row.setdefault("gap_shazam_last_confirmed",    "")

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
            gs_type = "extension" if gs.get("song_start_in_episode", gs["gap_start"]) < gs["gap_start"] else "new_gap_find"
            row["gap_shazam_song"]              = gs["matched_song"]
            row["gap_shazam_start"]             = gs["suggested_start"]
            row["gap_shazam_end"]               = gs["suggested_end"]
            row["gap_shazam_status"]            = "UNCERTAIN" if gs["uncertain"] else "ok"
            row["gap_shazam_type"]              = gs_type
            row["gap_shazam_n_hits"]            = gs.get("n_windows_matched", "")
            row["gap_shazam_uncertain_reasons"] = "; ".join(gs.get("uncertain_reasons", []))
            row["gap_shazam_last_confirmed"]    = gap_last_confirmed(gs)
            row["region_type"]                  = row["region_type"] + "+gap_shazam"

    # Stage 2b: standalone gap-shazam rows (not overlapping anything above)
    for i, gs in enumerate(gap_shazam):
        if i in used_gs:
            continue
        gs_type = "extension" if gs.get("song_start_in_episode", gs["gap_start"]) < gs["gap_start"] else "new_gap_find"
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
            "gap_shazam_type":             gs_type,
            "gap_shazam_n_hits":           gs.get("n_windows_matched", ""),
            "gap_shazam_uncertain_reasons": "; ".join(gs.get("uncertain_reasons", [])),
            "gap_shazam_last_confirmed":   gap_last_confirmed(gs),
        })

    # Stage 2c: region-remainder rows. When Shazam narrows a wide heuristic
    # region to just the matched song, the rest of the region silently vanishes
    # from every output — and can hide a second (often Icelandic, unmatchable)
    # song. Surface any uncovered chunk >= REMAINDER_MIN_DUR as its own
    # review row so it stays visible in Audacity. These rows have no Shazam
    # match, so they classify as REVIEW-H and can never touch AUTO precision.
    covered = []
    for row in rows:
        if row["suggested_start"] != "" and row["suggested_end"] != "":
            covered.append((float(row["suggested_start"]), float(row["suggested_end"])))
        if row["gap_shazam_start"] != "" and row["gap_shazam_end"] != "":
            covered.append((float(row["gap_shazam_start"]), float(row["gap_shazam_end"])))

    remainder_rows: list[dict] = []
    for row in rows:
        if not (row.get("heuristic_start") and row.get("heuristic_end")):
            continue
        if not (row.get("matched_song") or "").strip():
            continue  # nothing was narrowed away
        segs = [(float(row["heuristic_start"]), float(row["heuristic_end"]))]
        for cs, ce in covered:
            nxt = []
            for s0, s1 in segs:
                if cs < s1 and ce > s0:
                    if cs - s0 > 1:
                        nxt.append((s0, cs))
                    if s1 - ce > 1:
                        nxt.append((ce, s1))
                else:
                    nxt.append((s0, s1))
            segs = nxt
        for s0, s1 in segs:
            if s1 - s0 < REMAINDER_MIN_DUR:
                continue
            remainder_rows.append({
                "region_type":       "region_remainder",
                "heuristic_start":   row["heuristic_start"],
                "heuristic_end":     row["heuristic_end"],
                "heuristic_conf":    row.get("heuristic_conf", ""),
                "matched_song":      "",
                "shazam_status":     "unmatched",
                "uncertain_reasons": "",
                "suggested_start":   round(s0, 1),
                "suggested_end":     round(s1, 1),
                "gap_shazam_song":   "", "gap_shazam_start": "", "gap_shazam_end": "",
                "gap_shazam_status": "", "gap_shazam_type": "", "gap_shazam_n_hits": "",
                "gap_shazam_uncertain_reasons": "", "gap_shazam_last_confirmed": "",
            })
    if remainder_rows:
        print(f"\nRegion remainders: {len(remainder_rows)} uncovered chunk(s) >= "
              f"{REMAINDER_MIN_DUR:.0f}s surfaced for review")
        for r in remainder_rows:
            print(f"  remainder {hms(float(r['suggested_start']))}–{hms(float(r['suggested_end']))}"
                  f"  (from region {r['heuristic_start']}–{r['heuristic_end']})")
    rows.extend(remainder_rows)

    rows.sort(key=lambda r: (
        float(r["suggested_start"]) if r["suggested_start"] != "" else 0.0
    ))

    # ── Stage 3: merge adjacent rows with the same Shazam song ───────────────
    before_merge = len(rows)
    rows = merge_same_song_rows(rows, max_gap_s=args.same_song_max_gap)
    n_merged = before_merge - len(rows)
    if n_merged:
        print(f"\nSame-song merge: {n_merged} row(s) absorbed into adjacent same-song rows")
        for r in rows:
            if r.get("region_type") == "merged_same_song":
                ss = float(r["suggested_start"])
                se = float(r["suggested_end"])
                print(f"  merged → {r['matched_song'][:45]}  {hms(ss)}–{hms(se)}"
                      f"  [{r.get('merge_note','')}]")

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
        "n_samples_matched", "boundary_confidence",
        "start_confidence", "end_confidence", "start_uncertain", "end_uncertain", "end_info",
        # Heuristic columns
        "heuristic_start", "heuristic_end", "heuristic_conf",
        # Gap-shazam columns
        "gap_shazam_song", "gap_shazam_start", "gap_shazam_end",
        "gap_shazam_status", "gap_shazam_type", "gap_shazam_n_hits", "gap_shazam_uncertain_reasons",
        "gap_shazam_last_confirmed",
        # Merge info
        "merge_note",
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
                "n_samples_matched":           r.get("n_samples_matched", ""),
                "boundary_confidence":         r.get("boundary_confidence", ""),
                "start_confidence":            r.get("start_confidence", ""),
                "end_confidence":              r.get("end_confidence", ""),
                "start_uncertain":             r.get("start_uncertain", ""),
                "end_uncertain":               r.get("end_uncertain", ""),
                "end_info":                    r.get("end_info", ""),
                "heuristic_start":             r.get("heuristic_start", ""),
                "heuristic_end":               r.get("heuristic_end", ""),
                "heuristic_conf":              r.get("heuristic_conf", ""),
                "gap_shazam_song":             r.get("gap_shazam_song", ""),
                "gap_shazam_start":            r.get("gap_shazam_start", ""),
                "gap_shazam_end":              r.get("gap_shazam_end", ""),
                "gap_shazam_status":           r.get("gap_shazam_status", ""),
                "gap_shazam_type":             r.get("gap_shazam_type", ""),
                "gap_shazam_n_hits":           r.get("gap_shazam_n_hits", ""),
                "gap_shazam_uncertain_reasons": r.get("gap_shazam_uncertain_reasons", ""),
                "gap_shazam_last_confirmed":   r.get("gap_shazam_last_confirmed", ""),
                "merge_note":          r.get("merge_note", ""),
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
        # Skip impossible matches (offset > song duration → reversed boundaries)
        ss, se = sm["suggested_start"], sm["suggested_end"]
        if se <= ss:
            continue
        flag = " ⚠" if sm["uncertain"] else " ✓"
        lines.append((ss, se, f"[S{flag}] {sm.get('matched_song', '')}",))

    for gs in gap_shazam:
        flag = " ⚠" if gs["uncertain"] else " ✓"
        is_ext = gs.get("song_start_in_episode", gs["gap_start"]) < gs["gap_start"]
        prefix = "EXT" if is_ext else "GAP"
        lines.append((
            gs["suggested_start"], gs["suggested_end"],
            f"[{prefix}{flag}] {gs.get('matched_song', '')}",
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

    if args.gt_file and args.gt_file.exists():
        evaluate_against_gt(rows, args.gt_file, heuristic)

    print(f"""
Label guide for Audacity:
  [H conf=X.X]          = heuristic pipeline candidate
  [S ✓] Song            = Shazam-matched (boundaries ok)
  [S ⚠] Song            = Shazam-matched (boundary uncertain)
  [EXT ✓/⚠] Song        = gap-scan: song tail extending past heuristic boundary
  [GAP ✓/⚠] Song        = gap-scan: entirely missed song inside a gap

gap_shazam_type in CSV:
  extension     — song started before gap; heuristic cut off too early
  new_gap_find  — song started inside gap; heuristic missed it entirely

region_type in CSV:
  heuristic_only              — pipeline only, no Shazam
  heuristic+shazam            — pipeline + normal Shazam
  shazam_only                 — Shazam found it, pipeline missed
  gap_shazam_only             — gap scan only (review carefully — new find)
  *+gap_shazam                — gap scan corroborates/extends an existing row
""")

    # ── Auto-run hybrid classification ────────────────────────────────────────
    hybrid_script = Path(__file__).parent / "hybrid_review.py"
    subprocess.run(
        [sys.executable, str(hybrid_script),
         "--review-csv",   str(review_csv),
         "--episode-name", args.episode_name,
         "--out-dir",      str(args.out_dir)],
        check=True,
    )


if __name__ == "__main__":
    main()
