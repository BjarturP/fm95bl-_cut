"""
Hybrid review workflow — classify episode cuts into action categories.

Reads the review_sheet CSV produced by make_review.py and classifies each row:

  [AUTO]     — High-confidence Shazam: safe to cut without manual review
  [REVIEW-S] — Shazam matched but boundary uncertain: check in Audacity
  [REVIEW-B] — Shazam matched, song real, but start is well before the
                heuristic region — may include host talk or ads over intro;
                verify the start boundary before cutting
  [REVIEW-H] — Heuristic-only candidate: unknown song, possible Icelandic,
                ad, or jingle — needs manual review
  [EXT]      — Gap-scan extension: suggested longer end for existing cut
  [DROP?]    — Short or weak candidate: quick sanity check or ignore

Generates:
  hybrid_review_audacity_<name>.txt     — import into Audacity
  hybrid_review_<name>.csv             — all rows, all fields
  hybrid_auto_cuts_<name>.csv          — [AUTO] rows only, ready-to-cut
  hybrid_manual_review_<name>.csv      — [REVIEW-S] + [REVIEW-H] + [EXT] rows

Usage:
    python experiments/hybrid_review.py \\
        --review-csv  data/labels/review_sheet_<ep>.csv \\
        --episode-name <ep>

Run make_review.py first to produce review_sheet_<ep>.csv.
"""

import argparse
import csv
import json
from pathlib import Path


# ── Classification thresholds ─────────────────────────────────────────────────

AUTO_MIN_HITS         = 2      # minimum Shazam hits for [AUTO]
AUTO_MIN_DURATION     = 60.0   # minimum cut duration (s) for [AUTO]
EARLY_START_THRESHOLD = 30.0   # if AUTO cut starts >Xs before heuristic region → [REVIEW-B]
DROP_HEURISTIC_DUR    = 60.0   # heuristic-only cuts shorter than this → [DROP?]
DROP_SHAZAM_DUR       = 30.0   # Shazam cuts shorter than this → [DROP?]
SHAZAM_CONFUSED_DUR   = 45.0   # Shazam cut < this AND much shorter than heuristic region
SHAZAM_CONFUSED_RATIO = 3.0    # ...when heuristic region > Shazam cut * this ratio
TAIL_EXT_MARGIN       = 5.0    # pull back this far from last scan-confirmed music when
                               # auto-applying a same-song tail extension (errs short)

# AUTO cuts are DELETED without review, so both edges must err toward leaving
# music in, never toward eating host talk. shazam_detect/gap_scan pad every
# suggested cut outward (PRE_PADDING=3 before the song, POST_PADDING=5 after)
# — right for human review, wrong for unreviewed deletion. AUTO rows undo the
# padding and add a small inward safety margin on top.
AUTO_START_TRIM = 3.0 + 2.0    # undo PRE_PADDING, then 2s inward safety
AUTO_END_TRIM   = 5.0 + 3.0    # undo POST_PADDING, then 3s inward safety

# Word-guard: transcribed host speech inside an AUTO cut near its edge pulls
# the edge inward past the speech. Catches the Shazam-estimate outliers the
# fixed trim can't (e.g. a start 45s early inside DJ talk).
GUARD_NSP_MAX  = 0.5   # segments with no_speech_prob >= this are sung/garbled, ignored
GUARD_WIN      = 45.0  # how far inside the cut edge to look for host speech
GUARD_EDGE     = 3.0   # speech must start/end within this of the edge to count as "at the edge"
GUARD_GAP      = 4.0   # wordless gap that marks where the speech stops and music alone begins
GUARD_PAD      = 1.0   # land this far inside the wordless gap


def load_speech_words(transcript_path: Path) -> list[tuple[float, float]]:
    """(start, end) of transcript words from speech-like segments, sorted."""
    with transcript_path.open(encoding="utf-8") as f:
        data = json.load(f)
    words: list[tuple[float, float]] = []
    for seg in data.get("segments", []):
        if seg.get("no_speech_prob", 0.0) >= GUARD_NSP_MAX:
            continue
        for w in seg.get("words", []):
            words.append((float(w["start"]), float(w["end"])))
    words.sort()
    return words


def word_guard(cut_s: float, cut_e: float,
               words: list[tuple[float, float]]) -> tuple[float, float, str]:
    """
    Pull AUTO cut edges inward past any host speech sitting at the edge.
    Returns (new_start, new_end, note). Moves are bounded by GUARD_WIN; if
    speech runs deeper than that with no wordless gap, the edge is left alone
    (caller keeps the row AUTO — the trim already applied — but the note says
    the guard could not anchor).
    """
    notes = []

    # START edge: speech overlapping [cut_s, cut_s+WIN] that begins at the edge
    in_win = [w for w in words if w[1] > cut_s and w[0] < cut_s + GUARD_WIN]
    if in_win and in_win[0][0] <= cut_s + GUARD_EDGE:
        anchor = None
        for a, b in zip(in_win, in_win[1:]):
            if b[0] - a[1] >= GUARD_GAP:
                anchor = a[1]
                break
        if anchor is None and in_win[-1][1] < cut_s + GUARD_WIN - GUARD_GAP:
            anchor = in_win[-1][1]          # speech ends, then silence to window edge
        if anchor is not None:
            notes.append(f"start guarded +{anchor + GUARD_PAD - cut_s:.0f}s (speech at edge)")
            cut_s = anchor + GUARD_PAD
        else:
            notes.append("start guard: speech at edge but no anchor within window")

    # END edge: speech overlapping [cut_e-WIN, cut_e] that runs to the edge
    in_win = [w for w in words if w[0] < cut_e and w[1] > cut_e - GUARD_WIN]
    if in_win and in_win[-1][1] >= cut_e - GUARD_EDGE:
        anchor = None
        for a, b in zip(reversed(in_win[:-1]), reversed(in_win[1:])):
            if b[0] - a[1] >= GUARD_GAP:
                anchor = b[0]
                break
        if anchor is None and in_win[0][0] > cut_e - GUARD_WIN + GUARD_GAP:
            anchor = in_win[0][0]
        if anchor is not None:
            notes.append(f"end guarded -{cut_e - (anchor - GUARD_PAD):.0f}s (speech at edge)")
            cut_e = anchor - GUARD_PAD
        else:
            notes.append("end guard: speech at edge but no anchor within window")

    return cut_s, cut_e, "; ".join(notes)


def hms(s: float) -> str:
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{sec:05.2f}"


def _float(v, default=0.0) -> float:
    try:
        return float(v) if v not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _int(v, default=0) -> int:
    try:
        return int(float(v)) if v not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _bool(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def classify(row: dict) -> tuple[str, str, float, float]:
    """
    Return (label, reason, cut_start, cut_end).

    label   : one of AUTO / REVIEW-S / REVIEW-H / EXT / DROP?
    reason  : short machine-readable tag
    cut_start/end: the boundaries to use in the Audacity label
    """
    matched_song   = (row.get("matched_song") or "").strip()
    shazam_status  = (row.get("shazam_status") or "").strip()
    uncertain      = shazam_status == "UNCERTAIN"
    n_hits         = _int(row.get("n_samples_matched"))
    confidence     = (row.get("boundary_confidence") or "LOW").strip()
    region_type    = (row.get("region_type") or "").strip()

    # Split-confidence fields (present since the 2026-07 recall fix). A DB-duration
    # overshoot is an END-only signal and must not veto a well-clustered START, so
    # AUTO now gates on start_confidence; a genuinely bad END → REVIEW-S, not AUTO.
    start_conf   = (row.get("start_confidence") or "").strip()
    have_split   = start_conf != ""
    start_unc    = _bool(row.get("start_uncertain"))
    end_unc      = _bool(row.get("end_uncertain"))

    sugg_start = _float(row.get("suggested_start"))
    sugg_end   = _float(row.get("suggested_end"))
    cut_dur    = sugg_end - sugg_start

    h_start = _float(row.get("heuristic_start"), default=sugg_start)
    h_end   = _float(row.get("heuristic_end"),   default=sugg_end)
    h_dur   = h_end - h_start

    # ── No normal Shazam match ────────────────────────────────────────────────
    if not matched_song:
        # Gap-scan-only find: a song Shazam matched entirely inside a heuristic
        # gap (both the heuristic and the normal Shazam pass missed it). Classify
        # on the gap match itself, gated the same way normal Shazam AUTO is, using
        # the gap-scan boundaries. Without this these real songs fall through to
        # "REVIEW-H (no match)" and their identified title is thrown away.
        gs_song = (row.get("gap_shazam_song") or "").strip()
        gs_type = (row.get("gap_shazam_type") or "").strip()
        if gs_song and gs_type == "new_gap_find":
            gs_s   = _float(row.get("gap_shazam_start"))
            gs_e   = _float(row.get("gap_shazam_end"))
            gs_unc = (row.get("gap_shazam_status") or "").strip() == "UNCERTAIN"
            gs_hits = _int(row.get("gap_shazam_n_hits"))
            # END floor: the last scan-confirmed music beats the DB-duration
            # projection when the DB has a bogus short duration (e.g. a 29s
            # "duration" for Pitbull capped the cut at 37s and blocked AUTO).
            # Safe: every second up to last_confirmed was positively heard as
            # this song by the scan windows.
            gs_last_conf = _float(row.get("gap_shazam_last_confirmed"))
            if gs_hits >= 2 and gs_last_conf - TAIL_EXT_MARGIN > gs_e:
                gs_e = gs_last_conf - TAIL_EXT_MARGIN
            gs_dur = gs_e - gs_s
            if gs_dur < DROP_SHAZAM_DUR:
                return "DROP?", "gap_shazam_too_short", gs_s, gs_e
            if (not gs_unc and gs_hits >= AUTO_MIN_HITS and gs_dur >= AUTO_MIN_DURATION):
                return "AUTO", "gap_shazam_verified", gs_s, gs_e
            return "REVIEW-S", "gap_shazam_uncertain", gs_s, gs_e

        # heuristic-only path
        if cut_dur < DROP_HEURISTIC_DUR:
            return "DROP?", "short_heuristic", sugg_start, sugg_end
        return "REVIEW-H", "heuristic_only", sugg_start, sugg_end

    # ── Shazam matched ────────────────────────────────────────────────────────

    # Zorba's-Dance-style: single uncertain hit returned a very short cut inside
    # a wide heuristic region — Shazam is probably confused. Show the heuristic
    # region for manual review instead of the misleading 31-second stub.
    if (uncertain and n_hits <= 1
            and cut_dur < SHAZAM_CONFUSED_DUR
            and h_dur >= cut_dur * SHAZAM_CONFUSED_RATIO):
        return "REVIEW-H", "shazam_confused_show_heuristic", h_start, h_end

    # Shazam cut is implausibly short (reversed or stub)
    if cut_dur < DROP_SHAZAM_DUR:
        return "DROP?", "shazam_too_short", sugg_start, sugg_end

    # AUTO: multiple confirmed hits, stable START boundary.
    if have_split:
        auto_ok = (start_conf in ("MEDIUM", "HIGH") and not start_unc and not end_unc
                   and n_hits >= AUTO_MIN_HITS and cut_dur >= AUTO_MIN_DURATION)
    else:
        # Legacy pre-split CSV: fall back to the old blanket-uncertain gate.
        auto_ok = (not uncertain and confidence in ("MEDIUM", "HIGH")
                   and n_hits >= AUTO_MIN_HITS and cut_dur >= AUTO_MIN_DURATION)

    if auto_ok:
        # Safety: cut starts well before heuristic region → may include host/ads over intro
        if sugg_start < h_start - EARLY_START_THRESHOLD:
            return "REVIEW-B", "early_start_boundary", sugg_start, sugg_end
        return "AUTO", "shazam_verified", sugg_start, sugg_end

    # Otherwise: Shazam found the song but boundary needs checking
    return "REVIEW-S", "shazam_uncertain", sugg_start, sugg_end


def source_tag(row: dict) -> str:
    region_type = (row.get("region_type") or "").strip()
    merge_note  = (row.get("merge_note") or "").strip()
    if merge_note:
        return "shazam_merged"
    if "shazam" in region_type and "heuristic" in region_type:
        return "heuristic+shazam"
    if "shazam" in region_type:
        return "shazam"
    return "heuristic"


def confidence_display(row: dict) -> str:
    bc = (row.get("boundary_confidence") or "").strip()
    hc = row.get("heuristic_conf") or ""
    matched = (row.get("matched_song") or "").strip()
    if bc:
        n = _int(row.get("n_samples_matched"))
        return f"{bc} ({n} hits)" if n else bc
    if hc:
        return f"heuristic conf={hc}"
    return "unknown"


def review_priority(label: str, row: dict) -> str:
    if label == "AUTO":
        return "auto"
    if label == "REVIEW-B":
        return "high"
    if label == "REVIEW-H":
        h_dur = _float(row.get("heuristic_end")) - _float(row.get("heuristic_start"))
        return "high" if h_dur >= 120 else "medium"
    if label == "REVIEW-S":
        n = _int(row.get("n_samples_matched"))
        return "medium" if n >= 2 else "high"
    if label == "EXT":
        return "medium"
    return "low"  # DROP?


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--review-csv",   type=Path, required=True,
                        help="review_sheet_<ep>.csv from make_review.py")
    parser.add_argument("--episode-name", required=True)
    parser.add_argument("--out-dir",      type=Path, default=Path("data/labels"))
    parser.add_argument("--transcript",   type=Path, default=None,
                        help="transcript json for the word-guard "
                             "(default: data/transcripts/<episode-name>.json)")
    args = parser.parse_args()

    # ── Load review sheet ─────────────────────────────────────────────────────
    rows: list[dict] = []
    with args.review_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # ── Load transcript words for the AUTO word-guard ─────────────────────────
    transcript = args.transcript or Path(f"data/transcripts/{args.episode_name}.json")
    speech_words: list[tuple[float, float]] = []
    if transcript.exists():
        speech_words = load_speech_words(transcript)
        print(f"Word-guard: {len(speech_words)} speech words from {transcript.name}")
    else:
        print(f"Word-guard: transcript {transcript} not found — guard disabled")

    # ── Classify each primary row ─────────────────────────────────────────────
    classified: list[dict] = []

    for row in rows:
        label, reason, cut_s, cut_e = classify(row)

        # Gap-shazam extension — added as a sibling [EXT] entry later
        gs_song  = (row.get("gap_shazam_song") or "").strip()
        gs_start = _float(row.get("gap_shazam_start"))
        gs_end   = _float(row.get("gap_shazam_end"))
        gs_type  = (row.get("gap_shazam_type") or "").strip()
        gs_unc   = (row.get("gap_shazam_status") or "") == "UNCERTAIN"

        # Auto-apply same-song tail extension to AUTO cuts. The gap scan heard
        # the SAME song still playing past this cut's end, so the end ran short
        # (the "tail" pattern, ~80s on the worst cases). Extend only to the last
        # scan-CONFIRMED music minus a margin — never to the DB-duration
        # projection (gap_shazam_end), which overshoots into host talk when the
        # DB has the album edit.
        tail_extended = False
        m_song = (row.get("matched_song") or "").strip()
        gs_last_conf = _float(row.get("gap_shazam_last_confirmed"))
        if (label == "AUTO" and gs_type == "extension"
                and gs_song and m_song
                and gs_song.lower() == m_song.lower()
                and gs_last_conf - TAIL_EXT_MARGIN > cut_e):
            cut_e = gs_last_conf - TAIL_EXT_MARGIN
            reason += "+tail_extended"
            tail_extended = True

        # Inward safety trim for unreviewed deletion (see AUTO_*_TRIM above).
        # Ends set from scan-confirmed music (tail extension / gap-find floor)
        # already carry their own margin and are not trimmed again.
        guard_note = ""
        if label == "AUTO":
            cut_s += AUTO_START_TRIM
            end_from_scan_floor = tail_extended or (
                reason == "gap_shazam_verified"
                and abs(cut_e - (gs_last_conf - TAIL_EXT_MARGIN)) < 0.01)
            if not end_from_scan_floor:
                cut_e -= AUTO_END_TRIM
            if speech_words:
                cut_s, cut_e, guard_note = word_guard(cut_s, cut_e, speech_words)

        dur = cut_e - cut_s

        # When the row was classified off the gap-scan match (a new_gap_find),
        # surface the gap-scan song + hit count instead of the empty normal-Shazam
        # fields, so the row shows what it actually is rather than "(no match)".
        from_gap = reason.startswith("gap_shazam")
        song_disp = (row.get("matched_song") or "").strip()
        n_hits_disp = row.get("n_samples_matched", "")
        conf_disp = confidence_display(row)
        if from_gap:
            song_disp = gs_song
            n_hits_disp = row.get("gap_shazam_n_hits", "")
            gs_flag = "⚠" if gs_unc else "✓"
            conf_disp = f"gap_scan {gs_flag} ({n_hits_disp} hits)" if n_hits_disp else f"gap_scan {gs_flag}"

        classified.append({
            "label":               label,
            "reason":              reason,
            "start":               cut_s,
            "end":                 cut_e,
            "start_hms":          hms(cut_s),
            "end_hms":            hms(cut_e),
            "duration_s":         round(dur, 1),
            "song":               song_disp,
            "source":             source_tag(row),
            "boundary_confidence": conf_disp,
            "n_hits":             n_hits_disp,
            "auto_cut":           label == "AUTO",
            "review_priority":    review_priority(label, row),
            "reason_text":        reason,
            "guard_note":         guard_note,
            "merge_note":         (row.get("merge_note") or "").strip(),
            "uncertain_reasons":  (row.get("uncertain_reasons") or "").strip(),
            "heuristic_start":    row.get("heuristic_start", ""),
            "heuristic_end":      row.get("heuristic_end", ""),
            "heuristic_conf":     row.get("heuristic_conf", ""),
            "shazam_status":      row.get("shazam_status", ""),
            "gap_shazam_song":    gs_song,
            "gap_shazam_start":   gs_start if gs_song else "",
            "gap_shazam_end":     gs_end   if gs_song else "",
            "gap_shazam_type":    gs_type,
            # Fill-in fields
            "actual_start":       "",
            "actual_end":         "",
            "notes":              "",
        })

        # Add sibling [EXT] row for gap-scan extensions (skipped when the
        # extension was already auto-applied to the AUTO row above)
        if gs_song and gs_type == "extension" and gs_end > gs_start and not tail_extended:
            gs_flag = " ⚠" if gs_unc else " ✓"
            classified.append({
                "label":               "EXT",
                "reason":              "gap_scan_extension",
                "start":               gs_start,
                "end":                 gs_end,
                "start_hms":          hms(gs_start),
                "end_hms":            hms(gs_end),
                "duration_s":         round(gs_end - gs_start, 1),
                "song":               gs_song,
                "source":             "gap_scan",
                "boundary_confidence": "gap_scan" + gs_flag,
                "n_hits":             "",
                "auto_cut":           False,
                "review_priority":    "medium",
                "reason_text":        "gap_scan_extension",
                "merge_note":         "",
                "uncertain_reasons":  (row.get("gap_shazam_uncertain_reasons") or "").strip(),
                "heuristic_start":    row.get("heuristic_start", ""),
                "heuristic_end":      row.get("heuristic_end", ""),
                "heuristic_conf":     row.get("heuristic_conf", ""),
                "shazam_status":      "UNCERTAIN" if gs_unc else "ok",
                "gap_shazam_song":    gs_song,
                "gap_shazam_start":   gs_start,
                "gap_shazam_end":     gs_end,
                "gap_shazam_type":    gs_type,
                "actual_start":       "",
                "actual_end":         "",
                "notes":              "",
            })

    # Sort by start time
    classified.sort(key=lambda r: r["start"])

    # ── Audacity label file ───────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    aud_path = args.out_dir / f"hybrid_review_audacity_{args.episode_name}.txt"

    label_to_flag = {
        "AUTO":     "✓",
        "REVIEW-S": "⚠",
        "REVIEW-B": "⟳",
        "REVIEW-H": "?",
        "EXT":      "→",
        "DROP?":    "✗",
    }

    with aud_path.open("w", encoding="utf-8") as f:
        for r in classified:
            flag = label_to_flag.get(r["label"], "")
            song = f" {r['song']}" if r["song"] else ""
            conf = r["boundary_confidence"]
            text = f"[{r['label']} {flag}]{song}  [{conf}]"
            f.write(f"{r['start']:.3f}\t{r['end']:.3f}\t{text}\n")

    # ── hybrid_review.csv (all rows) ─────────────────────────────────────────
    all_fields = [
        "label", "reason_text",
        "start", "end", "start_hms", "end_hms", "duration_s",
        "song", "source", "boundary_confidence", "n_hits",
        "auto_cut", "review_priority",
        "guard_note", "merge_note", "uncertain_reasons",
        "heuristic_start", "heuristic_end", "heuristic_conf",
        "shazam_status",
        "gap_shazam_song", "gap_shazam_start", "gap_shazam_end", "gap_shazam_type",
        "actual_start", "actual_end", "notes",
    ]
    all_csv = args.out_dir / f"hybrid_review_{args.episode_name}.csv"
    with all_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(classified)

    # ── hybrid_auto_cuts.csv ─────────────────────────────────────────────────
    auto_fields = [
        "start", "end", "start_hms", "end_hms", "duration_s",
        "song", "source", "boundary_confidence", "n_hits",
    ]
    auto_rows = [r for r in classified if r["label"] == "AUTO"]
    auto_csv = args.out_dir / f"hybrid_auto_cuts_{args.episode_name}.csv"
    with auto_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=auto_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(auto_rows)

    # ── hybrid_manual_review.csv ──────────────────────────────────────────────
    manual_fields = [
        "label", "start", "end", "start_hms", "end_hms", "duration_s",
        "song", "source", "boundary_confidence", "review_priority",
        "reason_text", "merge_note", "uncertain_reasons",
        "actual_start", "actual_end", "notes",
    ]
    manual_rows = [r for r in classified
                   if r["label"] in ("REVIEW-S", "REVIEW-B", "REVIEW-H", "EXT")]
    manual_csv = args.out_dir / f"hybrid_manual_review_{args.episode_name}.csv"
    with manual_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=manual_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(manual_rows)

    # ── Summary report ─────────────────────────────────────────────────────────
    counts = {lbl: 0 for lbl in ("AUTO", "REVIEW-S", "REVIEW-B", "REVIEW-H", "EXT", "DROP?")}
    for r in classified:
        counts[r["label"]] = counts.get(r["label"], 0) + 1

    auto_dur_total = sum(r["duration_s"] for r in classified if r["label"] == "AUTO")

    print(f"\n{'='*68}")
    print(f"  Hybrid review: {args.episode_name}")
    print(f"{'='*68}")
    print(f"  Total rows  : {len(classified)}")
    print(f"  [AUTO]      : {counts['AUTO']:>3}  — safe to cut automatically")
    print(f"  [REVIEW-S]  : {counts['REVIEW-S']:>3}  — Shazam matched, boundary needs check")
    print(f"  [REVIEW-B]  : {counts['REVIEW-B']:>3}  — song real, start boundary needs check")
    print(f"  [REVIEW-H]  : {counts['REVIEW-H']:>3}  — heuristic only, song unknown")
    print(f"  [EXT]       : {counts['EXT']:>3}  — suggested end extensions")
    print(f"  [DROP?]     : {counts['DROP?']:>3}  — short/weak, likely not music")
    print(f"\n  Auto-cut total duration: {auto_dur_total/60:.1f} min")
    print()

    print(f"  {'Label':<10} {'Start':>8} {'End':>8} {'Dur':>5}  Song / Note")
    print(f"  {'─'*65}")
    for r in classified:
        song = r["song"][:35] if r["song"] else "(no match)"
        note = ""
        if r["merge_note"]:
            note = f"  ← {r['merge_note'][:40]}"
        elif r["uncertain_reasons"]:
            note = f"  ← {r['uncertain_reasons'][:40]}"
        print(f"  [{r['label']:<8}] {r['start_hms']:>8} {r['end_hms']:>8} "
              f"{r['duration_s']:>4.0f}s  {song}{note}")

    print(f"\n  Outputs:")
    print(f"    {aud_path.name}")
    print(f"    {all_csv.name}")
    print(f"    {auto_csv.name}")
    print(f"    {manual_csv.name}")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()
