"""
Gap-scanning Shazam detector for missed music segments.

Identifies large empty gaps between existing detected music candidates
and scans each gap with Shazam at regular intervals to find songs the
heuristic detector never produced a candidate for.

Primary test case: GT08 in the 2012 episode — Pitbull "Give Me Everything
Tonight" (4612–4845s). The song fell in the 621s gap between finalcuts
candidates 4152–4384s and 5005–5059s and was never detected by either
the heuristic or normal Shazam region matching.

Usage:
    python experiments/shazam_gap_scan.py \\
        --audio episodes/fm95blo-2012-03-09-mp3.mp3 \\
        --candidates data/candidates/fm95blo-2012-03-09_finalcuts.json \\
        --episode-name fm95blo-2012-03-09 \\
        [--features data/features/fm95blo-2012-03-09.json] \\
        [--min-gap 180]           # skip gaps shorter than this (seconds)
        [--sample-interval 45]    # seconds between clip start times
        [--clip-len 15]           # seconds per Shazam clip
        [--skip-edges 10]         # don't sample within this many s of gap edge
        [--ground-truth data/labels/fm95blo-2012-03-09-gt.csv]
        [--out-dir data/labels]

Ground truth CSV (optional --ground-truth):
    start,end,label   (times in seconds or HH:MM:SS; skip label=false_positive)

Outputs (written to --out-dir):
    gap_shazam_matches_<name>.csv
    gap_shazam_matches_<name>.json
    gap_shazam_audacity_<name>.txt

Install deps:
    pip install shazamio
"""

import argparse
import asyncio
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

try:
    from shazamio import Shazam  # noqa: F401 – verify import before main loop
except ImportError:
    sys.exit("Missing dep: pip install shazamio")

# Import shared utilities from shazam_detect.py (same directory)
sys.path.insert(0, str(Path(__file__).parent))
from shazam_detect import (
    shazam_clip,
    lookup_duration,
    _hms,
    PRE_PADDING,
    POST_PADDING,
    BOUNDARY_SLACK,
)

# ── Tunable defaults ───────────────────────────────────────────────────────────

MIN_GAP_SECONDS   = 180    # seconds; gaps shorter than this are skipped
SAMPLE_INTERVAL   = 45     # seconds between sample clip start times
CLIP_LEN          = 15     # seconds per Shazam clip
SKIP_EDGES        = 10     # skip this many seconds at each gap edge
MIN_CLIP_SECONDS  = 8      # clips shorter than this are not submitted to Shazam
OVERLAP_THRESHOLD = 30     # seconds of overlap needed to count a GT match

# GT08 is the primary validation target (always checked regardless of --ground-truth)
GT08_START = 4612.0
GT08_END   = 4845.0
GT08_NOTE  = "Pitbull - Give Me Everything Tonight (2012 episode)"


# ── Gap detection ──────────────────────────────────────────────────────────────

def find_gaps(candidates: list, episode_duration: float, min_gap: float) -> list:
    """
    Return sorted list of gap dicts {start, end} for all inter-candidate gaps
    (and pre/post gaps) that are at least min_gap seconds long.
    """
    sorted_cands = sorted(candidates, key=lambda c: c["start"])
    gaps = []

    boundaries = []
    if sorted_cands:
        # Gap before first candidate
        if sorted_cands[0]["start"] >= min_gap:
            boundaries.append((0.0, sorted_cands[0]["start"]))
        # Gaps between consecutive candidates
        for i in range(len(sorted_cands) - 1):
            g_start = sorted_cands[i]["end"]
            g_end   = sorted_cands[i + 1]["start"]
            if g_end - g_start >= min_gap:
                boundaries.append((g_start, g_end))
        # Gap after last candidate
        trail = episode_duration - sorted_cands[-1]["end"]
        if trail >= min_gap:
            boundaries.append((sorted_cands[-1]["end"], episode_duration))
    else:
        boundaries.append((0.0, episode_duration))

    for g_start, g_end in boundaries:
        gaps.append({"start": g_start, "end": g_end})

    return gaps


def sample_points(gap_start: float, gap_end: float,
                  skip_edges: float, sample_interval: float,
                  clip_len: float) -> list[float]:
    """Evenly spaced clip start times within the usable interior of a gap."""
    first   = gap_start + skip_edges
    last_ok = gap_end - skip_edges - clip_len
    if first > last_ok:
        return []
    points = []
    t = first
    while t <= last_ok + 1e-6:
        points.append(round(t, 1))
        t += sample_interval
    return points


# ── Boundary computation (gap-scan specific) ───────────────────────────────────

def compute_gap_cut(
    gap_start: float,
    gap_end: float,
    clip_start: float,
    shazam_offset: float,
    duration: float | None,
    episode_duration: float,
) -> dict:
    """
    Estimate cut boundaries from Shazam offset + optional song duration.

    song_start = clip_start - shazam_offset
    song_end   = song_start + duration  (if known)

    Flags uncertain when the estimates land far outside the gap (suggests a
    wrong offset or an album duration that doesn't match the radio edit).
    """
    song_start = clip_start - shazam_offset
    song_end   = (song_start + duration) if duration is not None else None

    uncertain_reasons: list[str] = []

    if song_start < gap_start - BOUNDARY_SLACK:
        uncertain_reasons.append(
            f"song_start {song_start:.0f}s is {gap_start - song_start:.0f}s "
            f"before gap start (offset may be wrong)"
        )
    if song_end is not None and song_end > gap_end + BOUNDARY_SLACK:
        uncertain_reasons.append(
            f"song_end {song_end:.0f}s overshoots gap end by "
            f"{song_end - gap_end:.0f}s (radio edit vs. album duration?)"
        )

    cut_start = max(0.0, song_start - PRE_PADDING)
    if song_end is not None:
        cut_end = min(episode_duration, song_end + POST_PADDING)
    else:
        cut_end = gap_end
        uncertain_reasons.append("no duration found — using gap end as fallback")

    return {
        "song_start_in_episode": round(song_start, 1),
        "song_end_in_episode":   round(song_end, 1) if song_end is not None else None,
        "suggested_start":       round(cut_start, 1),
        "suggested_end":         round(cut_end, 1),
        "uncertain":             bool(uncertain_reasons),
        "uncertain_reasons":     uncertain_reasons,
    }


# ── Scanning ───────────────────────────────────────────────────────────────────

async def scan_gap(
    audio_path: Path,
    gap: dict,
    episode_duration: float,
    sample_interval: float,
    clip_len: float,
    skip_edges: float,
) -> list[dict]:
    """
    Run Shazam across all sample windows in one gap.
    Returns a list of raw match dicts (one per window that matched).
    """
    points = sample_points(gap["start"], gap["end"], skip_edges, sample_interval, clip_len)
    if not points:
        print(
            f"  Gap {gap['start']:.0f}–{gap['end']:.0f}s "
            f"({gap['end'] - gap['start']:.0f}s) — too narrow for any sample, skipping",
            file=sys.stderr,
        )
        return []

    print(
        f"\n  Gap {gap['start']:.0f}–{gap['end']:.0f}s "
        f"({gap['end'] - gap['start']:.0f}s) — {len(points)} samples",
        file=sys.stderr,
    )

    raw_matches = []
    for i, clip_start in enumerate(points):
        clip_end = min(clip_start + clip_len, gap["end"] - skip_edges, episode_duration)
        if clip_end - clip_start < MIN_CLIP_SECONDS:
            continue

        match = await shazam_clip(audio_path, clip_start, clip_end)
        if match:
            song_label = f"{match['artist']} - {match['title']}"
            print(
                f"    [{i+1:2d}/{len(points)}] t={clip_start:.0f}s — "
                f"MATCH: {song_label}  offset={match['offset']:.1f}s",
                file=sys.stderr,
            )
            raw_matches.append({
                "gap_start":    gap["start"],
                "gap_end":      gap["end"],
                "clip_start":   clip_start,
                "clip_end":     round(clip_end, 1),
                "offset":       match["offset"],
                "artist":       match["artist"],
                "title":        match["title"],
                "trackadamid":  match.get("trackadamid", ""),
                "isrc":         match.get("isrc", ""),
            })
        else:
            print(
                f"    [{i+1:2d}/{len(points)}] t={clip_start:.0f}s — no match",
                file=sys.stderr,
            )

        await asyncio.sleep(0.5)

    return raw_matches


def deduplicate(raw_matches: list[dict]) -> list[dict]:
    """
    Within each gap, keep at most one entry per unique (artist, title).
    When the same song matches multiple windows, keep the window with the
    smallest Shazam offset (= caught earliest in the song = most reliable
    song_start estimate). Record all offsets for transparency.
    """
    groups: dict = defaultdict(list)
    for m in raw_matches:
        key = (m["gap_start"], m["artist"].lower(), m["title"].lower())
        groups[key].append(m)

    deduped = []
    for matches in groups.values():
        best = min(matches, key=lambda m: m["offset"])
        best = dict(best)
        best["all_offsets"]        = sorted(m["offset"] for m in matches)
        best["n_windows_matched"]  = len(matches)
        deduped.append(best)

    deduped.sort(key=lambda m: (m["gap_start"], m["clip_start"]))
    return deduped


async def enrich(raw_matches: list[dict], episode_duration: float) -> list[dict]:
    """Look up song duration and compute final cut boundaries for each match."""
    results = []
    for m in raw_matches:
        duration, dur_source = lookup_duration(
            m["artist"], m["title"], m["trackadamid"]
        )
        if duration:
            print(
                f"  duration {m['artist']} - {m['title']}: "
                f"{duration:.1f}s via {dur_source}",
                file=sys.stderr,
            )
        cut = compute_gap_cut(
            m["gap_start"], m["gap_end"],
            m["clip_start"], m["offset"],
            duration, episode_duration,
        )
        results.append({
            **m,
            "matched_song":    f"{m['artist']} - {m['title']}",
            "duration_s":      round(duration, 1) if duration is not None else None,
            "duration_source": dur_source,
            **cut,
        })
    return results


async def run_all(
    audio_path: Path,
    gaps: list[dict],
    episode_duration: float,
    sample_interval: float,
    clip_len: float,
    skip_edges: float,
) -> list[dict]:
    all_raw: list[dict] = []
    for gap in gaps:
        raw = await scan_gap(
            audio_path, gap, episode_duration,
            sample_interval, clip_len, skip_edges,
        )
        all_raw.extend(raw)

    deduped = deduplicate(all_raw)
    return await enrich(deduped, episode_duration)


# ── Output ─────────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "gap_start", "gap_end",
    "matched_song", "n_windows_matched", "all_offsets",
    "clip_start", "offset",
    "duration_s", "duration_source",
    "song_start_in_episode", "song_end_in_episode",
    "suggested_start", "suggested_end",
    "suggested_start_hms", "suggested_end_hms",
    "uncertain", "uncertain_reasons",
]


def write_outputs(results: list[dict], out_dir: Path, episode_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"gap_shazam_matches_{episode_name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_FIELDS)
        for r in results:
            w.writerow([
                r["gap_start"], r["gap_end"],
                r["matched_song"],
                r["n_windows_matched"],
                "; ".join(f"{o:.1f}" for o in r["all_offsets"]),
                r["clip_start"], r["offset"],
                r["duration_s"], r["duration_source"],
                r["song_start_in_episode"], r["song_end_in_episode"],
                r["suggested_start"], r["suggested_end"],
                _hms(r["suggested_start"]), _hms(r["suggested_end"]),
                r["uncertain"],
                "; ".join(r["uncertain_reasons"]),
            ])

    json_path = out_dir / f"gap_shazam_matches_{episode_name}.json"
    json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    aud_path = out_dir / f"gap_shazam_audacity_{episode_name}.txt"
    with aud_path.open("w", encoding="utf-8") as f:
        for r in results:
            flag  = " [UNCERTAIN]" if r["uncertain"] else ""
            label = f"[GAP] {r['matched_song']}{flag}"
            f.write(f"{r['suggested_start']:.3f}\t{r['suggested_end']:.3f}\t{label}\n")

    print(f"\nOutputs written:", file=sys.stderr)
    for p in (csv_path, json_path, aud_path):
        print(f"  {p}", file=sys.stderr)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def _parse_time(s: str) -> float:
    parts = s.strip().split(":")
    if len(parts) == 1:
        return float(parts[0])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def load_gt(path: Path) -> list[dict]:
    gt = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row.get("label", "").lower().strip()
            if label in ("false_positive", "talk", ""):
                continue
            gt.append({
                "start": _parse_time(row["start"]),
                "end":   _parse_time(row["end"]),
                "label": label,
            })
    return sorted(gt, key=lambda g: g["start"])


def overlap_s(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def evaluate(results: list[dict], gt: list[dict]) -> None:
    used_gt: set[int] = set()
    tps: list = []
    fps: list = []

    for r in results:
        best_i, best_ov = None, 0.0
        for i, g in enumerate(gt):
            ov = overlap_s(r["suggested_start"], r["suggested_end"], g["start"], g["end"])
            if ov > best_ov:
                best_ov = ov
                best_i  = i
        if best_i is not None and best_ov >= OVERLAP_THRESHOLD:
            tps.append((r, gt[best_i], best_ov))
            used_gt.add(best_i)
        else:
            fps.append(r)

    fns = [gt[i] for i in range(len(gt)) if i not in used_gt]

    recall    = len(tps) / len(gt) * 100   if gt      else 0.0
    precision = len(tps) / len(results) * 100 if results else 0.0

    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  GAP SCAN EVALUATION  (overlap threshold ≥ {OVERLAP_THRESHOLD}s)")
    print(sep)
    print(f"  GT music breaks:  {len(gt)}")
    print(f"  Gap-scan matches: {len(results)}")
    print(f"  TP: {len(tps)}  FP: {len(fps)}  FN: {len(fns)}")
    if gt:
        print(f"  Recall:    {recall:.1f}%  ({len(tps)}/{len(gt)})")
    if results:
        print(f"  Precision: {precision:.1f}%  ({len(tps)}/{len(results)})")

    if tps:
        print(f"\n  True positives:")
        for r, g, ov in tps:
            print(f"    {r['matched_song'][:44]:<44}  "
                  f"cut {_hms(r['suggested_start'])}–{_hms(r['suggested_end'])}  "
                  f"GT {g['start']:.0f}–{g['end']:.0f}s  overlap {ov:.0f}s")

    if fps:
        print(f"\n  False positives (no GT overlap):")
        for r in fps:
            print(f"    {r['matched_song'][:44]:<44}  "
                  f"gap {r['gap_start']:.0f}–{r['gap_end']:.0f}s  "
                  f"cut {_hms(r['suggested_start'])}–{_hms(r['suggested_end'])}")

    if fns:
        print(f"\n  False negatives (GT breaks not found by gap scan):")
        for g in fns:
            print(f"    {g['start']:.0f}–{g['end']:.0f}s  ({g['label']})")

    print(sep)


def check_gt08(results: list[dict]) -> None:
    """Always report whether the primary test case (GT08) was found."""
    for r in results:
        ov = overlap_s(r["suggested_start"], r["suggested_end"], GT08_START, GT08_END)
        if ov >= OVERLAP_THRESHOLD:
            print(
                f"\n  ✓  GT08 FOUND: {r['matched_song']}\n"
                f"     cut  {_hms(r['suggested_start'])}–{_hms(r['suggested_end'])}\n"
                f"     gap  {r['gap_start']:.0f}–{r['gap_end']:.0f}s  "
                f"overlap={ov:.0f}s with GT {GT08_START:.0f}–{GT08_END:.0f}s\n"
                f"     n_windows_matched={r['n_windows_matched']}  "
                f"offsets={[round(o,1) for o in r['all_offsets']]}",
                file=sys.stderr,
            )
            return
    print(
        f"\n  ✗  GT08 MISSED: no gap-scan match overlaps "
        f"{GT08_START:.0f}–{GT08_END:.0f}s by ≥{OVERLAP_THRESHOLD}s\n"
        f"     ({GT08_NOTE})",
        file=sys.stderr,
    )


# ── Summary table ──────────────────────────────────────────────────────────────

def print_summary(results: list[dict], gaps: list[dict], episode_name: str) -> None:
    W = 104
    print(f"\n{'─'*W}", file=sys.stderr)
    print(
        f"  {episode_name}  —  "
        f"{len(results)} songs found across {len(gaps)} gaps scanned",
        file=sys.stderr,
    )
    print(f"{'─'*W}", file=sys.stderr)
    print(
        f"  {'Gap':>14}  {'Song':<44}  {'Win':>4}  "
        f"{'Offset':>7}  {'Dur':>6}  {'Cut start':>9}  {'Cut end':>9}  Note",
        file=sys.stderr,
    )
    print(f"{'─'*W}", file=sys.stderr)
    for r in results:
        gap_str = f"{r['gap_start']:.0f}–{r['gap_end']:.0f}s"
        song    = r["matched_song"][:44]
        win     = str(r["n_windows_matched"])
        offset  = f"{r['offset']:.1f}s"
        dur     = f"{r['duration_s']:.0f}s" if r["duration_s"] else "—"
        note    = "UNCERTAIN" if r["uncertain"] else "ok"
        print(
            f"  {gap_str:>14}  {song:<44}  {win:>4}  "
            f"{offset:>7}  {dur:>6}  "
            f"{_hms(r['suggested_start']):>9}  {_hms(r['suggested_end']):>9}  {note}",
            file=sys.stderr,
        )
    print(f"{'─'*W}\n", file=sys.stderr)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--audio",           type=Path, required=True,
                        help="episode MP3 (or any ffmpeg-readable audio)")
    parser.add_argument("--candidates",      type=Path, required=True,
                        help="candidates JSON, e.g. *_finalcuts.json")
    parser.add_argument("--episode-name",    required=True,
                        help="short name used in output filenames")
    parser.add_argument("--features",        type=Path, default=None,
                        help="features JSON for episode duration (faster than ffprobe)")
    parser.add_argument("--min-gap",         type=float, default=MIN_GAP_SECONDS,
                        help=f"skip gaps shorter than this (default {MIN_GAP_SECONDS}s)")
    parser.add_argument("--sample-interval", type=float, default=SAMPLE_INTERVAL,
                        help=f"seconds between clip start times (default {SAMPLE_INTERVAL}s)")
    parser.add_argument("--clip-len",        type=float, default=CLIP_LEN,
                        help=f"seconds per Shazam clip (default {CLIP_LEN}s)")
    parser.add_argument("--skip-edges",      type=float, default=SKIP_EDGES,
                        help=f"skip this many s at each gap edge (default {SKIP_EDGES}s)")
    parser.add_argument("--ground-truth",    type=Path, default=None,
                        help="optional CSV (start,end,label) for TP/FP/FN evaluation")
    parser.add_argument("--out-dir",         type=Path, default=Path("data/labels"),
                        help="output directory (default: data/labels)")
    args = parser.parse_args()

    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))

    if args.features:
        episode_duration = json.loads(
            args.features.read_text(encoding="utf-8")
        )["duration"]
    else:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(args.audio)],
            capture_output=True, text=True, check=True,
        )
        episode_duration = float(probe.stdout.strip())

    gaps = find_gaps(candidates, episode_duration, args.min_gap)

    total_samples = sum(
        len(sample_points(g["start"], g["end"], args.skip_edges,
                          args.sample_interval, args.clip_len))
        for g in gaps
    )

    print(
        f"\n{len(gaps)} gaps ≥ {args.min_gap:.0f}s found in "
        f"{args.audio.name}  (episode: {episode_duration:.0f}s):",
        file=sys.stderr,
    )
    for g in gaps:
        dur = g["end"] - g["start"]
        n   = len(sample_points(g["start"], g["end"], args.skip_edges,
                                args.sample_interval, args.clip_len))
        print(
            f"  {g['start']:.0f}–{g['end']:.0f}s  ({dur:.0f}s, {n} samples)",
            file=sys.stderr,
        )
    print(f"\nTotal Shazam queries: {total_samples}", file=sys.stderr)

    results = asyncio.run(run_all(
        args.audio, gaps, episode_duration,
        args.sample_interval, args.clip_len, args.skip_edges,
    ))

    print_summary(results, gaps, args.episode_name)
    check_gt08(results)

    if args.ground_truth:
        gt = load_gt(args.ground_truth)
        print(
            f"\nLoaded {len(gt)} GT music breaks from {args.ground_truth.name}",
            file=sys.stderr,
        )
        evaluate(results, gt)

    write_outputs(results, args.out_dir, args.episode_name)

    print(
        f"\n{len(results)} gap matches  |  {total_samples} samples scanned  "
        f"|  {len(gaps)} gaps",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
