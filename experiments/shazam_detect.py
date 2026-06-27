"""
Shazam-based song detection with multi-source duration lookup.

Replaces the AcoustID approach with Shazam recognition (no API key needed,
much broader database including Icelandic music), then tries multiple sources
for song duration so cut boundaries can be estimated precisely.

Duration lookup order:
  1. iTunes API via Shazam's trackadamid (when present)
  2. MusicBrainz search by artist + title (free, no auth)
  3. Spotify search (needs SPOTIFY_CLIENT_ID + SPOTIFY_CLIENT_SECRET env vars)
  4. Fall back to heuristic region boundaries

Boundary logic:
  - song_start_in_episode = clip_start - shazam_offset
  - song_end_in_episode   = song_start + duration
  - Clamped to region with BOUNDARY_SLACK tolerance
  - Padded by PRE_PADDING / POST_PADDING
  - Flagged "uncertain" if estimated boundaries are implausible

IMPORTANT: Do not trust official duration blindly. Radio edits, early fades,
and talk-over at the intro/outro are common. Suggested cuts are always marked
for review, never auto-applied.

Usage:
    python experiments/shazam_detect.py \\
        --audio episodes/fm95blo-2012-03-09-mp3.mp3 \\
        --candidates data/candidates/fm95blo-2012-03-09.json \\
        --features data/features/fm95blo-2012-03-09.json \\
        --episode-name fm95blo-2012-03-09

Outputs (all written to data/labels/):
    song_matches_<name>.csv          -- full match + cut data
    song_matches_<name>.json         -- same, machine-readable
    song_cut_suggestions_<name>_audacity.txt
    song_unmatched_<name>.csv        -- regions with no Shazam match

Install deps:
    pip install shazamio
"""

import argparse
import asyncio
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

try:
    from shazamio import Shazam
except ImportError:
    sys.exit("Missing dep: pip install shazamio")

# ── Tuning parameters ──────────────────────────────────────────────────────────

CORE_TRIM_FRACTION = 0.20   # skip this fraction off each end of a region
MAX_CORE_SECONDS   = 90.0   # cap the fingerprinted window
MIN_CORE_SECONDS   = 8.0    # too short to recognise reliably
RETRY_OFFSET_S     = 30.0   # if first attempt fails, shift the window by this much

PRE_PADDING  = 3.0   # seconds of lead-in before estimated song start
POST_PADDING = 5.0   # seconds of trail-out after estimated song end

# A cut is flagged "uncertain" when the estimated boundaries overshoot the
# heuristic region by more than this (suggests wrong duration or wrong offset).
BOUNDARY_SLACK         = 40.0   # how far outside the region we trust
UNCERTAIN_OVERSHOOT    = 60.0   # beyond this → flag uncertain, clamp to region


MB_USER_AGENT = "fm95blo-cutter/1.0 (bjarturpall@gmail.com)"
MB_RATE_LIMIT  = 1.1   # seconds between MusicBrainz requests (their policy: 1/s)

_mb_last_request: float = 0.0
_duration_cache: dict[str, tuple[float | None, str]] = {}  # (artist,title) → (seconds, source)


# ── Duration lookup ────────────────────────────────────────────────────────────

def _itunes_duration(track_adam_id: str) -> float | None:
    if not track_adam_id:
        return None
    try:
        url = f"https://itunes.apple.com/lookup?id={track_adam_id}"
        with urllib.request.urlopen(url, timeout=8) as r:
            data = json.loads(r.read())
        results = data.get("results", [])
        if results and "trackTimeMillis" in results[0]:
            return results[0]["trackTimeMillis"] / 1000.0
    except Exception:
        pass
    return None


def _mb_duration(artist: str, title: str) -> float | None:
    global _mb_last_request
    wait = MB_RATE_LIMIT - (time.time() - _mb_last_request)
    if wait > 0:
        time.sleep(wait)
    try:
        q = urllib.parse.quote(f'artist:"{artist}" recording:"{title}"')
        url = f"https://musicbrainz.org/ws/2/recording?query={q}&limit=5&fmt=json"
        req = urllib.request.Request(url, headers={"User-Agent": MB_USER_AGENT})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        # Take first recording with a length; prefer exact title match
        for rec in data.get("recordings", []):
            if rec.get("length"):
                return rec["length"] / 1000.0
    except Exception:
        pass
    finally:
        _mb_last_request = time.time()
    return None


def _spotify_token() -> str | None:
    client_id     = os.environ.get("SPOTIFY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None
    try:
        import base64
        creds = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        data  = b"grant_type=client_credentials"
        req   = urllib.request.Request(
            "https://accounts.spotify.com/api/token", data=data,
            headers={"Authorization": f"Basic {creds}",
                     "Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            return json.loads(r.read()).get("access_token")
    except Exception:
        return None


_spotify_token_cache: str | None = None

def _spotify_duration(artist: str, title: str) -> float | None:
    global _spotify_token_cache
    if _spotify_token_cache is None:
        _spotify_token_cache = _spotify_token()
    if not _spotify_token_cache:
        return None
    try:
        q   = urllib.parse.quote(f"artist:{artist} track:{title}")
        url = f"https://api.spotify.com/v1/search?q={q}&type=track&limit=1"
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {_spotify_token_cache}"}
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
        items = data.get("tracks", {}).get("items", [])
        if items and "duration_ms" in items[0]:
            return items[0]["duration_ms"] / 1000.0
    except Exception:
        pass
    return None


def lookup_duration(artist: str, title: str, track_adam_id: str) -> tuple[float | None, str]:
    """Try each duration source in priority order. Returns (seconds, source_name)."""
    key = (artist.lower(), title.lower())
    if key in _duration_cache:
        cached_dur, cached_src = _duration_cache[key]
        return cached_dur, cached_src + " (cached)"

    dur = _itunes_duration(track_adam_id)
    if dur:
        _duration_cache[key] = (dur, "itunes")
        return dur, "itunes"

    dur = _mb_duration(artist, title)
    if dur:
        _duration_cache[key] = (dur, "musicbrainz")
        return dur, "musicbrainz"

    dur = _spotify_duration(artist, title)
    if dur:
        _duration_cache[key] = (dur, "spotify")
        return dur, "spotify"

    _duration_cache[key] = (None, "none")
    return None, "none"


# ── Shazam recognition ─────────────────────────────────────────────────────────

def extract_wav(audio_path: Path, start: float, end: float, tmp_path: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-ss", str(max(0.0, start)), "-to", str(end),
         "-i", str(audio_path), "-ar", "22050", "-ac", "1", tmp_path],
        check=True,
    )


def core_window(region: dict) -> tuple[float, float]:
    start, end = region["start"], region["end"]
    trim = (end - start) * CORE_TRIM_FRACTION
    cs, ce = start + trim, end - trim
    if ce - cs > MAX_CORE_SECONDS:
        mid = (cs + ce) / 2
        cs, ce = mid - MAX_CORE_SECONDS / 2, mid + MAX_CORE_SECONDS / 2
    return cs, ce


async def shazam_clip(audio_path: Path, start: float, end: float) -> dict | None:
    """Extract [start, end) and run Shazam. Returns parsed track info or None."""
    shazam = Shazam()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        extract_wav(audio_path, start, end, tmp_path)
        result = await shazam.recognize(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    matches = result.get("matches", [])
    track   = result.get("track")
    if not matches or not track:
        return None

    return {
        "title":        track.get("title", ""),
        "artist":       track.get("subtitle", ""),
        "offset":       matches[0].get("offset", 0.0),
        "trackadamid":  track.get("trackadamid") or "",
        "isrc":         track.get("isrc") or "",
        "key":          track.get("key", ""),
    }


# ── Boundary logic ─────────────────────────────────────────────────────────────

def compute_cut(region: dict, clip_start: float, match: dict,
                duration: float | None, episode_duration: float) -> dict:
    """
    Derive suggested cut boundaries from the Shazam offset + optional duration.

    song_start = clip_start - shazam_offset
    song_end   = song_start + duration (if known)

    Sanity checks:
    - If song_start is far before the heuristic region (>UNCERTAIN_OVERSHOOT),
      clamp to region_start and flag uncertain -- suggests the song started
      way earlier and we caught only the tail, or offset is wrong.
    - If song_end is far past the heuristic region (>UNCERTAIN_OVERSHOOT),
      clamp to region_end and flag uncertain -- suggests the duration is the
      album version but the radio used an edit or faded early.
    """
    r_start = region["start"]
    r_end   = region["end"]

    song_start = clip_start - match["offset"]
    song_end   = (song_start + duration) if duration else None

    uncertain_reasons: list[str] = []

    # ── suggested start ───────────────────────────────────────────────────────
    raw_start = song_start - PRE_PADDING
    if song_start < r_start - UNCERTAIN_OVERSHOOT:
        uncertain_reasons.append(
            f"song_start {song_start:.0f}s is {r_start - song_start:.0f}s before region"
        )
        cut_start = r_start  # be conservative
    else:
        cut_start = max(0.0, raw_start)

    # ── suggested end ─────────────────────────────────────────────────────────
    if song_end is not None:
        raw_end = song_end + POST_PADDING
        if song_end > r_end + UNCERTAIN_OVERSHOOT:
            uncertain_reasons.append(
                f"song_end {song_end:.0f}s is {song_end - r_end:.0f}s past region "
                f"(radio edit / early fade?)"
            )
            cut_end = r_end + 5.0  # slight overshoot to catch fade
        else:
            cut_end = min(episode_duration, raw_end)
    else:
        # No duration: use the heuristic region boundary
        cut_end = r_end
        if duration is None:
            uncertain_reasons.append("no duration found, using heuristic end")

    cut_end = min(cut_end, episode_duration)

    return {
        "song_start_in_episode": round(song_start, 1),
        "song_end_in_episode":   round(song_end, 1) if song_end is not None else None,
        "suggested_start":       round(cut_start, 1),
        "suggested_end":         round(cut_end, 1),
        "uncertain":             len(uncertain_reasons) > 0,
        "uncertain_reasons":     uncertain_reasons,
    }


# ── Main detection loop ────────────────────────────────────────────────────────

async def process_region(
    audio_path: Path, region: dict, episode_duration: float
) -> dict:
    core_start, core_end = core_window(region)
    core_dur = core_end - core_start

    match      = None
    clip_start = core_start

    if core_dur >= MIN_CORE_SECONDS:
        match = await shazam_clip(audio_path, core_start, core_end)

        # Retry at a different position if first attempt failed
        if not match and core_dur > RETRY_OFFSET_S + MIN_CORE_SECONDS:
            retry_s = core_start + RETRY_OFFSET_S
            retry_e = min(core_end, retry_s + MAX_CORE_SECONDS)
            if retry_e - retry_s >= MIN_CORE_SECONDS:
                match = await shazam_clip(audio_path, retry_s, retry_e)
                if match:
                    clip_start = retry_s

    if not match:
        return {
            **region,
            "core_start": round(core_start, 1), "core_end": round(core_end, 1),
            "matched_song": None, "artist": None, "title": None,
            "shazam_offset": None, "isrc": None,
            "duration_s": None, "duration_source": "none",
            "song_start_in_episode": None, "song_end_in_episode": None,
            "suggested_start": region["start"], "suggested_end": region["end"],
            "cut_source": "heuristic_fallback",
            "uncertain": True, "uncertain_reasons": ["no shazam match"],
        }

    song_label = f"{match['artist']} - {match['title']}"
    print(f"    MATCH: {song_label} | offset={match['offset']:.1f}s", file=sys.stderr)

    duration, dur_source = lookup_duration(
        match["artist"], match["title"], match["trackadamid"]
    )
    if duration:
        print(f"    duration={duration:.1f}s via {dur_source}", file=sys.stderr)
    else:
        print(f"    no duration found (tried itunes/mb/spotify)", file=sys.stderr)

    cut = compute_cut(region, clip_start, match, duration, episode_duration)
    if cut["uncertain"]:
        print(f"    UNCERTAIN: {'; '.join(cut['uncertain_reasons'])}", file=sys.stderr)

    return {
        **region,
        "core_start":    round(core_start, 1),
        "core_end":      round(core_end, 1),
        "matched_song":  song_label,
        "artist":        match["artist"],
        "title":         match["title"],
        "shazam_offset": round(match["offset"], 1),
        "isrc":          match["isrc"],
        "duration_s":    round(duration, 1) if duration else None,
        "duration_source": dur_source,
        "cut_source":    "shazam+" + dur_source if duration else "shazam_heuristic_end",
        **cut,
    }


async def run_all(audio_path: Path, candidates: list, episode_duration: float) -> list:
    results = []
    for i, region in enumerate(candidates):
        print(
            f"  [{i+1}/{len(candidates)}] {region['start']:.0f}–{region['end']:.0f}s ...",
            file=sys.stderr,
        )
        result = await process_region(audio_path, region, episode_duration)
        results.append(result)
        await asyncio.sleep(0.5)   # polite to Shazam
    return results


# ── Output writers ─────────────────────────────────────────────────────────────

MATCH_FIELDS = [
    "index", "region_start", "region_end",
    "matched_song", "shazam_offset", "duration_source", "duration_s",
    "song_start_in_episode", "song_end_in_episode",
    "suggested_start", "suggested_end", "suggested_start_hms", "suggested_end_hms",
    "cut_source", "uncertain", "uncertain_reasons", "decision",
]

UNMATCH_FIELDS = [
    "index", "region_start", "region_end", "uncertain_reasons",
]


def _hms(s: float) -> str:
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{sec:05.2f}"


def write_outputs(results: list, out_dir: Path, episode_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    matched   = [r for r in results if r["matched_song"]]
    unmatched = [r for r in results if not r["matched_song"]]

    # ── song_matches_<name>.csv ──────────────────────────────────────────────
    matches_csv = out_dir / f"song_matches_{episode_name}.csv"
    with matches_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(MATCH_FIELDS)
        for i, r in enumerate(matched):
            decision = "review (uncertain)" if r["uncertain"] else "review (matched)"
            writer.writerow([
                i, r["start"], r["end"],
                r["matched_song"], r["shazam_offset"], r["duration_source"], r["duration_s"],
                r["song_start_in_episode"], r["song_end_in_episode"],
                r["suggested_start"], r["suggested_end"],
                _hms(r["suggested_start"]), _hms(r["suggested_end"]),
                r["cut_source"], r["uncertain"],
                "; ".join(r.get("uncertain_reasons", [])),
                decision,
            ])

    # ── song_matches_<name>.json ─────────────────────────────────────────────
    matches_json = out_dir / f"song_matches_{episode_name}.json"
    matches_json.write_text(
        json.dumps(matched, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ── song_cut_suggestions_<name>_audacity.txt ─────────────────────────────
    aud = out_dir / f"song_cut_suggestions_{episode_name}_audacity.txt"
    with aud.open("w", encoding="utf-8") as f:
        for r in results:
            flag  = " [UNCERTAIN]" if r["uncertain"] else ""
            label = (r["matched_song"] or "unmatched") + flag
            f.write(f"{r['suggested_start']:.3f}\t{r['suggested_end']:.3f}\t{label}\n")

    # ── song_unmatched_<name>.csv ────────────────────────────────────────────
    unmatch_csv = out_dir / f"song_unmatched_{episode_name}.csv"
    with unmatch_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(UNMATCH_FIELDS)
        for i, r in enumerate(unmatched):
            writer.writerow([
                i, r["start"], r["end"],
                "; ".join(r.get("uncertain_reasons", [])),
            ])

    print(f"\nOutputs written to {out_dir}/", file=sys.stderr)
    print(f"  {matches_csv.name}", file=sys.stderr)
    print(f"  {matches_json.name}", file=sys.stderr)
    print(f"  {aud.name}", file=sys.stderr)
    print(f"  {unmatch_csv.name}", file=sys.stderr)


def print_summary_table(results: list, episode_name: str) -> None:
    matched = [r for r in results if r["matched_song"]]
    print(f"\n{'─'*110}", file=sys.stderr)
    print(f"  {episode_name}  —  {len(matched)}/{len(results)} matched", file=sys.stderr)
    print(f"{'─'*110}", file=sys.stderr)
    hdr = f"  {'Region':>15}  {'Song':<42}  {'Offset':>6}  {'Dur src':>11}  {'Dur':>6}  {'Cut start':>10}  {'Cut end':>10}  {'Note'}"
    print(hdr, file=sys.stderr)
    print(f"{'─'*110}", file=sys.stderr)
    for r in results:
        region = f"{r['start']:.0f}–{r['end']:.0f}s"
        song   = (r["matched_song"] or "(unmatched)")[:42]
        offset = f"{r['shazam_offset']:.1f}s" if r["shazam_offset"] is not None else "—"
        dsrc   = r["duration_source"] or "—"
        dur    = f"{r['duration_s']:.0f}s" if r["duration_s"] else "—"
        cs     = _hms(r["suggested_start"])
        ce     = _hms(r["suggested_end"])
        note   = "UNCERTAIN" if r["uncertain"] else "ok"
        print(f"  {region:>15}  {song:<42}  {offset:>6}  {dsrc:>11}  {dur:>6}  {cs:>10}  {ce:>10}  {note}", file=sys.stderr)
    print(f"{'─'*110}\n", file=sys.stderr)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio",          type=Path, required=True)
    parser.add_argument("--candidates",     type=Path, required=True,
                        help="data/candidates/<episode>.json from detect_breaks.py")
    parser.add_argument("--features",       type=Path, default=None,
                        help="data/features/<episode>.json (for episode duration)")
    parser.add_argument("--episode-name",   required=True,
                        help="short name used in output filenames, e.g. fm95blo-2011-11-09")
    parser.add_argument("--out-dir",        type=Path, default=Path("data/labels"),
                        help="directory for all output files (default: data/labels)")
    args = parser.parse_args()

    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))

    if args.features:
        episode_duration = json.loads(args.features.read_text(encoding="utf-8"))["duration"]
    else:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(args.audio)],
            capture_output=True, text=True, check=True,
        )
        episode_duration = float(probe.stdout.strip())

    print(f"Scanning {len(candidates)} candidate regions in {args.audio.name} ...", file=sys.stderr)
    results = asyncio.run(run_all(args.audio, candidates, episode_duration))

    print_summary_table(results, args.episode_name)
    write_outputs(results, args.out_dir, args.episode_name)

    n_matched   = sum(1 for r in results if r["matched_song"])
    n_with_dur  = sum(1 for r in results if r["duration_s"])
    n_uncertain = sum(1 for r in results if r["uncertain"])
    print(
        f"{len(results)} regions  |  {n_matched} matched  |  "
        f"{n_with_dur} with duration  |  {n_uncertain} uncertain",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
