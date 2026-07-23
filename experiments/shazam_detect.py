"""
Shazam-based song detection with multi-sample boundary estimation.

Instead of one clip per region, sweeps multiple clips across each candidate.
Multiple offsets → cluster estimated song_start values → median if they agree,
UNCERTAIN if they spread >30s.

End boundary uses:
  1. Song presence window: last matched sample gives a minimum on song length.
  2. End scan: a few extra clips probed past the region end.
  3. Acoustic snap: nearest music↔speech transition in features/transcript.

Boundary confidence:
  HIGH   — ≥3 matching samples, spread <15s, acoustic snap found
  MEDIUM — ≥2 matches and spread <30s, OR ≥3 matches without snap
  LOW    — single hit, spread >30s, or no duration

Usage:
    python experiments/shazam_detect.py \\
        --audio   episodes/fm95blo-2012-03-09-mp3.mp3 \\
        --candidates data/candidates/fm95blo-2012-03-09_finalcuts.json \\
        --features   data/features/fm95blo-2012-03-09.json \\
        --transcript data/transcripts/fm95blo-2012-03-09.json \\
        --episode-name fm95blo-2012-03-09

Outputs (all in data/labels/):
    song_matches_<name>.csv
    song_matches_<name>.json
    song_cut_suggestions_<name>_audacity.txt
    song_unmatched_<name>.csv
    boundary_debug_<name>.json   ← new: per-song boundary debug report
"""

import argparse
import asyncio
import csv
import json
import os
import statistics
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

SAMPLE_CLIP_LEN      = 15.0   # seconds per Shazam clip
SAMPLE_EDGE_SKIP     = 8.0    # skip from each region edge before sampling
SAMPLE_INTERVAL      = 45.0   # target spacing between sample centres
MAX_SAMPLES_PER_REG  = 6      # hard cap per region

CLUSTER_AGREE_S      = 15.0   # spread below this → HIGH start confidence
CLUSTER_WARN_S       = 30.0   # spread above this → UNCERTAIN

PRE_PADDING          = 3.0    # lead-in before estimated song start
POST_PADDING         = 5.0    # trail-out after estimated song end

BOUNDARY_SLACK       = 40.0   # how far outside region we still trust
UNCERTAIN_OVERSHOOT  = 60.0   # beyond this → clamp and flag uncertain

SNAP_RADIUS          = 30.0   # search ±Xs from estimated boundary
SNAP_SMOOTH_WIN      = 7      # smooth signal window half-width (seconds each side)
MUSIC_LO             = 0.30   # signal below this = clear speech

CLAWBACK_MAX         = 25.0   # never move a start more than this far forward
CLAWBACK_MUSIC_GAP   = 6.0    # a wordless stretch ≥ this = host stopped, music alone
CLAWBACK_PROBE       = 4.0    # no host word within this of the estimate → already
                              #   on clean music, nothing to claw back

END_SCAN_PROBES      = 4      # clips to probe past region end (n≥2 path)
END_SCAN_PROBES_N1   = 10    # clips for single-hit forward sweep
END_SCAN_STEP        = 30.0   # spacing between end-scan probes

MB_USER_AGENT = "fm95blo-cutter/1.0 (bjarturpall@gmail.com)"
MB_RATE_LIMIT  = 1.1

_mb_last_request: float = 0.0
_duration_cache: dict[str, tuple[float | None, str]] = {}


# ── Duration lookup (unchanged) ────────────────────────────────────────────────

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


# ── Shazam clip (unchanged) ────────────────────────────────────────────────────

def extract_wav(audio_path: Path, start: float, end: float, tmp_path: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-ss", str(max(0.0, start)), "-to", str(end),
         "-i", str(audio_path), "-ar", "22050", "-ac", "1", tmp_path],
        check=True,
    )


async def shazam_clip(audio_path: Path, start: float, end: float) -> dict | None:
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
        "title":       track.get("title", ""),
        "artist":      track.get("subtitle", ""),
        "offset":      matches[0].get("offset", 0.0),
        "trackadamid": track.get("trackadamid") or "",
        "isrc":        track.get("isrc") or "",
        "key":         track.get("key", ""),
    }


# ── Acoustic signal ────────────────────────────────────────────────────────────

def build_music_signal(
    features_windows: list[dict],
    transcript_segments: list[dict] | None,
    episode_duration: float,
) -> list[float]:
    """
    Per-second music likelihood signal.
    Combines music_score (spectral) with rms_db (loudness) from features,
    and no_speech_prob from transcript segments where available.
    Returns list indexed by integer second.
    """
    n = int(episode_duration) + 2
    signal = [0.0] * n

    # Normalise rms_db to [0,1]: typical range is -50 (silence) to -10 (loud)
    for w in features_windows:
        t = int(w["t"])
        if not (0 <= t < n):
            continue
        ms  = w.get("music_score", 0.0)
        rms = max(0.0, min(1.0, (w.get("rms_db", -50.0) + 50.0) / 40.0))
        signal[t] = 0.4 * ms + 0.6 * rms

    # Blend in no_speech_prob from transcript (high = non-speech = music/silence)
    if transcript_segments:
        for seg in transcript_segments:
            nsp = seg.get("no_speech_prob", 0.5)
            s_t = max(0, int(seg["start"]))
            e_t = min(n, int(seg["end"]) + 1)
            for t in range(s_t, e_t):
                signal[t] = max(signal[t], nsp)

    return signal


def smooth_signal(signal: list[float], half_win: int = SNAP_SMOOTH_WIN) -> list[float]:
    n = len(signal)
    result = [0.0] * n
    for t in range(n):
        lo = max(0, t - half_win)
        hi = min(n, t + half_win + 1)
        result[t] = sum(signal[lo:hi]) / (hi - lo)
    return result


def snap_boundary(
    smoothed: list[float],
    approx_t: float,
    direction: str,          # 'start' or 'end'
    radius: float = SNAP_RADIUS,
) -> tuple[float | None, str]:
    """
    Find the nearest music↔speech transition within ±radius of approx_t.
    direction='start': look for signal rising through MUSIC_LO (speech→music).
    direction='end':   look for signal falling through MUSIC_LO (music→speech).
    Returns (snapped_time, note) or (None, reason).
    """
    t_lo = max(0, int(approx_t - radius))
    t_hi = min(len(smoothed) - 2, int(approx_t + radius))

    crossings = []
    for t in range(t_lo, t_hi + 1):
        v0 = smoothed[t]
        v1 = smoothed[t + 1] if t + 1 < len(smoothed) else v0
        if direction == "start" and v0 < MUSIC_LO <= v1:
            crossings.append(t)
        elif direction == "end" and v0 >= MUSIC_LO > v1:
            crossings.append(t)

    if not crossings:
        return None, f"no {direction} acoustic transition found in ±{radius:.0f}s"

    best = min(crossings, key=lambda c: abs(c - approx_t))
    dist = abs(best - approx_t)
    direction_word = "earlier" if best < approx_t else "later"
    return float(best), f"{direction} snap {dist:.0f}s {direction_word} → {_hms(best)}"


def clawback_start(
    est_start: float,
    region_end: float,
    words: list[dict],
    max_move: float = CLAWBACK_MAX,
    music_gap: float = CLAWBACK_MUSIC_GAP,
    probe:    float = CLAWBACK_PROBE,
) -> tuple[float | None, str]:
    """
    Pull a music-cut start forward off host talk-over-intro, using the transcript.

    The acoustic snap fails when the host talks *over* the song intro: the music
    signal is already high, so there is no speech→music rising edge to snap to and
    we fall back to the raw Shazam offset, which sits ~10s early, inside the talk.
    The transcript still shows the seam — a run of host words, then a sustained
    wordless gap once the host stops and music plays alone. Move the start to the
    end of the last spoken word before that first sustained gap.

    Only ever moves the start *later*, never earlier, never past region_end, and
    never more than max_move. Returns (new_start, note) if a clawback applies,
    else (None, reason) — and only fires when a real speech→music onset is found,
    so continuous host talk or continuous sung vocals leave the start untouched.
    """
    window_end = min(region_end, est_start + max_move)
    # Words overlapping [est_start, window_end]: host speech at/after the estimate.
    talk = sorted(
        (w for w in words
         if w.get("end", 0.0) > est_start and w.get("start", 0.0) < window_end),
        key=lambda w: w["start"],
    )

    if not talk or talk[0]["start"] > est_start + probe:
        # No host words right after the estimate → already clean music. Leave it.
        return None, "clawback: start already on clean audio"

    # Walk forward; stop at the first sustained wordless gap = host stopped talking.
    prev_end = talk[0]["end"]
    gap_found = False
    for w in talk[1:]:
        if w["start"] - prev_end >= music_gap:
            gap_found = True           # gap before this word: music took over at prev_end
            break
        prev_end = w["end"]
    else:
        # No inter-word gap; is there a trailing gap out to the window edge?
        if window_end - prev_end >= music_gap:
            gap_found = True

    if not gap_found:
        # Host talks continuously through the window — can't locate the music
        # onset, so don't guess. Keep the acoustic/Shazam estimate.
        return None, "clawback: host talk continuous, no music onset found"

    new_start = min(prev_end, window_end)
    if new_start <= est_start + 1.0:
        return None, "clawback: no forward move"
    moved = new_start - est_start
    return round(new_start, 1), f"clawback +{moved:.0f}s off host talk → {_hms(new_start)}"


# ── Multi-sample sweep ─────────────────────────────────────────────────────────

def sample_region_points(
    r_start: float, r_end: float,
    edge_skip: float = SAMPLE_EDGE_SKIP,
    interval:  float = SAMPLE_INTERVAL,
    clip_len:  float = SAMPLE_CLIP_LEN,
    max_n:     int   = MAX_SAMPLES_PER_REG,
) -> list[float]:
    """Evenly spaced sample start times inside the region, skipping edges."""
    s = r_start + edge_skip
    e = r_end   - edge_skip - clip_len

    if e <= s + 1:
        # Too short for edge skipping: just try the centre
        mid = (r_start + r_end) / 2.0 - clip_len / 2.0
        return [round(max(r_start, min(mid, r_end - clip_len)), 1)]

    span = e - s
    n = max(2, min(max_n, int(span / interval) + 1))

    if n == 1:
        return [round(s + span / 2.0, 1)]

    pts = [round(s + i * span / (n - 1), 1) for i in range(n)]
    return pts


def _song_key(match: dict) -> str:
    k = match.get("key", "")
    if k:
        return k
    return f"{match.get('artist','').lower()}|{match.get('title','').lower()}"


def _pick_best_song(raw_matches: list[dict]) -> tuple[str, list[dict]]:
    """Return (song_key, matches_for_that_song) for the most-matched song."""
    counts: dict[str, int] = {}
    by_key: dict[str, list] = {}
    for m in raw_matches:
        k = _song_key(m)
        counts[k] = counts.get(k, 0) + 1
        by_key.setdefault(k, []).append(m)
    best_key = max(counts, key=lambda k: (counts[k], -by_key[k][0]["time"]))
    return best_key, by_key[best_key]


def _boundary_confidence(n: int, spread: float, start_snapped: bool, end_snapped: bool) -> str:
    if n >= 3 and spread < CLUSTER_AGREE_S:
        return "HIGH" if (start_snapped or end_snapped) else "MEDIUM"
    if n >= 2 and spread < CLUSTER_WARN_S:
        return "MEDIUM"
    return "LOW"


# ── End scan ───────────────────────────────────────────────────────────────────

async def scan_end(
    audio_path: Path,
    song_key: str,
    best_match: dict,
    est_end: float,
    episode_duration: float,
    n_probes:  int   = END_SCAN_PROBES,
    step:      float = END_SCAN_STEP,
    clip_len:  float = SAMPLE_CLIP_LEN,
) -> tuple[float | None, list[dict]]:
    """
    Probe clips starting one step before est_end and extending past it.
    Returns (last_match_clip_start, probe_list).
    """
    probes = []
    last_match_t: float | None = None
    consec_misses = 0

    probe_origin = max(0.0, est_end - step)

    for i in range(n_probes + 1):
        t = probe_origin + i * step
        if t + clip_len > episode_duration:
            break

        match = await shazam_clip(audio_path, t, t + clip_len)
        await asyncio.sleep(0.3)

        if match and _song_key(match) == song_key:
            last_match_t = t
            consec_misses = 0
            probes.append({"t": round(t, 1), "result": "match",
                           "song": f"{match['artist']} - {match['title']}"})
        else:
            consec_misses += 1
            probes.append({"t": round(t, 1),
                           "result": "no_match" if not match else "different_song",
                           "song": f"{match['artist']} - {match['title']}" if match else None})
            if consec_misses >= 2:
                break

    return last_match_t, probes


# ── Main region processor ──────────────────────────────────────────────────────

def _no_match_result(region: dict) -> dict:
    return {
        **region,
        "matched_song": None, "artist": None, "title": None,
        "shazam_offset": None, "isrc": None,
        "duration_s": None, "duration_source": "none",
        "song_start_in_episode": None, "song_end_in_episode": None,
        "suggested_start": region["start"], "suggested_end": region["end"],
        "n_samples_total": 0, "n_samples_matched": 0, "start_spread_s": 0.0,
        "boundary_confidence": "LOW",
        "start_confidence": "LOW", "end_confidence": "LOW",
        "start_uncertain": True, "end_uncertain": True, "end_info": "",
        "start_snap_note": "", "end_snap_note": "",
        "cut_source": "heuristic_fallback",
        "uncertain": True, "uncertain_reasons": ["no shazam match"],
        "_debug": None,
    }


async def process_region(
    audio_path: Path,
    region: dict,
    episode_duration: float,
    smoothed: list[float],
    words: list[dict] | None = None,
) -> dict:
    r_start, r_end = region["start"], region["end"]
    words = words or []

    # 1. Sample points
    sample_pts = sample_region_points(r_start, r_end)
    print(f"    {len(sample_pts)} samples: {[round(t) for t in sample_pts]}", file=sys.stderr)

    # 2. Run Shazam on each sample
    raw_matches: list[dict] = []
    for i, t in enumerate(sample_pts):
        clip_e = min(t + SAMPLE_CLIP_LEN, episode_duration)
        if clip_e - t < 5.0:
            continue
        match = await shazam_clip(audio_path, t, clip_e)
        if match:
            label = f"{match['artist']} - {match['title']}"
            est_s = round(t - match["offset"], 1)
            print(f"    [{i+1}/{len(sample_pts)}] t={t:.0f}s MATCH: {label} "
                  f"offset={match['offset']:.1f}s → est_start={_hms(est_s)}", file=sys.stderr)
            raw_matches.append({"time": t, **match})
        else:
            print(f"    [{i+1}/{len(sample_pts)}] t={t:.0f}s no match", file=sys.stderr)
        await asyncio.sleep(0.3)

    if not raw_matches:
        return _no_match_result(region)

    # 3. Best song + cluster
    song_key, song_matches = _pick_best_song(raw_matches)
    n  = len(song_matches)
    est_starts = [m["time"] - m["offset"] for m in song_matches]
    median_start = statistics.median(est_starts)
    spread = (max(est_starts) - min(est_starts)) if n > 1 else 0.0

    best = min(song_matches, key=lambda m: m["time"])
    song_label = f"{best['artist']} - {best['title']}"

    print(f"    → {song_label}: {n}/{len(sample_pts)} hits, "
          f"spread={spread:.1f}s, median_start={_hms(median_start)}", file=sys.stderr)

    # 4. Duration
    duration, dur_source = lookup_duration(
        best["artist"], best["title"], best.get("trackadamid", "")
    )
    if duration:
        print(f"    duration={duration:.1f}s via {dur_source}", file=sys.stderr)

    # 5. Acoustic start snap
    snapped_start, start_note = snap_boundary(smoothed, median_start, "start")
    final_start = snapped_start if snapped_start is not None else median_start
    print(f"    start_snap: {start_note}", file=sys.stderr)

    # 5b. Word clawback — only when the acoustic snap found no speech→music edge
    #     (the talk-over-intro signature: host talks over the song intro, so the
    #     music signal never dips to speech level and there is nothing to snap
    #     to). Pull the start forward off host talk using the transcript.
    if snapped_start is None:
        clawed_start, claw_note = clawback_start(final_start, r_end, words)
        if clawed_start is not None:
            final_start = clawed_start
    else:
        claw_note = "clawback: skipped (acoustic snap succeeded)"
    print(f"    clawback:   {claw_note}", file=sys.stderr)

    # 6. Estimate end: duration is reference; end scan finds the truth
    est_end_from_dur     = (median_start + duration) if duration else None
    last_sample_coverage = max(m["time"] for m in song_matches) + SAMPLE_CLIP_LEN

    scan_last_t: float | None = None
    end_scan_probes: list[dict] = []
    end_scan_from_last_sample = False

    if n >= 2 and est_end_from_dur is not None:
        # Multi-hit: probe past the expected song end to confirm / find true end
        print(f"    end scan: probing past {_hms(est_end_from_dur)}", file=sys.stderr)
        scan_last_t, end_scan_probes = await scan_end(
            audio_path, song_key, best, est_end_from_dur, episode_duration
        )
    elif n == 1:
        # Single hit: DB duration is unverified. Sweep forward from last observed
        # sample to find where the song actually stops, ignoring the DB duration.
        end_scan_from_last_sample = True
        print(f"    end scan (n=1): sweeping from {_hms(last_sample_coverage)}", file=sys.stderr)
        scan_last_t, end_scan_probes = await scan_end(
            audio_path, song_key, best, last_sample_coverage, episode_duration,
            n_probes=END_SCAN_PROBES_N1,
        )
    if scan_last_t is not None:
        print(f"    end scan: song still playing at t={scan_last_t:.0f}s", file=sys.stderr)

    # Resolve final end
    end_scan_ran    = len(end_scan_probes) > 0
    end_scan_no_hit = end_scan_ran and scan_last_t is None

    if scan_last_t is not None:
        # End scan confirmed song: use last confirmed clip as floor
        final_end_raw = scan_last_t + SAMPLE_CLIP_LEN
    elif end_scan_no_hit and not end_scan_from_last_sample:
        # n>=2 scan from est_end found nothing → DB duration is wrong version;
        # fall back to last observed sample.
        final_end_raw = last_sample_coverage
    elif est_end_from_dur is not None:
        # n=1 scan failed (unusual) or no scan ran: trust duration, floor at coverage
        final_end_raw = max(est_end_from_dur, last_sample_coverage)
    else:
        final_end_raw = last_sample_coverage

    # 7. Acoustic end snap
    snapped_end, end_note = snap_boundary(smoothed, final_end_raw, "end")
    final_end = snapped_end if snapped_end is not None else final_end_raw
    print(f"    end_snap:   {end_note}", file=sys.stderr)

    # 8. Uncertainty flags — split START vs END.
    #    Rationale (2026-07 recall fix): a DB-duration overshoot is an END-only
    #    signal, and the heuristic region end (r_end) is itself known to run
    #    short (the tail pattern). So a well-clustered START must not be vetoed
    #    just because the END scan confirmed the song extends past that short
    #    r_end. DB duration is now display-only; END trust comes from the
    #    empirical end scan, never from the duration number.
    start_reasons: list[str] = []
    end_reasons:   list[str] = []
    end_info:      list[str] = []   # informational, does NOT lower confidence

    # ── START: can we trust where the song begins? ──
    if n == 1:
        start_reasons.append("single Shazam hit — start is an estimate only")
    if n > 1 and spread > CLUSTER_WARN_S:
        start_reasons.append(
            f"estimated starts spread {spread:.0f}s across {n} samples (>{CLUSTER_WARN_S:.0f}s)"
        )
    if final_start < r_start - UNCERTAIN_OVERSHOOT:
        start_reasons.append(
            f"song_start {final_start:.0f}s is {r_start - final_start:.0f}s before region"
        )
        final_start = r_start

    # ── END: is the endpoint empirically supported? ──
    #   end_confirmed → scan positively heard the song near/after its estimated end
    #   clamp-to-last-sample → DB duration overshot; end anchored to last heard
    #                          music (safe: errs short, leaves a little music)
    #   n=1 sweep miss / no duration → end genuinely unknown
    end_confirmed = scan_last_t is not None
    if end_confirmed and final_end > r_end + UNCERTAIN_OVERSHOOT:
        end_info.append(
            f"end extends {final_end - r_end:.0f}s past heuristic region "
            f"(scan-confirmed — heuristic end ran short)"
        )
    if end_scan_no_hit and not end_scan_from_last_sample:
        # n>=2 scan past DB-duration end found nothing → song stopped earlier;
        # end was anchored to last observed sample. Safe, mildly conservative.
        end_info.append("end anchored to last observed sample (DB duration overshot)")
    if end_scan_no_hit and end_scan_from_last_sample:
        end_reasons.append("n=1 end sweep found no song past last sample — end unreliable")
    if final_end > r_end + UNCERTAIN_OVERSHOOT and not end_confirmed and duration is None:
        overshoot = final_end - r_end
        end_reasons.append(
            f"song_end {final_end:.0f}s is {overshoot:.0f}s past region (no duration; clamped)"
        )
        final_end = r_end + 5.0

    # Backwards cut guard (offset > duration) — hard failure, kills both sides
    reversed_cut = final_end <= final_start
    if reversed_cut:
        start_reasons.append(
            f"reversed boundary (end {final_end:.0f}s ≤ start {final_start:.0f}s) — offset > duration?"
        )
        final_start = r_start
        final_end   = r_end

    # 9. Per-side confidence + cut
    snapped_start_ok = snapped_start is not None
    snapped_end_ok   = snapped_end is not None

    if start_reasons:
        start_confidence = "LOW"
    elif n >= 3 and spread < CLUSTER_AGREE_S:
        start_confidence = "HIGH" if snapped_start_ok else "MEDIUM"
    elif n >= 2 and spread < CLUSTER_WARN_S:
        start_confidence = "MEDIUM"
    else:
        start_confidence = "LOW"

    if reversed_cut or end_reasons:
        end_confidence = "LOW"
    elif end_confirmed:
        end_confidence = "HIGH" if snapped_end_ok else "MEDIUM"
    else:
        end_confidence = "MEDIUM"   # clamped to last observed sample: safe

    start_uncertain = bool(start_reasons)
    end_uncertain   = bool(end_reasons)
    # Backward-compat union flag + reason list (still consumed downstream / display)
    uncertain_reasons = start_reasons + end_reasons

    # Legacy combined confidence (kept for display / older consumers)
    confidence = _boundary_confidence(n, spread, snapped_start_ok, snapped_end_ok)
    cut_start  = round(max(0.0, final_start - PRE_PADDING), 1)
    cut_end    = round(min(episode_duration, final_end + POST_PADDING), 1)

    # 10. Debug payload
    debug = {
        "song":                   song_label,
        "region":                 [r_start, r_end],
        "n_samples_total":        len(sample_pts),
        "all_matches": [
            {
                "time":       round(m["time"], 1),
                "offset":     round(m["offset"], 1),
                "song":       f"{m['artist']} - {m['title']}",
                "est_start":  round(m["time"] - m["offset"], 1),
                "est_start_hms": _hms(m["time"] - m["offset"]),
            }
            for m in raw_matches
        ],
        "best_song_n_hits":       n,
        "estimated_starts":       [round(s, 1) for s in est_starts],
        "median_start":           round(median_start, 1),
        "median_start_hms":       _hms(median_start),
        "start_spread_s":         round(spread, 1),
        "official_duration_s":    round(duration, 1) if duration else None,
        "official_duration_src":  dur_source,
        "last_sample_coverage_s": round(last_sample_coverage, 1),
        "end_scan_probes":        end_scan_probes,
        "est_end_from_duration":  round(est_end_from_dur, 1) if est_end_from_dur else None,
        "scan_last_match_t":      round(scan_last_t, 1) if scan_last_t else None,
        "final_end_raw":          round(final_end_raw, 1),
        "start_snap_note":        start_note,
        "clawback_note":          claw_note,
        "end_snap_note":          end_note,
        "final_start":            round(final_start, 1),
        "final_end":              round(final_end, 1),
        "cut_start":              cut_start,
        "cut_end":                cut_end,
        "boundary_confidence":    confidence,
        "start_confidence":       start_confidence,
        "end_confidence":         end_confidence,
        "start_uncertain":        start_uncertain,
        "end_uncertain":          end_uncertain,
        "end_info":               end_info,
        "uncertain_reasons":      uncertain_reasons,
    }

    return {
        **region,
        "matched_song":          song_label,
        "artist":                best["artist"],
        "title":                 best["title"],
        "shazam_offset":         round(best["offset"], 1),
        "isrc":                  best.get("isrc", ""),
        "duration_s":            round(duration, 1) if duration else None,
        "duration_source":       dur_source,
        "song_start_in_episode": round(final_start, 1),
        "song_end_in_episode":   round(final_end, 1),
        "suggested_start":       cut_start,
        "suggested_end":         cut_end,
        "n_samples_total":       len(sample_pts),
        "n_samples_matched":     n,
        "start_spread_s":        round(spread, 1),
        "boundary_confidence":   confidence,
        "start_confidence":      start_confidence,
        "end_confidence":        end_confidence,
        "start_uncertain":       start_uncertain,
        "end_uncertain":         end_uncertain,
        "end_info":              "; ".join(end_info),
        "start_snap_note":       start_note,
        "end_snap_note":         end_note,
        "cut_source":            f"multi_shazam_{n}_of_{len(sample_pts)}_samples",
        "uncertain":             start_uncertain or end_uncertain,
        "uncertain_reasons":     uncertain_reasons,
        "_debug":                debug,
    }


# ── Orchestration ──────────────────────────────────────────────────────────────

async def run_all(
    audio_path: Path,
    candidates: list,
    episode_duration: float,
    smoothed: list[float],
    words: list[dict] | None = None,
) -> list:
    results = []
    for i, region in enumerate(candidates):
        print(
            f"\n  [{i+1}/{len(candidates)}] {region['start']:.0f}–{region['end']:.0f}s",
            file=sys.stderr,
        )
        result = await process_region(audio_path, region, episode_duration, smoothed, words)
        results.append(result)
    return results


# ── Output writers ─────────────────────────────────────────────────────────────

MATCH_FIELDS = [
    "index", "region_start", "region_end",
    "matched_song", "shazam_offset",
    "n_samples_matched", "n_samples_total", "start_spread_s",
    "boundary_confidence", "start_confidence", "end_confidence",
    "start_uncertain", "end_uncertain", "end_info",
    "duration_source", "duration_s",
    "song_start_in_episode", "song_end_in_episode",
    "suggested_start", "suggested_end", "suggested_start_hms", "suggested_end_hms",
    "start_snap_note", "end_snap_note",
    "cut_source", "uncertain", "uncertain_reasons", "decision",
]

UNMATCH_FIELDS = ["index", "region_start", "region_end", "uncertain_reasons"]


def _hms(s: float) -> str:
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{sec:05.2f}"


def write_outputs(results: list, out_dir: Path, episode_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    matched   = [r for r in results if r["matched_song"]]
    unmatched = [r for r in results if not r["matched_song"]]

    # song_matches_<name>.csv
    matches_csv = out_dir / f"song_matches_{episode_name}.csv"
    with matches_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(MATCH_FIELDS)
        for i, r in enumerate(matched):
            decision = "review (uncertain)" if r["uncertain"] else "review (matched)"
            writer.writerow([
                i, r["start"], r["end"],
                r["matched_song"], r["shazam_offset"],
                r["n_samples_matched"], r["n_samples_total"], r["start_spread_s"],
                r["boundary_confidence"], r["start_confidence"], r["end_confidence"],
                r["start_uncertain"], r["end_uncertain"], r.get("end_info", ""),
                r["duration_source"], r["duration_s"],
                r["song_start_in_episode"], r["song_end_in_episode"],
                r["suggested_start"], r["suggested_end"],
                _hms(r["suggested_start"]), _hms(r["suggested_end"]),
                r.get("start_snap_note", ""), r.get("end_snap_note", ""),
                r["cut_source"], r["uncertain"],
                "; ".join(r.get("uncertain_reasons", [])),
                decision,
            ])

    # song_matches_<name>.json (strip _debug for cleaner JSON)
    matches_json = out_dir / f"song_matches_{episode_name}.json"
    clean = [{k: v for k, v in r.items() if k != "_debug"} for r in matched]
    matches_json.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")

    # boundary_debug_<name>.json
    debug_json = out_dir / f"boundary_debug_{episode_name}.json"
    debug_records = [r["_debug"] for r in results if r.get("_debug")]
    debug_json.write_text(json.dumps(debug_records, indent=2, ensure_ascii=False), encoding="utf-8")

    # Audacity cut suggestions (skip backwards)
    aud = out_dir / f"song_cut_suggestions_{episode_name}_audacity.txt"
    with aud.open("w", encoding="utf-8") as f:
        for r in results:
            ss, se = r["suggested_start"], r["suggested_end"]
            if se <= ss:
                continue
            flag  = " [UNCERTAIN]" if r["uncertain"] else ""
            label = (r["matched_song"] or "unmatched") + flag
            f.write(f"{ss:.3f}\t{se:.3f}\t{label}\n")

    # song_unmatched_<name>.csv
    unmatch_csv = out_dir / f"song_unmatched_{episode_name}.csv"
    with unmatch_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(UNMATCH_FIELDS)
        for i, r in enumerate(unmatched):
            writer.writerow([i, r["start"], r["end"],
                             "; ".join(r.get("uncertain_reasons", []))])

    print(f"\nOutputs written to {out_dir}/", file=sys.stderr)
    for p in [matches_csv, matches_json, debug_json, aud, unmatch_csv]:
        print(f"  {p.name}", file=sys.stderr)


def print_summary_table(results: list, episode_name: str) -> None:
    matched = [r for r in results if r["matched_song"]]
    sep = "─" * 120
    print(f"\n{sep}", file=sys.stderr)
    print(f"  {episode_name}  —  {len(matched)}/{len(results)} matched", file=sys.stderr)
    print(sep, file=sys.stderr)
    hdr = (f"  {'Region':>15}  {'Song':<40}  {'Hits':>4}  {'Spread':>7}  "
           f"{'Conf':<6}  {'Cut start':>10}  {'Cut end':>10}  Note")
    print(hdr, file=sys.stderr)
    print(sep, file=sys.stderr)
    for r in results:
        region = f"{r['start']:.0f}–{r['end']:.0f}s"
        song   = (r["matched_song"] or "(unmatched)")[:40]
        hits   = f"{r['n_samples_matched']}/{r['n_samples_total']}" if r["matched_song"] else "—"
        spread = f"{r['start_spread_s']:.0f}s" if r.get("start_spread_s") else "—"
        conf   = r.get("boundary_confidence", "—")
        cs     = _hms(r["suggested_start"])
        ce     = _hms(r["suggested_end"])
        note   = "UNCERTAIN" if r["uncertain"] else "ok"
        ss, se = r["suggested_start"], r["suggested_end"]
        if r["matched_song"] and se <= ss:
            note = "REVERSED (false match?)"
        print(f"  {region:>15}  {song:<40}  {hits:>4}  {spread:>7}  "
              f"{conf:<6}  {cs:>10}  {ce:>10}  {note}", file=sys.stderr)
    print(f"{sep}\n", file=sys.stderr)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio",        type=Path, required=True)
    parser.add_argument("--candidates",   type=Path, required=True)
    parser.add_argument("--features",     type=Path, default=None,
                        help="data/features/<episode>.json — for acoustic signal + duration")
    parser.add_argument("--transcript",   type=Path, default=None,
                        help="data/transcripts/<episode>.json — for no_speech_prob signal")
    parser.add_argument("--episode-name", required=True)
    parser.add_argument("--out-dir",      type=Path, default=Path("data/labels"))
    args = parser.parse_args()

    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))

    if args.features:
        feat_data = json.loads(args.features.read_text(encoding="utf-8"))
        episode_duration  = feat_data["duration"]
        features_windows  = feat_data.get("windows", [])
    else:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(args.audio)],
            capture_output=True, text=True, check=True,
        )
        episode_duration = float(probe.stdout.strip())
        features_windows = []

    transcript_segments: list[dict] = []
    words: list[dict] = []
    if args.transcript:
        tr = json.loads(args.transcript.read_text(encoding="utf-8"))
        transcript_segments = tr.get("segments", [])
        for seg in transcript_segments:
            for w in seg.get("words", []):
                if "start" in w and "end" in w:
                    words.append({"start": w["start"], "end": w["end"]})
        words.sort(key=lambda w: w["start"])

    print(f"Building acoustic signal ({len(features_windows)} feature windows, "
          f"{len(transcript_segments)} transcript segments, {len(words)} words) ...",
          file=sys.stderr)
    raw_signal = build_music_signal(features_windows, transcript_segments, episode_duration)
    smoothed   = smooth_signal(raw_signal)

    print(f"Scanning {len(candidates)} candidate regions in {args.audio.name} ...", file=sys.stderr)
    results = asyncio.run(run_all(args.audio, candidates, episode_duration, smoothed, words))

    print_summary_table(results, args.episode_name)
    write_outputs(results, args.out_dir, args.episode_name)

    n_matched   = sum(1 for r in results if r["matched_song"])
    n_uncertain = sum(1 for r in results if r["uncertain"])
    n_high = sum(1 for r in results if r.get("boundary_confidence") == "HIGH")
    n_med  = sum(1 for r in results if r.get("boundary_confidence") == "MEDIUM")
    n_low  = sum(1 for r in results if r.get("boundary_confidence") == "LOW")
    print(
        f"{len(results)} regions  |  {n_matched} matched  |  {n_uncertain} uncertain  |  "
        f"confidence: {n_high} HIGH / {n_med} MEDIUM / {n_low} LOW",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
