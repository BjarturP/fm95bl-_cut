"""
Combine transcript keywords + acoustic features into scored candidate
ad/music-break segments, each annotated with *why* it was flagged.

Usage:
    python detect_breaks.py \\
        --transcript data/transcripts/episode1.json \\
        --features data/features/episode1.json \\
        --out data/candidates/episode1.json

Philosophy: bias toward recall here (over-flag rather than miss things) --
review_export.py + your manual review is the precision filter, since a
missed ad (false negative) is much cheaper than a wrongly-cut conversation
(false positive).
"""
import argparse
import bisect
import json
import sys
from pathlib import Path

import config


def word_rate_series(transcript: dict, timestamps: list, window: float) -> list:
    """Rolling words-per-second centered at each timestamp, using transcript
    word start times. Low word-rate stretches usually mean music/no host
    talking, even when the acoustic music_score alone doesn't catch it."""
    word_starts = sorted(
        w["start"] for seg in transcript["segments"] for w in seg.get("words", [])
    )
    rates = []
    for t in timestamps:
        lo = bisect.bisect_left(word_starts, t - window)
        hi = bisect.bisect_right(word_starts, t + window)
        rates.append((hi - lo) / (2 * window))
    return rates


def no_speech_prob_series(transcript: dict, timestamps: list, window: float) -> list:
    """Rolling average of Whisper's per-segment no_speech_prob centered at
    each timestamp. Defaults to 1.0 (maximally "not speech") for windows
    with no transcribed segments nearby -- no segment transcribed there at
    all is itself evidence of non-speech for this signal, unlike
    word_rate_series where 0 words just means "didn't catch any."""
    segs = sorted(transcript["segments"], key=lambda s: s["start"])
    seg_starts = [s["start"] for s in segs]
    out = []
    for t in timestamps:
        lo = bisect.bisect_left(seg_starts, t - window)
        hi = bisect.bisect_right(seg_starts, t + window)
        vals = [
            segs[i]["no_speech_prob"]
            for i in range(lo, hi)
            if segs[i]["end"] > t - window and segs[i]["start"] < t + window
        ]
        out.append(sum(vals) / len(vals) if vals else 1.0)
    return out


def build_combined_timeline(transcript: dict, features: dict) -> tuple:
    """Shared (timestamps, rate_norm, combined) arrays used both for
    music-run detection and for the merge/expand post-processing below -- one
    signal, reused everywhere instead of recomputed differently in different
    places.

    The non-speech component is max(1 - rate_norm, no_speech_prob), not
    word-rate alone: sung lyrics that Whisper transcribes as if they were
    spoken words defeat word-rate (it reads as real speech) but tend to have
    high no_speech_prob (Whisper's own estimate that the segment isn't
    speech), and either signal alone catching it is enough. Validated via
    experiments/no_speech_prob.py (variant "nsp_or") against the one labeled
    episode: improved final-cuts recall 73.33% -> 80.00% AND precision
    84.62% -> 90.91% over plain word-rate. (A similar attempt using
    avg_logprob/transcription-confidence instead was tried and reverted --
    it added a false positive with no recall gain; no_speech_prob is a more
    direct VAD-style signal and behaved differently.)"""
    windows = features["windows"]
    timestamps = [w["t"] for w in windows]
    rates = word_rate_series(transcript, timestamps, config.WORD_RATE_WINDOW)
    rate_norm = [min(r / config.WORD_RATE_NORM_CAP, 1.0) for r in rates]
    nsp = no_speech_prob_series(transcript, timestamps, config.WORD_RATE_WINDOW)
    non_speech = [max(1 - rn, n) for rn, n in zip(rate_norm, nsp)]
    combined = [
        config.MUSIC_COMBINE_WEIGHT_ACOUSTIC * w["music_score"]
        + config.MUSIC_COMBINE_WEIGHT_WORDRATE * ns
        for w, ns in zip(windows, non_speech)
    ]
    return timestamps, rate_norm, combined


def compute_music_runs(transcript: dict, features: dict) -> list:
    """Combine smoothed acoustic music_score with inverse word-rate into one
    score, then extract runs via hysteresis thresholding. See config.py for
    why music_score alone isn't used.

    Hysteresis (separate enter/exit thresholds) instead of a single cutoff:
    a song with occasional talk-over or a quiet bridge dips the combined
    score periodically, which fragments a single ~5min song into many tiny
    candidates under one threshold. Requiring the score to drop further
    (below MUSIC_EXIT_THRESHOLD) before ending a run keeps those dips from
    splitting one real music block into pieces.
    """
    timestamps, _, combined = build_combined_timeline(transcript, features)

    runs = []
    in_run = False
    run_start = None
    for t, score in zip(timestamps, combined):
        if not in_run and score >= config.MUSIC_ENTER_THRESHOLD:
            in_run = True
            run_start = t
        elif in_run and score < config.MUSIC_EXIT_THRESHOLD:
            runs.append({"start": run_start, "end": t})
            in_run = False
    if in_run:
        runs.append({"start": run_start, "end": timestamps[-1]})

    return [r for r in runs if (r["end"] - r["start"]) >= config.MUSIC_MIN_DURATION]


def find_keyword_occurrences(transcript: dict) -> list:
    """Return [{"t": float, "phrase": str}] for each keyword phrase found in
    the transcript, timestamped at the start of the segment it occurs in."""
    occurrences = []
    for seg in transcript["segments"]:
        text_lower = seg["text"].lower()
        for phrase in config.BREAK_KEYWORDS:
            if phrase.lower() in text_lower:
                occurrences.append({"t": seg["start"], "phrase": phrase})
    return occurrences


def merge_overlapping_runs(runs: list, tag: str) -> list:
    """Tag each run dict with its source signal name."""
    return [{**r, "signals": {tag}} for r in runs]


def union_runs(*run_lists: list) -> list:
    """Union a set of {start,end,signals} runs into merged, non-overlapping
    spans, keeping track of which signals contributed to each."""
    events = []
    for runs in run_lists:
        for r in runs:
            events.append((r["start"], r["end"], r["signals"]))
    if not events:
        return []
    events.sort(key=lambda e: e[0])

    merged = []
    cur_start, cur_end, cur_signals = events[0][0], events[0][1], set(events[0][2])
    for start, end, signals in events[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
            cur_signals |= set(signals)
        else:
            merged.append({"start": cur_start, "end": cur_end, "signals": cur_signals})
            cur_start, cur_end, cur_signals = start, end, set(signals)
    merged.append({"start": cur_start, "end": cur_end, "signals": cur_signals})
    return merged


def near_loudness_jump(start: float, end: float, jumps: list, tolerance: float = 2.0) -> bool:
    return any(start - tolerance <= j["t"] <= end + tolerance for j in jumps)


def supporting_keywords(start: float, end: float, occurrences: list, window: float) -> list:
    return [
        occ
        for occ in occurrences
        if (start - window) <= occ["t"] <= (end + window)
    ]


def score_run(run: dict, occurrences: list, jumps: list) -> dict:
    reasons = []
    confidence = 0.0

    if "silence" in run["signals"]:
        reasons.append(f"silence:{run['end'] - run['start']:.1f}s")
        confidence += config.WEIGHT_SILENCE
    if "music" in run["signals"]:
        reasons.append(f"music-like:{run['end'] - run['start']:.1f}s")
        confidence += config.WEIGHT_MUSIC

    kws = supporting_keywords(run["start"], run["end"], occurrences, config.KEYWORD_SUPPORT_WINDOW)
    for kw in kws:
        reasons.append(f'keyword:"{kw["phrase"]}"')
    if kws:
        confidence += config.WEIGHT_KEYWORD

    if near_loudness_jump(run["start"], run["end"], jumps):
        reasons.append("loudness-jump")
        confidence += config.WEIGHT_LOUDNESS_JUMP

    return {
        "start": round(run["start"], 2),
        "end": round(run["end"], 2),
        "duration": round(run["end"] - run["start"], 2),
        "confidence": round(min(confidence, 1.0), 3),
        "reasons": reasons,
    }


def keyword_only_candidates(occurrences: list, supported_occurrences: set) -> list:
    candidates = []
    for occ in occurrences:
        key = (occ["t"], occ["phrase"])
        if key in supported_occurrences:
            continue
        start = max(0.0, occ["t"] - config.KEYWORD_PREROLL)
        end = occ["t"] + config.KEYWORD_POSTROLL
        candidates.append(
            {
                "start": round(start, 2),
                "end": round(end, 2),
                "duration": round(end - start, 2),
                "confidence": round(min(config.WEIGHT_KEYWORD, config.KEYWORD_ONLY_CONFIDENCE_CAP), 3),
                "reasons": [f'keyword-only:"{occ["phrase"]}" (no acoustic support)'],
            }
        )
    return candidates


def merge_candidates(candidates: list) -> list:
    """Merge candidates that overlap or sit within a 5s gap of each other,
    unioning reasons and taking the max confidence."""
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda c: c["start"])
    merged = [dict(candidates[0])]
    for c in candidates[1:]:
        last = merged[-1]
        if c["start"] <= last["end"] + 5.0:
            last["end"] = max(last["end"], c["end"])
            last["duration"] = round(last["end"] - last["start"], 2)
            last["confidence"] = max(last["confidence"], c["confidence"])
            last["reasons"] = list(dict.fromkeys(last["reasons"] + c["reasons"]))
        else:
            merged.append(dict(c))
    return merged


def _avg_in_range(t0: float, t1: float, timestamps: list, values: list) -> float:
    lo = bisect.bisect_left(timestamps, t0)
    hi = bisect.bisect_right(timestamps, t1)
    vals = values[lo:hi]
    return sum(vals) / len(vals) if vals else 0.0


def merge_nearby_candidates(candidates: list, timestamps: list, rate_norm: list) -> list:
    """Bridge candidates separated by a short gap into one block, UNLESS that
    gap contains real host speech (checked via word-rate, not just gap size).
    Real ad/music breaks are usually one continuous block; the per-window
    detector above fragments them at every brief dip below threshold."""
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda c: c["start"])
    merged = [dict(candidates[0])]
    for c in candidates[1:]:
        last = merged[-1]
        gap = c["start"] - last["end"]
        if gap <= 0:
            bridge = True
        elif gap <= config.MERGE_GAP_ALWAYS_BELOW:
            bridge = True
        elif gap <= config.MERGE_GAP_MAX:
            avg_rate_norm = _avg_in_range(last["end"], c["start"], timestamps, rate_norm)
            bridge = avg_rate_norm < config.MERGE_GAP_SPEECH_RATE_NORM
        else:
            bridge = False

        if bridge:
            if gap > 0:
                last["reasons"].append(f"merged-gap:{gap:.0f}s")
            last["end"] = max(last["end"], c["end"])
            last["confidence"] = max(last["confidence"], c["confidence"])
            last["reasons"] = list(dict.fromkeys(last["reasons"] + c["reasons"]))
        else:
            merged.append(dict(c))
    for c in merged:
        c["duration"] = round(c["end"] - c["start"], 2)
    return merged


def expand_boundaries(candidates: list, timestamps: list, combined: list) -> list:
    """Grow each candidate's start/end outward through ambiguous audio until
    hitting a sustained run of clear host speech. The per-window detector's
    boundaries mark only the most-confident core of a break; real breaks
    extend further into quieter intros/outros that don't clear the entry
    threshold on their own."""
    n = len(timestamps)
    hop = timestamps[1] - timestamps[0] if n > 1 else 1.0
    max_steps = int(config.EXPAND_MAX_SECONDS / hop)
    sustain_steps = max(1, int(config.EXPAND_CLEAR_SPEECH_SUSTAIN_SECONDS / hop))

    def index_at_or_before(t):
        i = bisect.bisect_right(timestamps, t) - 1
        return max(0, min(i, n - 1))

    def scan(start_idx: int, direction: int) -> int:
        i = start_idx
        last_nonspeech = start_idx
        consec_speech = 0
        steps = 0
        while steps < max_steps:
            i += direction
            if i < 0 or i >= n:
                break
            steps += 1
            if combined[i] < config.EXPAND_CLEAR_SPEECH_THRESHOLD:
                consec_speech += 1
                if consec_speech >= sustain_steps:
                    break
            else:
                consec_speech = 0
                last_nonspeech = i
        return last_nonspeech

    expanded = []
    for c in candidates:
        start_idx = index_at_or_before(c["start"])
        end_idx = index_at_or_before(c["end"])
        new_start_idx = scan(start_idx, -1)
        new_end_idx = scan(end_idx, +1)
        c = dict(c)
        if timestamps[new_start_idx] < c["start"]:
            c["start"] = timestamps[new_start_idx]
            c["reasons"] = c["reasons"] + ["expanded-start"]
        if timestamps[new_end_idx] > c["end"]:
            c["end"] = timestamps[new_end_idx]
            c["reasons"] = c["reasons"] + ["expanded-end"]
        c["duration"] = round(c["end"] - c["start"], 2)
        expanded.append(c)
    return expanded


def postprocess(candidates: list, transcript: dict, features: dict) -> list:
    """Raw per-window candidates -> merged, boundary-expanded final cuts.
    Runs merge -> expand -> merge again since expansion can close gaps that
    were too wide to bridge on the first pass."""
    timestamps, rate_norm, combined = build_combined_timeline(transcript, features)
    merged = merge_nearby_candidates(candidates, timestamps, rate_norm)
    expanded = expand_boundaries(merged, timestamps, combined)
    return merge_nearby_candidates(expanded, timestamps, rate_norm)


def detect(transcript: dict, features: dict) -> list:
    occurrences = find_keyword_occurrences(transcript)

    silence_runs = merge_overlapping_runs(features["silence_runs"], "silence")
    music_runs = merge_overlapping_runs(compute_music_runs(transcript, features), "music")
    non_speech_runs = union_runs(silence_runs, music_runs)

    supported_occurrences = set()
    scored = []
    for run in non_speech_runs:
        kws = supporting_keywords(run["start"], run["end"], occurrences, config.KEYWORD_SUPPORT_WINDOW)
        for kw in kws:
            supported_occurrences.add((kw["t"], kw["phrase"]))
        scored.append(score_run(run, occurrences, features["loudness_jumps"]))

    scored += keyword_only_candidates(occurrences, supported_occurrences)
    return merge_candidates(scored)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="final merged/expanded candidates")
    parser.add_argument("--out-raw", type=Path, default=None, help="pre-merge per-window candidates")
    args = parser.parse_args()

    transcript = json.loads(args.transcript.read_text(encoding="utf-8"))
    features = json.loads(args.features.read_text(encoding="utf-8"))

    raw = detect(transcript, features)
    final = postprocess(raw, transcript, features)

    out_path = args.out or Path("data/candidates") / args.transcript.name
    out_raw_path = args.out_raw or Path("data/candidates") / (args.transcript.stem + "_raw.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    out_raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Raw: {len(raw)} candidates -> {out_raw_path}", file=sys.stderr)
    print(f"Final (merged+expanded): {len(final)} candidates -> {out_path}", file=sys.stderr)
    for c in final:
        print(f"  [{c['start']:7.1f}s - {c['end']:7.1f}s] conf={c['confidence']:.2f} {c['reasons']}", file=sys.stderr)


if __name__ == "__main__":
    main()
