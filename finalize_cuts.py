"""
Stage 3: turn the merged/expanded "review" candidates from detect_breaks.py
into tighter, more conservative FINAL CUTS.

Why a separate stage: detect_breaks.py's postprocess() is deliberately
generous (merge nearby candidates, expand boundaries outward) so nothing
gets missed -- but calibration showed that generosity costs precision:
blocks expanded into ambiguous audio further than is safe to actually cut
on, and standalone weak keyword hits got treated as real candidates even
with no acoustic support nearby. This stage is the precision pass:

1. Drop low-confidence keyword-only candidates that don't actually bridge
   two strong candidates (they're usually just the host saying a brand/
   station name in normal conversation, not a real break).
2. Merge what's left with a tiered gap rule: short gaps always bridge,
   medium gaps only bridge long blocks with no real host talk between them,
   and the gap right before the literal end of the recording gets extra
   leeway (end-of-show breaks often run straight through to EOF).
3. Trim (never re-expand) the boundaries of any surviving low-confidence
   candidate back toward its strongest evidence, instead of trusting the
   full extent stage 2 expanded it to.

Usage:
    python finalize_cuts.py \\
        --candidates data/candidates/episode1.json \\
        --transcript data/transcripts/episode1.json \\
        --features data/features/episode1.json
"""
import argparse
import bisect
import json
import sys
from pathlib import Path

import config
import detect_breaks as db


def is_strong(c: dict) -> bool:
    return c["confidence"] >= config.FINAL_STRONG_CONFIDENCE


def has_acoustic_support(c: dict) -> bool:
    """True if the candidate has any real silence/music/loudness evidence --
    including evidence merged in from a neighboring candidate during stage 2
    -- as opposed to having been seeded purely from a keyword hit."""
    return any(
        r.startswith("music-like") or r.startswith("silence") or r == "loudness-jump"
        for r in c["reasons"]
    )


def is_keyword_only(c: dict) -> bool:
    """A candidate counts as keyword-only only if it has NO acoustic support
    at all, even after stage-2 merging -- a keyword hit that got merged with
    a real music-like/silence run is no longer "just a keyword" in substance,
    even though its reasons list still mentions how it originated."""
    return any("keyword-only" in r for r in c["reasons"]) and not has_acoustic_support(c)


def drop_unconnected_keyword_only(candidates: list) -> tuple:
    """Rule 3: a standalone keyword hit with no acoustic support is too weak
    to cut on its own -- only keep it if it sits close enough to strong
    candidates on both sides that it's really just bridging one real break."""
    kept, dropped = [], []
    for i, c in enumerate(candidates):
        if not is_keyword_only(c) or is_strong(c):
            kept.append(c)
            continue
        prev_strong = next((p for p in reversed(candidates[:i]) if is_strong(p)), None)
        next_strong = next((n for n in candidates[i + 1:] if is_strong(n)), None)
        bridges = (
            prev_strong is not None
            and next_strong is not None
            and (c["start"] - prev_strong["end"]) <= config.FINAL_MERGE_GAP_LONG_BLOCK
            and (next_strong["start"] - c["end"]) <= config.FINAL_MERGE_GAP_LONG_BLOCK
        )
        if bridges:
            kept.append(c)
        else:
            dropped.append(c)
    return kept, dropped


def gap_has_host_speech(t0: float, t1: float, timestamps: list, rate_norm: list) -> bool:
    return db._avg_in_range(t0, t1, timestamps, rate_norm) >= config.FINAL_MERGE_GAP_SPEECH_RATE_NORM


def merge_final(candidates: list, timestamps: list, rate_norm: list, episode_duration: float) -> list:
    """Tiered gap merge (rules 1, 6): short gaps always bridge, medium gaps
    only bridge long blocks with no real talk in between, and the final gap
    before the end of the file gets extra leeway since these breaks often
    run straight through to the literal end of the recording."""
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda c: c["start"])
    merged = [dict(candidates[0])]
    for c in candidates[1:]:
        last = merged[-1]
        gap = c["start"] - last["end"]
        near_eof = (c is candidates[-1]) and (episode_duration - c["end"]) <= config.FINAL_EOF_TAIL_SECONDS

        if gap <= 0:
            bridge = True
        elif gap <= config.FINAL_MERGE_GAP_DEFAULT:
            bridge = True
        elif near_eof and gap <= config.FINAL_EOF_MERGE_GAP_MAX:
            # No host-speech gate here on purpose: a candidate sitting this
            # close to the literal end of the recording is almost certainly
            # part of the same outro break running through to EOF, even if
            # a stray garbled transcript segment in the gap reads as "talk."
            bridge = True
        elif gap <= config.FINAL_MERGE_GAP_LONG_BLOCK:
            long_block = (
                (last["end"] - last["start"]) >= config.FINAL_MERGE_LONG_BLOCK_MIN_DURATION
                or (c["end"] - c["start"]) >= config.FINAL_MERGE_LONG_BLOCK_MIN_DURATION
            )
            bridge = long_block and not gap_has_host_speech(last["end"], c["start"], timestamps, rate_norm)
        else:
            bridge = False

        if bridge:
            if gap > 0:
                last["reasons"].append(f"final-merged-gap:{gap:.0f}s")
            last["end"] = max(last["end"], c["end"])
            last["confidence"] = max(last["confidence"], c["confidence"])
            last["reasons"] = list(dict.fromkeys(last["reasons"] + c["reasons"]))
        else:
            merged.append(dict(c))
    for c in merged:
        c["duration"] = round(c["end"] - c["start"], 2)
    return merged


def _trim_index(idx: int, direction: int, timestamps: list, combined: list, max_lead_in_steps: int) -> int:
    """Walk from idx outward looking for the first point with real evidence
    (combined score above threshold). If that point is further away than the
    allowed lead-in AND the stretch in between isn't itself continuously
    music/ad-like, trim back to evidence + a small lead-in margin instead of
    trusting the full original extent (rules 4 and 5)."""
    n = len(timestamps)
    i = idx
    steps = 0
    evidence_idx = None
    while 0 <= i < n and steps <= max_lead_in_steps * 4:
        if combined[i] >= config.FINAL_CONTINUOUS_EVIDENCE_THRESHOLD:
            evidence_idx = i
            break
        i += direction
        steps += 1
    if evidence_idx is None:
        return idx  # no real evidence found nearby at all; leave as-is

    lead_in_steps = abs(evidence_idx - idx)
    if lead_in_steps <= max_lead_in_steps:
        return idx  # short lead-in already, nothing to trim

    lo, hi = (idx, evidence_idx) if direction > 0 else (evidence_idx, idx)
    stretch = combined[lo:hi + 1]
    if stretch and (sum(stretch) / len(stretch)) >= config.FINAL_CONTINUOUS_EVIDENCE_THRESHOLD * 0.85:
        return idx  # continuous music/ad audio the whole way -- the wide boundary is justified

    trimmed = evidence_idx - direction * max_lead_in_steps
    return max(0, min(trimmed, n - 1))


def trim_final_boundaries(candidates: list, timestamps: list, combined: list) -> list:
    n = len(timestamps)
    hop = timestamps[1] - timestamps[0] if n > 1 else 1.0
    max_lead_in_steps = max(1, int(config.FINAL_MAX_LEAD_IN_SECONDS / hop))

    result = []
    for c in candidates:
        if c["confidence"] >= config.FINAL_TRIM_LOW_CONFIDENCE:
            result.append(c)
            continue
        start_idx = max(0, min(bisect.bisect_left(timestamps, c["start"]), n - 1))
        end_idx = max(0, min(bisect.bisect_right(timestamps, c["end"]) - 1, n - 1))

        skip_start_trim = c["start"] < config.FINAL_TRIM_SKIP_NEAR_START_SECONDS
        new_start_idx = start_idx if skip_start_trim else _trim_index(start_idx, +1, timestamps, combined, max_lead_in_steps)
        new_end_idx = _trim_index(end_idx, -1, timestamps, combined, max_lead_in_steps)

        c = dict(c)
        if timestamps[new_start_idx] > c["start"]:
            c["start"] = timestamps[new_start_idx]
            c["reasons"] = c["reasons"] + ["trimmed-start (low confidence)"]
        if timestamps[new_end_idx] < c["end"]:
            c["end"] = timestamps[new_end_idx]
            c["reasons"] = c["reasons"] + ["trimmed-end (low confidence)"]
        c["duration"] = round(c["end"] - c["start"], 2)
        if c["end"] > c["start"]:
            result.append(c)
    return result


def finalize(review_candidates: list, transcript: dict, features: dict) -> dict:
    timestamps, rate_norm, combined = db.build_combined_timeline(transcript, features)
    episode_duration = features["duration"]

    # Trim BEFORE merging -- each candidate's own original confidence is the
    # signal for whether it deserves trimming. Merging first would let a
    # weak piece "borrow" a strong neighbor's confidence (max of the two)
    # and skip trimming it should have gotten.
    kept, dropped = drop_unconnected_keyword_only(review_candidates)
    trimmed = trim_final_boundaries(kept, timestamps, combined)
    merged = merge_final(trimmed, timestamps, rate_norm, episode_duration)
    final_cuts = merged

    uncertain = [c for c in final_cuts if c["confidence"] < config.FINAL_STRONG_CONFIDENCE]
    confident = [c for c in final_cuts if c["confidence"] >= config.FINAL_STRONG_CONFIDENCE]

    return {
        "merged": merged,
        "final_cuts": final_cuts,
        "confident_cuts": confident,
        "uncertain_review_zones": uncertain,
        "false_positives_removed": dropped,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True,
                         help="merged/expanded 'review' candidates from detect_breaks.py")
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="final cuts (confident + uncertain, combined)")
    parser.add_argument("--out-dropped", type=Path, default=None, help="false positives removed")
    parser.add_argument("--out-uncertain", type=Path, default=None, help="uncertain review zones (subset of final cuts)")
    args = parser.parse_args()

    review_candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    transcript = json.loads(args.transcript.read_text(encoding="utf-8"))
    features = json.loads(args.features.read_text(encoding="utf-8"))

    result = finalize(review_candidates, transcript, features)

    stem = args.candidates.stem
    out_path = args.out or Path("data/candidates") / f"{stem}_finalcuts.json"
    out_dropped_path = args.out_dropped or Path("data/candidates") / f"{stem}_dropped.json"
    out_uncertain_path = args.out_uncertain or Path("data/candidates") / f"{stem}_uncertain.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(json.dumps(result["final_cuts"], ensure_ascii=False, indent=2), encoding="utf-8")
    out_dropped_path.write_text(json.dumps(result["false_positives_removed"], ensure_ascii=False, indent=2), encoding="utf-8")
    out_uncertain_path.write_text(json.dumps(result["uncertain_review_zones"], ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Input (review candidates): {len(review_candidates)}", file=sys.stderr)
    print(f"Dropped as unsupported keyword-only false positives ({len(result['false_positives_removed'])}) -> {out_dropped_path}:", file=sys.stderr)
    for c in result["false_positives_removed"]:
        print(f"  DROPPED [{c['start']:7.1f}s - {c['end']:7.1f}s] conf={c['confidence']:.2f} {c['reasons']}", file=sys.stderr)
    print(f"Final cuts ({len(result['final_cuts'])}: {len(result['confident_cuts'])} confident, "
          f"{len(result['uncertain_review_zones'])} uncertain) -> {out_path}", file=sys.stderr)
    for c in result["final_cuts"]:
        tag = "CONFIDENT" if c["confidence"] >= config.FINAL_STRONG_CONFIDENCE else "UNCERTAIN"
        print(f"  [{tag:9s}] [{c['start']:7.1f}s - {c['end']:7.1f}s] conf={c['confidence']:.2f} {c['reasons']}", file=sys.stderr)
    print(f"Uncertain review zones -> {out_uncertain_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
