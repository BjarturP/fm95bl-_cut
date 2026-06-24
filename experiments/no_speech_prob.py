"""
Standalone experiment: does Whisper's per-segment no_speech_prob improve
break detection over the current word-rate + acoustic music_score combo?

This does NOT touch detect_breaks.py / config.py / finalize_cuts.py. It
re-derives an alternative "combined" music-likeness score using functions
imported from detect_breaks.py, runs that alternative through the same
postprocess (merge/expand) + finalize_cuts logic, and A/B compares against
the real pipeline output on the one labeled episode. Promote a variant into
the main pipeline only if it's run through here first and shown to help.

Background: avg_logprob (transcription confidence) was tried for the same
purpose and reverted -- see README "Calibration results" -- because
per-window noise wiped out the separation seen in 90s-clip averages.
no_speech_prob is a more direct VAD-style signal (Whisper's own estimate of
whether a segment contains speech at all) that hasn't been tried in the
real pipeline yet.

Usage:
    python experiments/no_speech_prob.py \\
        --transcript data/transcripts/fm95blo_2011_11_09.json \\
        --features data/features/fm95blo_2011_11_09.json \\
        --labels data/labels/fm95blo_2011_11_09.csv
"""
import argparse
import bisect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import detect_breaks as db
import finalize_cuts as fc
import calibrate as cal


def no_speech_prob_series(transcript: dict, timestamps: list, window: float) -> list:
    """Rolling average of Whisper's per-segment no_speech_prob centered at
    each timestamp. Defaults to 1.0 (maximally "not speech") for windows
    with no transcribed segments nearby -- true silence/instrumental
    stretches should read as non-speech for lack of data, the opposite
    default from confidence_series's 0.0 (which read missing data as
    speech-neutral), since "no segment transcribed here at all" is itself
    evidence of non-speech for this particular signal."""
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


def build_variant_combined(transcript: dict, features: dict, variant: str) -> tuple:
    """Same shape as db.build_combined_timeline, but the word-rate
    contribution to the combined score is computed differently per variant:
      - "baseline":   (1 - rate_norm)                          [current pipeline]
      - "nsp_only":   nsp                                      [replace word-rate entirely]
      - "nsp_or":     max(1 - rate_norm, nsp)                  [either signal can flag non-speech]
      - "nsp_avg":    average of (1 - rate_norm) and nsp        [blend]
    """
    windows = features["windows"]
    timestamps = [w["t"] for w in windows]
    rates = db.word_rate_series(transcript, timestamps, config.WORD_RATE_WINDOW)
    rate_norm = [min(r / config.WORD_RATE_NORM_CAP, 1.0) for r in rates]
    nsp = no_speech_prob_series(transcript, timestamps, config.WORD_RATE_WINDOW)

    if variant == "baseline":
        non_speech = [1 - rn for rn in rate_norm]
    elif variant == "nsp_only":
        non_speech = nsp
    elif variant == "nsp_or":
        non_speech = [max(1 - rn, n) for rn, n in zip(rate_norm, nsp)]
    elif variant == "nsp_avg":
        non_speech = [((1 - rn) + n) / 2 for rn, n in zip(rate_norm, nsp)]
    else:
        raise ValueError(f"unknown variant {variant!r}")

    combined = [
        config.MUSIC_COMBINE_WEIGHT_ACOUSTIC * w["music_score"]
        + config.MUSIC_COMBINE_WEIGHT_WORDRATE * ns
        for w, ns in zip(windows, non_speech)
    ]
    return timestamps, rate_norm, combined


def compute_music_runs_variant(transcript: dict, features: dict, variant: str) -> list:
    timestamps, _, combined = build_variant_combined(transcript, features, variant)
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


def detect_variant(transcript: dict, features: dict, variant: str) -> list:
    """Mirrors detect_breaks.detect(), swapping in the variant's music runs."""
    occurrences = db.find_keyword_occurrences(transcript)
    silence_runs = db.merge_overlapping_runs(features["silence_runs"], "silence")
    music_runs = db.merge_overlapping_runs(
        compute_music_runs_variant(transcript, features, variant), "music"
    )
    non_speech_runs = db.union_runs(silence_runs, music_runs)

    supported_occurrences = set()
    scored = []
    for run in non_speech_runs:
        kws = db.supporting_keywords(run["start"], run["end"], occurrences, config.KEYWORD_SUPPORT_WINDOW)
        for kw in kws:
            supported_occurrences.add((kw["t"], kw["phrase"]))
        scored.append(db.score_run(run, occurrences, features["loudness_jumps"]))

    scored += db.keyword_only_candidates(occurrences, supported_occurrences)
    return db.merge_candidates(scored)


def postprocess_variant(candidates: list, transcript: dict, features: dict, variant: str) -> list:
    timestamps, rate_norm, combined = build_variant_combined(transcript, features, variant)
    merged = db.merge_nearby_candidates(candidates, timestamps, rate_norm)
    expanded = db.expand_boundaries(merged, timestamps, combined)
    return db.merge_nearby_candidates(expanded, timestamps, rate_norm)


def run_variant(transcript: dict, features: dict, variant: str) -> dict:
    raw = detect_variant(transcript, features, variant)
    review_candidates = postprocess_variant(raw, transcript, features, variant)
    result = fc.finalize(review_candidates, transcript, features)
    return {"review_candidates": review_candidates, "final_cuts": result["final_cuts"]}


def score(candidates: list, labels: list) -> dict:
    matched, missed, fps = cal.evaluate(candidates, labels)
    recall = len(matched) / len(labels) if labels else float("nan")
    precision = (len(candidates) - len(fps)) / len(candidates) if candidates else float("nan")
    return {"n": len(candidates), "recall": recall, "precision": precision,
            "matched": len(matched), "missed": len(missed), "fps": len(fps)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()

    transcript = json.loads(args.transcript.read_text(encoding="utf-8"))
    features = json.loads(args.features.read_text(encoding="utf-8"))
    labels = cal.load_labels(args.labels)

    print(f"{'variant':10s}  {'stage':11s}  {'n':>3s}  {'recall':>8s}  {'precision':>9s}  matched/missed/fp")
    for variant in ["baseline", "nsp_only", "nsp_or", "nsp_avg"]:
        result = run_variant(transcript, features, variant)
        for stage, key in [("review", "review_candidates"), ("final_cuts", "final_cuts")]:
            s = score(result[key], labels)
            print(
                f"{variant:10s}  {stage:11s}  {s['n']:3d}  {s['recall']:8.2%}  "
                f"{s['precision']:9.2%}  {s['matched']}/{s['missed']}/{s['fps']}"
            )


if __name__ == "__main__":
    main()
