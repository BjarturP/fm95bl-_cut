"""
Cross-episode repetition matcher — identify unknown (REVIEW-H) regions by
finding the SAME audio in other episodes.

Radio rotation: the same Icelandic hits, jingles and ads recur across
episodes. Shazam cannot name them (poor Icelandic coverage), but if an
unknown region's chromaprint fingerprint matches an unknown region in a
DIFFERENT episode, the audio is provably recorded/repeated content — a song
or jingle in rotation — and NOT live host talk. That is exactly the safety
evidence needed to consider cutting it without a name.

No external services, no reference files needed: the episodes themselves are
the database. (Complementary: drop known Icelandic songs into
experiments/song_refs/ and song_fingerprint.py will name them.)

Usage:
    .venv/bin/python experiments/cross_episode_match.py
    # optionally: --min-dur 60 --labels REVIEW-H --episodes ep1 ep2 ...

Output: pairwise match table + cluster summary, and
data/labels/cross_episode_matches.csv
"""
import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    import acoustid
    import chromaprint
except ImportError:
    sys.exit("pip install pyacoustid; brew install chromaprint")

EPISODES_DEFAULT = [
    "fm95blo_2011_11_09",
    "fm95blo-2012-01-30",
    "fm95blo-2012-02-28",
    "fm95blo-2012-03-09",
    "fm95blo-2012-03-02",
]

# Matching thresholds — bit_thresh/frac follow the values validated in the
# song_fingerprint.py investigation; frac is raised because same-recording
# radio plays should match far better than the 0.20 name-lookup floor.
BIT_THRESH   = 10     # max differing bits (of 32) for a frame to count as matching
MATCH_FRAC   = 0.35   # fraction of query frames matching at the best offset
CORE_LEN     = 60.0   # query slice from the middle of each region
EDGE_TRIM    = 12.0   # skip region edges (talk-over / boundary slop)

_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def popcount32(arr: np.ndarray) -> np.ndarray:
    return _POP8[arr.view(np.uint8)].reshape(arr.shape[0], 4).sum(axis=1)


def best_match(ref: np.ndarray, query: np.ndarray) -> tuple[int | None, float]:
    """Slide query over ref; return (best_offset_frames, best_match_fraction)."""
    n, m = len(ref), len(query)
    if m > n:
        ref, query, n, m = query, ref, m, n
    if m == 0 or n < m:
        return None, 0.0
    best_frac, best_off = -1.0, None
    for off in range(0, n - m + 1):
        xor = np.bitwise_xor(ref[off:off + m], query)
        frac = float(np.mean(popcount32(xor) <= BIT_THRESH))
        if frac > best_frac:
            best_frac, best_off = frac, off
    return best_off, best_frac


def fingerprint_span(audio: Path, start: float, end: float) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-ss", str(max(0.0, start)), "-to", str(end),
             "-i", str(audio), "-ar", "22050", "-ac", "1", str(tmp_path)],
            check=True,
        )
        _dur, fp = acoustid.fingerprint_file(str(tmp_path), maxlength=int(end - start) + 1)
        ints, _algo = chromaprint.decode_fingerprint(fp)
        return np.array(ints, dtype=np.uint32)
    finally:
        tmp_path.unlink(missing_ok=True)


def episode_audio(ep: str) -> Path:
    for cand in (Path(f"episodes/{ep}-mp3.mp3"), Path(f"episodes/{ep}.mp3")):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"no audio for {ep}")


def collect_regions(ep: str, labels: set[str], min_dur: float) -> list[dict]:
    out = []
    path = Path(f"data/labels/hybrid_review_{ep}.csv")
    for r in csv.DictReader(path.open(encoding="utf-8")):
        if r["label"] not in labels:
            continue
        if (r.get("song") or "").strip():
            continue  # already named
        s, e = float(r["start"]), float(r["end"])
        if e - s < min_dur:
            continue
        out.append({"ep": ep, "start": s, "end": e, "label": r["label"]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="*", default=EPISODES_DEFAULT)
    ap.add_argument("--labels", nargs="*", default=["REVIEW-H"])
    ap.add_argument("--min-dur", type=float, default=60.0)
    ap.add_argument("--out", type=Path, default=Path("data/labels/cross_episode_matches.csv"))
    args = ap.parse_args()

    regions: list[dict] = []
    for ep in args.episodes:
        regs = collect_regions(ep, set(args.labels), args.min_dur)
        print(f"{ep}: {len(regs)} unknown region(s)")
        regions.extend(regs)

    print(f"\nFingerprinting {len(regions)} regions ...", file=sys.stderr)
    for i, r in enumerate(regions):
        audio = episode_audio(r["ep"])
        r["fp_full"] = fingerprint_span(audio, r["start"], r["end"])
        # queries: CORE_LEN slices tiling the region (edges trimmed), so a
        # repeat anywhere inside a long block can be found, not just its middle
        s, e = r["start"] + EDGE_TRIM, r["end"] - EDGE_TRIM
        r["fp_slices"] = []
        if e - s < 15:
            r["fp_slices"] = [r["fp_full"]]
        else:
            t = s
            while t < e:
                t2 = min(t + CORE_LEN, e)
                if t2 - t >= 15:
                    r["fp_slices"].append(fingerprint_span(audio, t, t2))
                t += CORE_LEN
        print(f"  [{i+1}/{len(regions)}] {r['ep']} {r['start']:.0f}-{r['end']:.0f}s "
              f"({len(r['fp_full'])} frames, {len(r['fp_slices'])} slices)", file=sys.stderr)

    print("\nPairwise matching ...", file=sys.stderr)
    matches = []
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            a, b = regions[i], regions[j]
            frac = 0.0
            for q in b["fp_slices"]:
                _off, f = best_match(a["fp_full"], q)
                frac = max(frac, f)
            for q in a["fp_slices"]:
                _off, f = best_match(b["fp_full"], q)
                frac = max(frac, f)
            if frac >= MATCH_FRAC:
                matches.append((frac, a, b))

    matches.sort(reverse=True, key=lambda t: t[0])

    def hms(x):
        h, rem = divmod(x, 3600); m, s = divmod(rem, 60)
        return f"{int(h):01d}:{int(m):02d}:{int(s):02d}"

    print(f"\n{'='*76}")
    print(f"  Cross-episode repeats found: {len(matches)} "
          f"(threshold: {MATCH_FRAC:.0%} of frames within {BIT_THRESH} bits)")
    print(f"{'='*76}")
    for frac, a, b in matches:
        same = "SAME-EP" if a["ep"] == b["ep"] else "cross"
        print(f"  {frac:5.0%}  {a['ep']:<22} {hms(a['start'])}-{hms(a['end'])}"
              f"  ~  {b['ep']:<22} {hms(b['start'])}-{hms(b['end'])}  [{same}]")

    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["match_frac", "ep_a", "start_a", "end_a", "ep_b", "start_b", "end_b", "same_episode"])
        for frac, a, b in matches:
            w.writerow([f"{frac:.3f}", a["ep"], a["start"], a["end"],
                        b["ep"], b["start"], b["end"], a["ep"] == b["ep"]])
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
