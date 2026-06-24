# Icelandic Podcast Ad/Music-Break Remover

Semi-automatic pipeline: transcribe → detect candidate ad/music breaks
(with reasons) → review/edit → cut → export cleaned audio.

## Setup

```
pip install -r requirements.txt
```

ffmpeg must be installed and on PATH (already confirmed present on this
machine). The first transcription run downloads Whisper model weights.

## Pipeline

Run each stage manually so you can inspect intermediate output:

```
# 1. Transcribe (word-level timestamps)
python transcribe.py data/raw/episode1.mp3

# 2. Extract acoustic features (silence / music-likeness / loudness jumps)
python audio_features.py data/raw/episode1.mp3

# 3. Detect candidate break segments (merges transcript + acoustic signals,
#    then merges nearby candidates + expands boundaries outward -- this is
#    the generous "review candidates" stage, biased toward not missing
#    anything, written to <name>.json (merged) and <name>_raw.json (pre-merge)
python detect_breaks.py \
    --transcript data/transcripts/episode1.json \
    --features data/features/episode1.json

# 4. (First time / when you have ground truth) calibrate against your
#    manually-marked timestamps to tune config.py
python calibrate.py \
    --candidates data/candidates/episode1.json \
    --labels data/labels/episode1.csv

# 5. Refine into final cuts -- the precision pass: drops unsupported
#    keyword-only candidates, merges remaining candidates with a tiered gap
#    rule, and trims (never re-expands) the boundaries of low-confidence
#    candidates back toward their actual evidence. Writes
#    <name>_finalcuts.json, <name>_dropped.json (false positives removed),
#    <name>_uncertain.json (low-confidence cuts worth a closer look)
python finalize_cuts.py \
    --candidates data/candidates/episode1.json \
    --transcript data/transcripts/episode1.json \
    --features data/features/episode1.json

# 6. Export a reviewable report + Audacity label file (from the final cuts,
#    not the raw review candidates)
python review_export.py \
    --candidates data/candidates/episode1_finalcuts.json \
    --out-csv data/labels/episode1_review.csv \
    --out-audacity data/labels/episode1_audacity.txt

# 7. Review: edit the `decision` column in the CSV (remove/keep), or import
#    the Audacity label file, adjust boundaries on the waveform, delete any
#    false-positive labels, then re-export labels from Audacity.

# 8. Cut and export the cleaned episode
python export.py \
    --audio data/raw/episode1.mp3 \
    --cuts data/labels/episode1_review.csv \
    --out output/episode1_clean.mp3
```

### Why three detection stages

- **Stage 1+2** (`detect_breaks.py`): generous on purpose -- raw per-window
  signals, then merged across small gaps and expanded outward, so a real
  break never gets missed just because one window's score dipped.
- **Stage 3** (`finalize_cuts.py`): the precision pass on top of that. Stage
  2's generosity has a cost -- it occasionally expands too far into ambiguous
  audio, or treats a standalone keyword mention as a real candidate. Stage 3
  drops unsupported keyword-only hits (unless they're genuinely bridging two
  strong candidates), re-merges with a tiered gap rule (always bridge under
  45s; bridge up to 90s only between long blocks with no real host talk in
  the gap; extra leeway right at the very end of the file, since closing
  breaks usually run straight through to EOF), and trims -- never re-expands
  -- low-confidence boundaries back toward their actual evidence.
- Run stage 5 (`review_export.py`) against `*_finalcuts.json`, not the raw
  stage-2 output -- that's the list you should actually be reviewing/cutting
  from.

## Calibrating against your labeled example episode

1. Put the audio in `data/raw/`.
2. Create `data/labels/<episode>.csv` with your manual timestamps:
   ```
   start,end,label
   12:34,14:10,ad
   25:00,26:45,music
   ```
   (times accept `SS`, `MM:SS`, or `HH:MM:SS`)
3. Run steps 1-3 above, then `calibrate.py` against that label file.
4. Read the false-positive / false-negative list it prints. Adjust
   `config.py`:
   - False positives from silence/music alone → raise `SILENCE_MIN_DURATION`,
     `MUSIC_SCORE_THRESHOLD`, or `MUSIC_MIN_DURATION`.
   - Missed breaks → check the transcript around that timestamp for phrases
     the host used and add them to `BREAK_KEYWORDS`.
   - Re-run `detect_breaks.py` + `calibrate.py` until there are no false
     positives on the known episode (some missed recall is fine -- you'll
     catch those in the review step).
5. As you label more episodes, repeat step 4 to keep `config.py` accurate.

## Calibration results so far (fm95blo_2011_11_09, 2h05m)

- Hardware note: an 8GB M2 Mac can't run Whisper `large-v3`/`medium` on a
  multi-hour file in reasonable time (saw 60h+ and 20-30h ETAs) -- using
  `small` (~1.26x realtime) and chunked acoustic feature extraction
  (`FEATURE_CHUNK_SECONDS`) instead, both load-bearing for this to be
  practical at all on modest hardware.
- Acoustic music_score alone caught 0% of labeled music segments (some
  songs are percussive/guitar-heavy, not just "tonal"). Combining it with
  transcript word-rate (host talks less during music) plus hysteresis
  thresholding got every true segment at least partial candidate overlap.
- The detector's first real candidate set was badly fragmented: each long
  music/ad block was being split into many small slivers instead of one
  continuous cut, and boundaries stopped short of the true edges.
  `detect_breaks.py` now runs a `postprocess` pass (`merge_nearby_candidates`
  + `expand_boundaries`) after the initial per-window detection: it bridges
  candidates across gaps up to `MERGE_GAP_MAX` seconds when there's no real
  host speech in between, and grows each boundary outward until it hits a
  sustained run of clear speech. Current state on this episode: **19 final
  candidates / 66.67% recall (10 of 15 ground-truth breaks) / 3 false
  positives** at the review (stage 2) level, **13 final cuts / 73.33% recall
  (11 of 15) / 2 false positives** after the stage-3 precision pass. Both
  `data/candidates/<episode>.json` (final, merged+expanded -- what you
  should review/cut from) and `data/candidates/<episode>_raw.json`
  (pre-merge, for debugging the detector itself) are written by
  `detect_breaks.py`. (An earlier pass through this README claimed 93.33%
  recall / 3 false positives -- that number predates later config/threshold
  changes and is no longer reproducible from the current code + config; the
  numbers above are from re-running the current pipeline end to end.)
- 4 of the 5 remaining missed breaks involve the same root cause as the
  known hard case below: low acoustic music_score *and* a word-rate signal
  that reads as real speech (either sung lyrics Whisper transcribes as
  spoken words, or genuinely sparse-but-real host talk that still produces
  enough recognized words to clear the speech threshold) -- it defeats both
  halves of the combined signal at once.
  - **Tried and reverted:** added Whisper's per-segment `avg_logprob`
    (transcription confidence) as a third signal, on the theory that
    hallucinated "lyrics" would be lower-confidence than real speech. A
    91s-clip-level probe (`probe_confidence.py`,
    `data/confidence_probe/<episode>.json`) showed real separation (talk
    mean -0.91 vs music mean -0.46), but wiring it into the per-window
    (`AVG_LOGPROB_WINDOW`-sized) merge-gating logic and A/B testing against
    the labeled episode showed it added a false positive with **zero**
    recall improvement -- per-window avg_logprob is much noisier than the
    90s-clip averages that motivated it. Reverted; the code no longer
    computes or uses it. `no_speech_prob` (also captured per-segment in
    `transcribe.py`'s output, unused) may be worth trying instead, since
    it's a more direct VAD-style signal than confidence -- not yet tried.
  - Revisit once more labeled episodes show whether this is common enough
    to be worth a new signal at all.
- The 3 review-stage false positives are all heavily garbled Whisper output
  over real host conversation (the `small` model mistranscribes Icelandic
  badly in places), not actual breaks -- stray acoustic "music-like" blips
  on quiet talk. None of their confidence scores reach
  `AUTO_REMOVE_CONFIDENCE`, so none of them would actually get auto-cut --
  they show up as "review" rows, not "remove" rows, in the CSV. The
  stage-3 precision pass (`finalize_cuts.py`) drops one further false
  positive (an unsupported keyword-only "fm95" mid-conversation hit) by
  design, leaving 2 in the final cuts.
- Spoken ad reads are NOT acoustically distinct from normal talk on this
  show -- they rely on the (currently thin) `BREAK_KEYWORDS` list and stay
  low-confidence by design. Expect to manually extend ad boundaries in
  Audacity more often than music boundaries.
- `AUTO_REMOVE_CONFIDENCE` is set high (0.85) on purpose: the highest
  confidence score the detector produced (0.8, silence+music+loudness-jump
  together) turned out to be a quiet talk-show pause, not music. With only
  one labeled episode, no confidence level is proven safe to auto-remove --
  **every candidate currently defaults to "review", not "remove."** Treat
  this as a strong filter that tells you where to look, not a final cut
  list, until you've calibrated against a few more episodes.

## On new, unlabeled episodes

Run steps 1, 2, 3, 5 (skip calibrate -- no ground truth for a new episode),
review the CSV/Audacity labels, then run `export.py`. Always do the review
step before exporting on a new episode -- the detector is tuned to avoid
false positives on episodes it's seen, not guaranteed on unseen ones.
