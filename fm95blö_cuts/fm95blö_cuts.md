# fm95blö_cuts

> **Instructions for Claude:** At the start of every session, read this file first, then `README.md` and recent git log. Summarize current state back to the user before writing any code. At the end of every session, update this file.

---

## Project Goal

Automatically detect and cut **music breaks** from Icelandic FM95Blö podcast/radio episodes. Ads are out of scope — only music segments are targeted. The pipeline transcribes audio with Whisper, detects music candidates using acoustic features + transcript signals, refines them through a precision pass, and exports Audacity label files and review CSVs for human approval before cutting.

**Repo:** `/Users/bjarturpall/Documents/BjarturPall/Haskoli/drasl/Projects/claudetest2`

---

## Current Repo State

| Field | Value |
|---|---|
| Branch | `main` |
| Latest commit | `c267aab` — "Add Shazam gap-scanner and integrate into combined review package" (2026-06-27) |
| Working tree | Untracked files only (clean for committed code) |
| Ahead of remote | 3 commits (not yet pushed) |

**Untracked (not committed):**
- `data/candidates/fm95blo-2012-02-28*.json` (new episode, pipeline just run)
- `data/features/fm95blo-2012-02-28.json`
- `data/transcripts/fm95blo-2012-02-28.json`
- `data/labels/gap_shazam_*` (gap scan outputs)
- `data/labels/review_sheet_*`, `review_audacity_*` (review packages)
- `data/labels/song_matches_*` (normal Shazam outputs)
- `experiments/song_fingerprint.py`
- `fm95blö_cuts/` (this vault)

### Important Scripts

| Script | Role |
|---|---|
| `detect_breaks.py` | Stage 1–2: transcribe signals + acoustic features → raw candidates → merge/expand |
| `finalize_cuts.py` | Stage 3: precision pass — drop unsupported candidates, tiered gap merge, trim boundaries |
| `calibrate.py` | Evaluate candidates against ground truth labels, tune `config.py` |
| `review_export.py` | Export `*_finalcuts.json` → reviewable CSV + Audacity TXT |
| `export.py` | Cut audio using approved CSV labels |
| `audio_features.py` | Extract silence/music-likeness/loudness features |
| `transcribe.py` | Whisper transcription (word-level timestamps) |
| `probe_confidence.py` | One-off probe for Whisper confidence signals (diagnostic only) |
| `experiments/no_speech_prob.py` | A/B harness that validated the `no_speech_prob` improvement |
| `experiments/song_fingerprint.py` | Prototype: fingerprint-based song detection (NOT in main pipeline) |

### Current Stable Pipeline (run in order)

```bash
# Stage 1–3: heuristic detection (unchanged)
python transcribe.py data/raw/<ep>.mp3
python audio_features.py data/raw/<ep>.mp3
python detect_breaks.py --transcript data/transcripts/<ep>.json --features data/features/<ep>.json
python finalize_cuts.py --candidates data/candidates/<ep>.json --transcript data/transcripts/<ep>.json --features data/features/<ep>.json
python review_export.py --candidates data/candidates/<ep>_finalcuts.json --out-csv data/labels/<ep>_review.csv --out-audacity data/labels/<ep>_audacity.txt

# Stage 4: Shazam + hybrid classification (promoted from experiments)
.venv/bin/python experiments/shazam_detect.py \
    --audio episodes/<ep>.mp3 --candidates data/candidates/<ep>_finalcuts.json \
    --features data/features/<ep>.json --transcript data/transcripts/<ep>.json \
    --episode-name <ep>
python experiments/shazam_gap_scan.py \
    --audio episodes/<ep>.mp3 --candidates data/candidates/<ep>_finalcuts.json \
    --episode-name <ep>
python experiments/make_review.py \
    --heuristic-csv data/labels/<ep>_review.csv \
    --shazam-json data/labels/song_matches_<ep>.json \
    --shazam-unmatched data/labels/song_unmatched_<ep>.csv \
    --gap-shazam-json data/labels/gap_shazam_matches_<ep>.json \
    --episode-name <ep>
# → data/labels/hybrid_auto_cuts_<ep>.csv      ← cut these without review
# → data/labels/hybrid_manual_review_<ep>.csv  ← review these in Audacity
# → data/labels/hybrid_review_audacity_<ep>.txt ← import into Audacity
```

---

## Current Best Baseline

Measured on: **`fm95blo_2011_11_09`** (2h05m episode, 15 labeled breaks)

| Stage | Recall | Precision | Notes |
|---|---|---|---|
| final-cuts (prior baseline) | 73.33% (11/15) | 84.62% (2 FP) | word-rate only |
| **final-cuts (current)** | **80.00% (12/15)** | **90.91% (1 FP)** | word-rate OR no_speech_prob |
| review stage (current) | 86.67% (13/15) | 72.22% (5 FP) | pre-precision-pass |

**What caused the improvement:** Using `max(1 - rate_norm, no_speech_prob)` instead of just `1 - rate_norm` as the non-speech signal in `build_combined_timeline`. Either signal flagging non-speech is sufficient — `no_speech_prob` stays high through sung/garbled stretches that Whisper transcribes as if they were real speech (defeating word-rate).

**Remaining missed breaks (3):**
- `3376–3600s` — "music", 0–35% coverage
- `6188–6381s` — "music", 0–35% coverage
- `5010–5252s` — "ads", keyword-only, no acoustic support

**Remaining false positive (1):**
- `5976–6033s` — conf=0.40, quiet talk misread as music-like; below `AUTO_REMOVE_CONFIDENCE` so shows as review row, not auto-removed

---

## Pipeline Overview

1. **Transcribe** (`transcribe.py`) — Whisper `small` model, word-level timestamps. Using `small` because `medium`/`large-v3` are impractically slow on 8GB M2 Mac (20–60h ETAs on multi-hour files).
2. **Extract features** (`audio_features.py`) — silence, music-likeness (tonal/spectral), loudness jumps. Chunked (`FEATURE_CHUNK_SECONDS`) for memory.
3. **Detect candidates** (`detect_breaks.py`) — per-window scoring combining acoustic + transcript signals, then `merge_nearby_candidates` + `expand_boundaries` postprocess. Writes `*_raw.json` (pre-merge) and `*.json` (merged+expanded). **Biased toward recall** — catches everything; next stage handles precision.
4. **Finalize cuts** (`finalize_cuts.py`) — precision pass: drops unsupported keyword-only candidates, tiered gap merge (always bridge <45s; bridge up to 90s between long blocks with no host talk; extra leeway at EOF), trims (never re-expands) low-confidence boundaries. Writes `*_finalcuts.json`, `*_dropped.json`, `*_uncertain.json`.
5. **Calibrate/evaluate** (`calibrate.py`) — compares candidates against ground truth labels, prints FP/FN list to guide `config.py` tuning.
6. **Review export** (`review_export.py`) — converts `*_finalcuts.json` to human-reviewable CSV + Audacity TXT. **Always run against `*_finalcuts.json`**, not raw candidates.

---

## Ground Truth Labels

### `fm95blo_2011_11_09` (2h05m) — primary calibration episode

| Start | End | Label | Notes |
|---|---|---|---|
| 0:00 | 6:20 | music/ads | Opening block |
| 10:00 | 13:45 | music | |
| 15:35 | 19:15 | music | |
| 25:25 | 28:48 | music | |
| 32:55 | 36:58 | music | |
| 41:28 | 44:56 | ads | Keyword-only, no acoustic support |
| 44:56 | 48:38 | music | |
| 56:16 | 1:00:00 | music | |
| 1:05:35 | 1:12:37 | music | |
| 1:15:43 | 1:19:33 | music | |
| 1:23:30 | 1:27:32 | ads | |
| 1:27:32 | 1:29:49 | music | |
| 1:33:25 | 1:36:43 | music | |
| 1:43:08 | 1:46:21 | music | |
| 1:50:13 | 2:05:02 | music/ads | Closing block |

**Difficult areas:** 3376–3600s and 6188–6381s (low music-score, partially percussive/guitar-heavy). 5010–5252s ads block is keyword-only — no acoustic signal.

### `fm95blo-2012-03-09` — second episode (untracked, not yet labeled)

Data files present in `data/` but no ground truth labels yet. Do not evaluate pipeline on this episode until labels exist.

---

## Experiments Tried

| # | Signal | What was tested | Result | Status |
|---|---|---|---|---|
| 1 | `avg_logprob` | Whisper transcription confidence as third signal — theory: hallucinated lyrics score lower than real speech | 90s-clip probe showed real separation (talk −0.91 vs music −0.46) but per-window signal too noisy; added a FP with zero recall gain | **Rejected & reverted** |
| 2 | `no_speech_prob` | Whisper's per-segment VAD-style non-speech estimate as OR-gate with word-rate | A/B in `experiments/no_speech_prob.py` against labeled episode: final-cuts +6.67pp recall, +6.29pp precision simultaneously | **Promoted into main pipeline** |
| 3 | AcoustID fingerprinting | `experiments/song_fingerprint.py` — chromaprint + Hamming-distance + AcoustID API | **Weak: 1/18 matched on 2011 episode.** Poor Icelandic music coverage in MusicBrainz/AcoustID database. | **Superseded by Shazam** |
| 4 | Shazam recognition | `experiments/shazam_detect.py` — shazamio (no API key), multi-source duration lookup (iTunes → MusicBrainz → Spotify) | **9/18 on 2011, 10/21 on 2012.** Identified 1 Icelandic song (Magnús Eiríksson & Icy - Gleðibankinn). All matched songs now have duration. Boundary sanity checks flag uncertain cases. | **In progress (separate)** |
| 5 | Shazam gap scan | `experiments/shazam_gap_scan.py` — scans large gaps between heuristic finalcuts candidates with Shazam at 45s intervals (15s clips). Targets songs the heuristic never produced a candidate for. | **GT08 FOUND.** Pitbull matched on 5/14 windows (including rap section), start error 4s. Also confirmed Fun.+Lil Wayne in 2823–3762s gap. 3 "tail" matches show heuristic candidate ends cut off 85–120s early. Zero FPs in silent gaps. | **In experiments, validated** |

### Shazam experiment detail (2026-06-25)

**2011 episode (`fm95blo_2011_11_09`) — 9/18 matched, all with duration:**

| Region | Song | Duration src | Cut start | Cut end | Note |
|---|---|---|---|---|---|
| 112–378s | Gotye - Somebody That I Used to Know | musicbrainz | 00:01:16 | 00:04:24 | ok |
| 1521–1855s | David Guetta - Without You (feat. Usher) | musicbrainz | 00:24:39 | 00:29:20 | ok |
| 1976–2244s | Coldplay - Paradise | musicbrainz | 00:31:55 | 00:36:22 | ok |
| 2530–2924s | Lana Del Rey - Video Games | musicbrainz | 00:44:04 | 00:48:49 | ok |
| 3522–3600s | Florence + the Machine - Shake It Out | itunes | 00:58:42 | 01:00:45 | UNCERTAIN (song start 160s before region) |
| 3888–4328s | JAY-Z & Kanye West - Why I Love You | musicbrainz | 01:04:52 | 01:08:56 | ok |
| 4480–4921s | Eddie Vedder - Society | musicbrainz | 01:15:00 | 01:20:46 | ok |
| 5201–5385s | Calvin Harris - Feel So Close (Radio Edit) | musicbrainz | 01:26:04 | 01:29:40 | ok |
| 5605–5917s | Delilah - Go | musicbrainz | 01:32:40 | 01:36:26 | ok |

**2012 episode (`fm95blo-2012-03-09`) — 10/21 matched, all with duration:**

| Region | Song | Duration src | Cut start | Cut end | Note |
|---|---|---|---|---|---|
| 210–397s | Flo Rida - Wild Ones (feat. Sia) | musicbrainz | 00:03:07 | 00:06:46 | ok |
| 787–917s | Eminem - No Love (feat. Lil Wayne) | musicbrainz | 00:13:27 | 00:15:22 | UNCERTAIN (dur 280s >> region 130s) |
| 1570–1845s | AWOLNATION - Sail | musicbrainz | 00:25:30 | 00:29:46 | ok |
| 2881–3186s | Fun. - We Are Young (feat. Janelle Monáe) | musicbrainz | 00:47:41 | 00:53:11 | UNCERTAIN (dur 632s >> region 305s) |
| 3625–3732s | Lil Wayne - Mirror (feat. Bruno Mars) | musicbrainz | 01:00:25 | 01:02:45 | UNCERTAIN (song start 186s before region) |
| 4152–4384s | Coldplay & Rihanna - Princess of China | musicbrainz | 01:08:35 | 01:12:43 | ok |
| 5642–5952s | Iggy Azalea - Pu$$Y | musicbrainz | 01:35:45 | 01:38:02 | ok |
| 6094–6263s | JAY-Z & Kanye West - Ni**as in Paris | musicbrainz | 01:42:20 | 01:44:28 | UNCERTAIN (dur 219s >> region 169s) |
| 6589–6746s | Magnús Eiríksson & Icy - Gleðibankinn | itunes | 01:50:05 | 01:53:15 | ok ✓ Icelandic! |
| 6820–6891s | Florence + the Machine - Cosmic Love | musicbrainz | 01:53:06 | 01:54:56 | UNCERTAIN (dur 336s >> region 71s) |

**Conclusion:** Shazam recognition is promising. Main remaining issues:
- ~50% of regions unmatched (Icelandic music not in database, ad regions)
- Several "UNCERTAIN" flags where MusicBrainz returns album duration but radio used a shorter edit — need to validate against actual audio or use heuristic region end as cap

---

## Known Problems / Remaining Gaps

- **Ads are out of scope** — only music segments are targeted; keyword-based ad detection is irrelevant.
- **Some music breaks still missed** — percussive/guitar-heavy songs score poorly on tonal music-likeness; word-rate + no_speech_prob helps but doesn't fully solve it.
- **Rap/spoken-word music structurally hard** — speech-like delivery defeats both word-rate and no_speech_prob simultaneously (see GT08 diagnostic).
- **Quiet talk → false positive** — low-energy host speech can read as "music-like"; current FP at 5976–6033s is this case. `AUTO_REMOVE_CONFIDENCE=0.85` keeps it in review rather than auto-removing.
- **Generalization untested** — all tuning is against one labeled episode (`fm95blo_2011_11_09`). Pipeline may behave differently on other episodes. Need more ground truth labels before raising confidence in `config.py` thresholds.
- **`AUTO_REMOVE_CONFIDENCE` not yet safe to use** — highest observed confidence (0.8) was a quiet talk pause, not music. Every candidate currently defaults to review.

---

## Current Direction

### Hybrid review workflow (promoted — now standard)
`experiments/make_review.py` automatically calls `experiments/hybrid_review.py` after building the review sheet. The hybrid outputs are the recommended review files:

| Output | Use |
|---|---|
| `hybrid_auto_cuts_<ep>.csv` | Cut these automatically — 100% precision on 2 episodes |
| `hybrid_manual_review_<ep>.csv` | REVIEW-S / REVIEW-B / REVIEW-H / EXT rows |
| `hybrid_review_audacity_<ep>.txt` | Import into Audacity for visual review |

**Classification categories:**
- `[AUTO]` — ≥2 Shazam hits, MEDIUM/HIGH confidence, ≥60s, start within 30s of heuristic
- `[REVIEW-B]` — would be AUTO but start is >30s before heuristic start (song may play under host talk)
- `[REVIEW-S]` — Shazam matched, boundary UNCERTAIN (end scan miss, n=1, etc.)
- `[REVIEW-H]` — heuristic only, no Shazam match (Icelandic, ads, jingles)
- `[EXT]` — gap-scan tail extension suggestion
- `[DROP?]` — short/weak, likely not music

**Cross-episode AUTO precision: 100% (11/11 auto-cuts are real GT music, 0 FP across 2 episodes)**

### Stable path: improve detection incrementally
Continue refining `detect_breaks.py` + `finalize_cuts.py` signals. Do not change `config.py` thresholds until 3+ labeled episodes confirm stability.

---

## Next Recommended Step

1. **Label `fm95blo-2012-03-09`** — third episode with data but no ground truth. Cross-episode AUTO precision (100% on 2 episodes) is promising but should hold on a third before trusting it in production.
2. **Calvin Harris boundary edge case** — the 51s-before-GT start for Calvin Harris slipped through the 30s safety rule (because the heuristic itself was 34s early). Consider whether `EARLY_START_THRESHOLD` should drop to 15s, but wait for a third episode to inform this decision.
3. **Icelandic music coverage** — only 1 Icelandic song (Magnús Eiríksson) found by Shazam across all episodes. Remaining unmatched regions (GT2, GT3 in 2011; GT8 in 2012) are likely Icelandic. No fix without a local database.
4. **Tail boundary improvement** — gap scan found Flo Rida/Eminem/JAY-Z extending 85–120s past heuristic candidate ends. A short boundary-extension Shazam pass (probe 2 min after each finalcut end) could fix this without changing detection thresholds.

---

## Session Log

### 2026-06-25 — Session 1: memory setup + Shazam experiment

**What changed:**
- Created this Obsidian memory note
- Tested AcoustID API: only 1/18 matched (poor Icelandic coverage) — superseded
- Built `experiments/shazam_detect.py` using shazamio + multi-source duration (iTunes → MusicBrainz → Spotify fallback)
- Ran Shazam on both episodes: 9/18 (2011), 10/21 (2012) — all matched songs now have duration
- Shazam matched 1 Icelandic song: Magnús Eiríksson & Icy - Gleðibankinn (2012 episode)

**Files changed:**
- `fm95blö_cuts/fm95blö_cuts.md` (created + updated)
- `experiments/shazam_detect.py` (new)
- `data/labels/song_matches_fm95blo-2011-11-09.{csv,json}` (generated)
- `data/labels/song_cut_suggestions_fm95blo-2011-11-09_audacity.txt` (generated)
- `data/labels/song_unmatched_fm95blo-2011-11-09.csv` (generated)
- `data/labels/song_matches_fm95blo-2012-03-09.{csv,json}` (generated)
- `data/labels/song_cut_suggestions_fm95blo-2012-03-09_audacity.txt` (generated)
- `data/labels/song_unmatched_fm95blo-2012-03-09.csv` (generated)

**Results summary:**
- 2011: 9/18 matched, 8 ok cuts + 1 uncertain (Florence offset implausible)
- 2012: 10/21 matched, 5 ok cuts + 5 uncertain (duration >> region → likely album vs. radio edit)
- Main open question: how accurate are the "ok" Shazam cuts vs. ground truth? Need 2012 labels to evaluate.

**Commits made:** None (all untracked)

**What failed:** AcoustID invalid API key attempts (first two keys were wrong). MusicBrainz ISRC lookup via `isrc=` endpoint returns 400; switched to artist+title search which works.

**Next:** Evaluate 2012 ground truth against heuristic + Shazam. ✓ Done (session 3).

---

### 2026-06-25 — Session 2: Shazam experiment + review workflow

**What changed:**
- Replaced AcoustID with `shazamio` in `experiments/shazam_detect.py`
- Added multi-source duration: iTunes → MusicBrainz (artist+title) → Spotify
- Added boundary sanity checks with UNCERTAIN flag (overshoot >60s)
- Built `experiments/make_review.py` — unified review sheet + combined Audacity label file
- Ran Shazam on both episodes; ran make_review on 2012

**Shazam results vs AcoustID:**
- AcoustID: 1/18 on 2011 (poor Icelandic/small-database coverage)
- Shazam 2011: 9/18 matched, all 9 with duration
- Shazam 2012: 10/21 matched, all 10 with duration
- 1 Icelandic song matched: Magnús Eiríksson & Icy - Gleðibankinn

**Review files generated for 2012:**
- `data/labels/review_sheet_fm95blo-2012-03-09.csv` — 15 rows, fill in actual_start/end/label/notes
- `data/labels/review_audacity_fm95blo-2012-03-09.txt` — [H] heuristic + [S ✓/⚠] Shazam labels for Audacity

**Files added:** `experiments/shazam_detect.py` (rewritten), `experiments/make_review.py` (new), all `song_matches_*`, `review_sheet_*`, `review_audacity_*` in `data/labels/`

**Commits made:** None

**Next:** User labels 2012 review sheet → run evaluation (precision/recall/boundary error, Shazam ok vs uncertain accuracy).

---

### 2026-06-26 — Session 3: 2012 evaluation + GT08 diagnostic

**What changed:** Ground truth labeling for 2012 episode received. Full evaluation run. GT08 diagnosed.

**2012 evaluation results (10 ground truth breaks):**

| Detector | Predictions | Recall | Precision | Missed |
|---|---|---|---|---|
| Heuristic only | 12 | 70.0% | 58.3% | 3 (GT05, GT06, GT08) |
| Shazam matched only | 10 | 60.0% | 60.0% | 4 (GT04, GT08, GT09, GT10) |
| **Combined** | **13** | **90.0%** | **69.2%** | **1 (GT08 only)** |

- Shazam rescued GT05 (Fun. - We Are Young) and GT06 (Lil Wayne - Mirror) that heuristic missed completely
- 4 FPs in both detectors are late-episode music (after 01:32:15) — real music but not labeled breaks
- Shazam "ok" vs "UNCERTAIN" classification: NOT predictive — both have exactly 60% precision
- Boundary accuracy: Shazam slightly better on average (mean start err 62s vs 69s heuristic)
- Best individual accuracy: AWOLNATION end boundary 4s off (Shazam offset method working well)

**GT08 diagnostic — "Give Me Everything Tonight" by Pitbull:**

- **Verdict: Detection failure (acoustic + transcript signal failure simultaneously)**
- Never appeared in any candidate file — not in raw.json, not in dropped.json
- Peak combined score: **0.5982** — missed MUSIC_ENTER_THRESHOLD=0.60 by just **0.0018**
- **Why it failed:**
  1. **First half (4612–4666s)**: Whisper hallucinates pseudo-Icelandic from the sung melody (nsp=0.846 → correctly flagged as non-speech). Music_score is weak (0.18–0.20) — no sustained harmonics. Combined peaks at 0.598, never crosses 0.60.
  2. **Second half (4700–4845s)**: Pitbull raps with speech-like delivery. Whisper confidently transcribes full English lyrics (nsp drops to 0.07–0.28). Rate_norm spikes to 0.9–1.0. Combined collapses to 0.17–0.35. Both non-speech signals are simultaneously defeated.
- Lowering threshold to 0.595 would start a partial run (~4612–4700s = 88s) but exit immediately when rap begins. Full break is 233s — would miss ~60%.
- Not a finalize/drop failure. Not a Shazam coverage failure. Not labeling ambiguity.

**Why the H5_FP (2756–2823s) passed when GT08 didn't:**
- H5 mean combined = 0.504 (passed threshold briefly, got through)
- GT08 mean combined = 0.445 (never passed threshold, rap half drags average down)

**Root cause category: rap/spoken-word music** — melodic intro below threshold, then speech-like rap defeats both word-rate and no_speech_prob. This is a structural gap in the current signal set. The fix requires either:
- **Gap-scanning with Shazam** (scan the 621s gap between H7 end at 4384s and H8 start at 5005s directly, without needing a heuristic candidate) — targeted, safe, keeps in experiments
- **Beat/rhythm detector** as a new signal — more robust to rap, but significant work and FP risk
- **Accept as known limitation** for rap-heavy songs until more labeled episodes show the pattern

**Commits made:** None

**Next:** Decide GT08 fix approach. Options: gap-scan Shazam, beat detector, or accept.

---

---

### 2026-06-27 — Session 4: Shazam gap scanner + GT08 found

**What changed:**
- Built `experiments/shazam_gap_scan.py` — scans large gaps between finalcut candidates with Shazam at 45s intervals (15s clips). Imports core logic (shazam_clip, lookup_duration, _hms) from shazam_detect.py.
- Clarified scope: music-only, ads are out of scope. Updated Obsidian note and Claude memory accordingly.
- Ran gap scan on 2012 episode: 87 queries across 9 gaps, ~15 min runtime.

**Key result: GT08 FOUND**
- Pitbull - Give Me Everything (feat. Ne-Yo, Afrojack & Nayer)
- 5/14 windows matched (t=4619, 4664, 4709, 4754, 4799) — including the RAP section
- Shazam audio fingerprinting is robust to rap/spoken-word where VAD signals (word-rate, no_speech_prob) fail completely
- song_start estimate: 4608s (GT actual 4612s, **4s error** — excellent)
- song_end estimate: 4861s (GT actual 4845s, 16s overshoot — album vs. radio edit)
- Not flagged UNCERTAIN

**Other gap-scan results on 2012:**

| Gap | Song | Category | Windows |
|---|---|---|---|
| 397–647s | Flo Rida - Wild Ones | tail of candidate 210–397s | 1 |
| 917–1570s | Eminem - No Love | tail of candidate 647–917s | 5 |
| 2399–2756s | Magnús Eiríksson & Icy | near candidate 2756–2823s, UNCERTAIN | 1 |
| 2823–3762s | Fun. - We Are Young | new find (GT05, already in normal Shazam) | 5 |
| 2823–3762s | Lil Wayne - Mirror | new find (GT06, already in normal Shazam) | 4 |
| 4384–5005s | **Pitbull** | **GT08 — new find!** | 5 |
| 6263–6589s | JAY-Z & Kanye - Ni**as in Paris | tail of candidate 6025–6263s | 3 |

**"Tail" pattern:** Flo Rida, Eminem, JAY-Z appear in gaps because the heuristic candidate ends cut off the song 85–120s early. Shazam still matches at the start of the gap because the song is still playing. Not new breaks — but suggests a future boundary-extension pass.

**FP risk:**
- Gaps 1845–2108s and 5270–5642s: zero matches (host talk, correct)
- No new false positives introduced by gap scan

**Files added:**
- `experiments/shazam_gap_scan.py` (new)
- `data/labels/gap_shazam_matches_fm95blo-2012-03-09.{csv,json}` (generated)
- `data/labels/gap_shazam_audacity_fm95blo-2012-03-09.txt` (generated)

**Commits made:** None (all untracked)

**Next:** Run gap scan on 2011 episode to test generalization. The two missed breaks there (3376–3600s, 6188–6381s) are the validation target.

---

### 2026-06-27 — Session 5: gap scan integrated into combined review package

**What changed:**
- Ran gap scan on 2011 episode (52 queries, 7 gaps). Both remaining missed breaks found.
- Updated `experiments/make_review.py` to accept `--gap-shazam-json` and merge all three sources (heuristic + normal Shazam + gap scan) into one review CSV and Audacity label file.
- Updated `README.md` with new workflow steps (6b–6d) and Shazam documentation.

**2011 gap scan results:**

| Gap | Song | Status | Start error | End error |
|---|---|---|---|---|
| 3070–3522s | Florence + the Machine - Shake It Out | UNCERTAIN (duration overshoots) | **3s** | album vs radio edit |
| 6033–6592s | **Daughtry - Crawling Back to You** | **ok** | **19s** | **6s** |
| 7291–7520s | Labrinth + Backstreet Boys | UNCERTAIN | — | — |

- **Both 2011 missed breaks found.** Zero FPs in 4 silent gaps (1166–1521s, 2244–2530s, 4921–5218s, 5385–5605s).

**Combined review validation (both episodes):**

| Episode | Heuristic | Normal Shazam | Gap-Shazam | Review rows | Key missed breaks now visible |
|---|---|---|---|---|---|
| 2011 | 15 | 9 | 4 | 16 | Florence (row 8), Daughtry (row 14) ✓ |
| 2012 | 12 | 10 | 7 | 16 | Pitbull GT08 (row 10) ✓ |

**Region types in combined review:**
- `heuristic+shazam` — most music (both agree)
- `heuristic_only+gap_shazam` — heuristic found something, gap scan identifies the song
- `gap_shazam_only` — **gap scan only; pure new find** (review carefully)
- `shazam_only+gap_shazam` — Shazam + gap scan, heuristic missed entirely

**Files changed:**
- `experiments/make_review.py` — rewritten with gap-shazam integration
- `README.md` — updated pipeline steps and Shazam section
- `fm95blö_cuts/fm95blö_cuts.md` — updated (this file)
- `data/labels/review_sheet_fm95blo_2011_11_09.csv` (regenerated)
- `data/labels/review_audacity_fm95blo_2011_11_09.txt` (regenerated)
- `data/labels/review_sheet_fm95blo-2012-03-09.csv` (regenerated)
- `data/labels/review_audacity_fm95blo-2012-03-09.txt` (regenerated)
- `data/labels/gap_shazam_matches_fm95blo_2011_11_09.{csv,json,txt}` (new)

**Commits made:** See git log.

**Next:** Label more episodes. The review package is now the authoritative workflow — run steps 6b–6d on any new episode after finalcuts.

---

### 2026-06-28 — Session 7: ground truth evaluation + hybrid review workflow

**Episode:** `fm95blo-2012-02-28` (first full GT evaluation on this episode)

**Ground truth file:** `data/labels/2012-02-08_actual_cuts.txt` (11 music cuts)

#### Experiment improvements (all in `experiments/`, pipeline unchanged)

**A. Multi-sample Shazam boundary estimation** (`experiments/shazam_detect.py` — rewritten)
- Sweeps 2–6 clips per region; clusters estimated song_start values; median if agree, UNCERTAIN if spread >30s
- End scan: for n≥2 hits, probes past expected song end to find empirical end
- **New (this session):** for n=1, forward sweep from last observed sample instead of trusting DB duration

**B. Same-song merge** (`experiments/make_review.py`)
- Merges adjacent Shazam rows with same song and gap ≤120s
- Fixed Drake (+100s start error → -14s after merge)

**C. Safer end rule** (`experiments/shazam_detect.py`)
- n=1 end scan now sweeps forward from last observed sample (not from DB duration)
- Fixed The Fray (+67s end error → +1s after scan)

**D. Hybrid review workflow** (`experiments/hybrid_review.py` — new)
- Takes `review_sheet_<ep>.csv` (output of `make_review.py`) and classifies each row

#### Measured accuracy on `fm95blo-2012-02-28` (11 GT cuts)

| Metric | Before improvements | After improvements |
|---|---|---|
| Matched GT cuts | 10/11 | 10/11 |
| Mean \|Δstart\| | 17s | **8s** |
| Mean \|Δend\| | 13s | **7s** |
| The Fray end error | +67s | **+1s** |
| Drake start error | +100s | **-14s** |
| False positives introduced | — | **0** |

#### Hybrid review categories (`fm95blo-2012-02-28`)

| Category | Count | Precision | Description |
|---|---|---|---|
| [AUTO] | 6 | **100%** | Coldplay, David Guetta, Outasight, Lil Wayne, Flo Rida, Guru Josh |
| [REVIEW-S] | 4 | — | The Fray (1 hit), Drake (merged), Childish Gambino (end uncertain), Gotye (wrong DB dur) |
| [REVIEW-H] | 4 | — | 3 unknown (ads?), 1 covering GT miss (Icelandic, Shazam confused by Zorba) |
| [EXT] | 2 | — | The Fray tail, Outasight tail |
| [DROP?] | 2 | — | 32s and 33s stubs, likely jingles |

**AUTO precision = 100% (6/6 true positives, 0 false positives).**
Auto-cut total duration: 21.1 min. All 6 start errors ≤19s, all 6 end errors ≤16s.

#### Classification thresholds (in `hybrid_review.py`)

| Threshold | Value | Purpose |
|---|---|---|
| `AUTO_MIN_HITS` | 2 | Minimum Shazam hits for [AUTO] |
| `AUTO_MIN_DURATION` | 60s | Minimum cut length for [AUTO] |
| `DROP_HEURISTIC_DUR` | 60s | Heuristic-only cuts shorter than this → [DROP?] |
| `DROP_SHAZAM_DUR` | 30s | Shazam cuts shorter than this → [DROP?] |
| `SHAZAM_CONFUSED_DUR` | 45s | If Shazam cut < this AND much shorter than heuristic → [REVIEW-H] |
| `SAME_SONG_MAX_GAP` | 120s | Adjacent same-song regions merged if gap ≤ this |

#### Workflow (run in order for any episode after finalcuts)

```bash
# 1. Normal Shazam scan
.venv/bin/python experiments/shazam_detect.py \
    --audio episodes/<ep>.mp3 \
    --candidates data/candidates/<ep>_finalcuts.json \
    --features data/features/<ep>.json \
    --transcript data/transcripts/<ep>.json \
    --episode-name <ep>

# 2. Gap scan
.venv/bin/python experiments/shazam_gap_scan.py \
    --audio episodes/<ep>.mp3 \
    --candidates data/candidates/<ep>_finalcuts.json \
    --episode-name <ep>

# 3. Combined review sheet (with merge + GT evaluation if available)
python3 experiments/make_review.py \
    --heuristic-csv data/labels/<ep>_review.csv \
    --shazam-json data/labels/song_matches_<ep>.json \
    --shazam-unmatched data/labels/song_unmatched_<ep>.csv \
    --gap-shazam-json data/labels/gap_shazam_matches_<ep>.json \
    --episode-name <ep> \
    [--gt-file data/labels/<ep>_actual_cuts.txt]

# 4. Hybrid classification
python3 experiments/hybrid_review.py \
    --review-csv data/labels/review_sheet_<ep>.csv \
    --episode-name <ep>
```

#### Output files per episode

| File | Use |
|---|---|
| `hybrid_review_audacity_<ep>.txt` | Import into Audacity — primary review file |
| `hybrid_auto_cuts_<ep>.csv` | Safe to cut automatically (100% precision so far) |
| `hybrid_manual_review_<ep>.csv` | REVIEW-S + REVIEW-H + EXT rows for manual check |
| `hybrid_review_<ep>.csv` | All rows, all fields, for analysis |
| `review_sheet_<ep>.csv` | Intermediate: full annotated sheet from make_review.py |
| `boundary_debug_<ep>.json` | Per-song Shazam debug: all sample times, offsets, end scan |

#### Known limitations (unchanged)

- ~45% of regions unmatched (Icelandic music not in Shazam database)
- GT miss #8 (01:16:16–01:19:13): heuristic covers it as [REVIEW-H] but Shazam cannot identify the song (likely Icelandic)
- `SHAZAM_CONFUSED_DUR` threshold (45s) removes the Zorba's Dance false stub correctly but song label "Zorba's Dance" still appears in [REVIEW-H] note
- Do not raise `AUTO_MIN_HITS` below 2 — single-hit matches are often false or have unreliable ends

---

### 2026-06-27 — Session 6: gap scan on new episode fm95blo-2012-02-28

**What changed:**
- Ran full pipeline (transcribe → audio_features → detect_breaks → finalize_cuts) on new episode `fm95blo-2012-02-28`.
- Ran gap scan on this episode.

**New episode stats:**
- Duration: 7107s (1h58m)
- Heuristic finalcuts: 17 candidates, well distributed
- 5 gaps ≥ 3 min found; 39 Shazam queries

**Gap scan results:**

| Gap | Song | Windows | Category | Tail in gap |
|---|---|---|---|---|
| 289–747s | The Fray - Heartbeat | 4/10 | tail of candidate 8–289s | 174s |
| 2063–2503s | Outasight - Tonight Is the Night | 1/10 | tail of candidate 1674–2063s | 24s only |
| 1429–1674s | — | 0 | correctly silent | — |
| 4893–5282s | — | 0 | correctly silent | — |
| 6311–6572s | — | 0 | correctly silent | — |

**Key findings:**
- **No GT08-style standalone missed break** — no new song appeared entirely within a gap. All matches are tails of existing heuristic candidates.
- **The Fray - Heartbeat** is a meaningful tail: song runs 243–463s, heuristic candidate ends at 289s, leaving 174s of song in the gap. Heuristic boundary is 174s early for this song.
- **Outasight** is a minor tail: only 24s of song bleeds into the gap.
- **Zero FPs** in 3 genuinely silent gaps (1429–1674s, 4893–5282s, 6311–6572s).
- Both matches correctly flagged UNCERTAIN (song_start is before gap start, meaning the song was already playing in the heuristic candidate period).

**Interpretation:**
This episode is better-covered by the heuristic than 2012-03-09. The gap scan correctly finds no standalone missed breaks. The tail findings confirm the previously observed pattern: heuristic candidate end boundaries can cut off songs early. The gap scanner behaves as expected.

**No GT to evaluate against** — this episode has no ground truth labels yet.

**Files added:**
- `data/transcripts/fm95blo-2012-02-28.json`
- `data/features/fm95blo-2012-02-28.json`
- `data/candidates/fm95blo-2012-02-28*.json`
- `data/labels/gap_shazam_matches_fm95blo-2012-02-28.{csv,json,txt}`

**Commits made:** None (all untracked)

**Next:** Label this episode for ground truth, or move on to another episode. Consider a boundary-extension pass to fix the tail pattern (Shazam the 2 min after each candidate end).

---

### 2026-06-28 — Session 8: 2011 cross-episode validation + REVIEW-B safety rule + promotion

**What changed:**
- Re-ran multi-sample Shazam scan on 2011 episode (new `shazam_detect.py` — multi-sample clustering)
- Ran `make_review.py` + `hybrid_review.py` on 2011 with GT evaluation
- Added `[REVIEW-B]` safety rule to `hybrid_review.py`: if AUTO cut starts >30s before heuristic region → boundary review
- Made `make_review.py` auto-call `hybrid_review.py` (promotion: one command generates all outputs)
- Updated `README.md` with hybrid workflow as standard pipeline

**2011 GT evaluation (13 music cuts, excluding 2 ads-only blocks):**

| Metric | Value |
|---|---|
| GT cuts matched | 11/13 (GT2 + GT3 are Icelandic, no Shazam match) |
| Mean \|Δstart\| | 51s (large heuristic region 112-1166s with 3 GT cuts inflates this) |
| Mean \|Δend\| | 79s (Snoop Dogg 15-min closing block inflates this) |

**Cross-episode hybrid classification results:**

| | fm95blo-2011-11-09 | fm95blo-2012-02-28 | Combined |
|---|---|---|---|
| AUTO cuts | 5 | 6 | **11** |
| REVIEW-B | 0 | 0 | 0 |
| AUTO precision | 100% | 100% | **100%** |
| AUTO recall | 38% (5/13) | 55% (6/11) | — |
| False positives | 0 | 0 | **0** |

**REVIEW-B safety rule evaluation:**
- Triggered: 0 times on either episode
- Calvin Harris (2011, Δstart=-51s vs GT) NOT downgraded: Shazam start is only 16s before heuristic region (threshold=30s)
- Root cause: heuristic itself started 34s early inside ads block; Shazam added 16s more. Safety rule compares Shazam to heuristic, not GT — can't catch this split-responsibility case
- Clean AUTO cuts unnecessarily downgraded: 0

**Files changed:**
- `experiments/hybrid_review.py` — added EARLY_START_THRESHOLD=30s, REVIEW-B category, REVIEW-B in all output paths
- `experiments/make_review.py` — added subprocess auto-call to hybrid_review.py at end of main(); added `import subprocess, sys`
- `README.md` — updated pipeline steps (6b–7), Shazam section, cross-episode validation table
- `fm95blö_cuts/fm95blö_cuts.md` — current stable pipeline updated, direction updated, session log added
- `data/labels/song_matches_fm95blo-2011-11-09.{csv,json}` (regenerated with multi-sample scan)
- `data/labels/hybrid_*_fm95blo-2011-11-09.*` (new)
- `data/labels/hybrid_*_fm95blo-2012-02-28.*` (regenerated)

**Commits made:** None (all untracked)

**Next:** Label `fm95blo-2012-03-09` for third-episode validation of AUTO precision.

---

*This note is the primary human-readable memory for this project. Code truth lives in git. Update this file at the end of every Claude session.*
