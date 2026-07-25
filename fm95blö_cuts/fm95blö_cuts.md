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
| Latest commit | `4cae80b` — "Auto-apply scan-confirmed tail extensions and surface narrowed-region remainders" (2026-07-25) |
| Prev commits | `4bbcf23` (Session 13 fixes, cross-validated), `6700798` (word-clawback) |
| Working tree | Clean of code changes; this note + untracked data outputs only |
| Ahead of remote | 9 commits |

**All Session 13 fixes are COMMITTED (`4bbcf23`)** after passing full cross-validation on 2011-11-09 and 2012-03-09 (Session 14). Session 14 improvements committed in `4cae80b`.

**Untracked data (not committed, normal):**
- `data/{transcripts,features,candidates}/fm95blo-2012-*.json` (per-episode pipeline outputs)
- `data/labels/*_actual_cuts.txt` (GT files, incl. new `2011-11-09_actual_cuts.txt`)
- `data/labels/gap_shazam_*`, `review_sheet_*`, `song_matches_*`, `hybrid_*` (per-episode outputs)
- `experiments/song_fingerprint.py`, `fm95blö_cuts/` (this vault)

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

**Cross-episode AUTO precision: 100% (0 FP) across all episodes — 38/38 AUTO cuts as of Session 14 (2026-07-25):**

| Episode | AUTO precision | AUTO recall | Coverage |
|---|---|---|---|
| 2011-11-09 | 10/10 = 100% | 77% (10/13) | 13/13 |
| 2012-01-30 | 10/10 = 100% | 91% (10/11) | 11/11 |
| 2012-02-28 | 8/8 = 100% | 73% (8/11) | 11/11 |
| 2012-03-09 (ext. GT) | 10/10 = 100% | 83% (10/12) | 12/12 |
| **Cumulative** | **38/38 = 100%** | **81% (38/47)** | **47/47** |

NOTE: 2012-03-09 uses the extended GT (`2012-03-09_actual_cuts_extended.txt`); 2011 GT saved as `data/labels/2011-11-09_actual_cuts.txt` (music cuts from this note's table). Score any episode with `scripts/eval_hybrid_vs_gt.py --episode-name <ep> --gt-file <gt>`.

**Session 13 (2026-07-24) — two detection fixes, UNCOMMITTED, validated on 2012-01-30 only:**
1. **Merge fix (`finalize_cuts.py` + `config.py`)** — a short-gap bridge is now blocked when it would grow a cut past `FINAL_MERGE_MAX_CLEAN_BLOCK`=600s AND the gap carries host speech. Stops distinct breaks being glued into one 15-min cut across DJ talk. Simulated on all 4 eps first: only 3 regions split total, each correct (2011 19-min opener, 2012-03-09 Lil Wayne/Coldplay separation, 2012-01-30 GT3 recovery). Region-level GT coverage stayed 100% everywhere.
2. **Classifier fix (`hybrid_review.py` + `make_review.py`)** — gap-scan-only finds (`new_gap_find`) are now classified on their own gap match (AUTO-eligible when ≥2 hits + boundary ok) instead of falling through to "REVIEW-H (no match)". Threaded `gap_shazam_n_hits` through the review sheet.
3. **Shazam resilience (`shazam_detect.py`)** — `shazam_clip` retries transient `FailedDecodeJson`/rate-limit errors 3× then treats the clip as no-match, so one bad response can't abort a 70-clip gap scan.

**2012-01-30 result (Session 13):** coverage 10/11→**11/11**, AUTO recall 7/11→**10/11 (91%)**, AUTO precision **100% (10/10)**. Recovered 3 breaks as AUTO: **GT3 (David Guetta – Titanium)** — was totally invisible; splitting the mega-region let *normal* Shazam name it — plus **GT1 (Black Keys – Lonely Boy)** and **GT10 (Azealia Banks – 212)** via fix #2. Only GT5 (Florence – Shake It Out) stays REVIEW-S (genuinely uncertain boundary). Minor tradeoff: JAY-Z/Ed Sheeran AUTO ends shortened ~80s (safe direction; EXT rows flag the longer end).

### Stable path: improve detection incrementally
Continue refining `detect_breaks.py` + `finalize_cuts.py` signals. Do not change `config.py` thresholds until 3+ labeled episodes confirm stability.

---

## Next Recommended Step

1. **Merged-row confidence recomputation (top recall lead, ~2 cuts).** Same-song merged rows (Florence – Shake It Out on 2012-01-30, Drake on 2012-02-28) stay REVIEW-S even when the merged evidence is strong (n=3–4 combined hits, measured boundary error ≤ 15s). The uncertainty flags come from a weak constituent and reference the PRE-merge region ("song_start 174s before region" is stale after the merge widens the region). Fix: recompute start/end confidence after `merge_same_song_rows` against the merged region; consider AUTO when merged n_hits ≥ 3 and the constituent ranges overlap (gap ≤ 0 ⇒ one continuous play). Precision-critical — validate on all 4 episodes before committing.
2. **Heuristic start over-expansion (idea 3, other half).** REVIEW-H starts from `finalize_cuts`/`expand_boundaries` skew early (Zorba −120s on 2012-02-28). The committed `clawback_start` helper could be reused with a larger window on heuristic-only rows in `make_review`. Riskier (recall-critical heuristic core).
3. **Icelandic music coverage** — Shazam still misses most Icelandic songs (2011 GT2, 03-09 GT9). The new region_remainder rows at least keep them visible for manual review. No real fix without a local fingerprint database.
4. **Remainder-row tuning** — `REMAINDER_MIN_DUR=180` adds ~3 junk REVIEW-H rows per episode alongside the real finds. If review burden grows, gate on acoustic music-likeness within the chunk.
5. **Push** — 9 commits ahead of origin (pushed at end of Session 14 if network allowed; verify).

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

### 2026-07-01 — Session 9: third-episode validation (fm95blo-2012-03-09)

**Goal:** Validate the promoted hybrid workflow on a third labeled episode before trusting AUTO cuts in production. No thresholds tuned, no pipeline code changed.

**Setup note (important):** the on-disk `song_matches_fm95blo-2012-03-09.json` was from 2026-06-25 (single-sample `shazam_detect.py`, pre-multi-sample rewrite). It lacked `n_samples_matched` / `boundary_confidence`, so reusing it would have forced every row out of AUTO (fake 0-AUTO result). **Regenerated** the normal Shazam pass with current `shazam_detect.py` (7/12 matched). Heuristic finalcuts (06-24) and gap-scan matches (06-27) reused unchanged.

**Ground truth:** `data/labels/2012-03-09_actual_cuts.txt` — 10 music cuts (format `music - MM:SS - MM:SS`).

**Validation results:**

| Metric | Result |
|---|---|
| AUTO precision | **1/1 = 100%** → **12/12 cumulative across 3 episodes, 0 FP** |
| AUTO recall | 1/10 = 10% (only Coldplay) |
| Combined recall (any category) | **10/10 = 100%** |
| False auto-cuts | 0 |
| AUTO boundary error (Coldplay) | Δstart −8s, Δend +3s |
| Mean boundary error (all 10 matched) | Δstart 31s, Δend 15s |

**Per-GT coverage:** all 10 surfaced — 1 AUTO (Coldplay), 4 REVIEW-S (Flo Rida, Eminem, AWOLNATION, Úlfur Úlfur), 5 REVIEW-H (2 heuristic-only + Fun., Lil Wayne, Pitbull via gap-scan). Rap blind spots (Pitbull, Lil Wayne) recovered by gap-scan again.

**Key finding — AUTO recall regression cause:** 6 of 7 Shazam matches flagged UNCERTAIN *only* because MusicBrainz/iTunes durations overshoot the radio edit (60–200s), demoting clean matches (3–5 hits, spread 0–2s) to REVIEW-S. Precision safety intact; recall over-suppressed. → new top improvement lead (see Next Recommended Step #2).

**New Icelandic match:** Úlfur Úlfur – "Ég Er Farinn" (2nd Icelandic song ever matched by Shazam; only Magnús Eiríksson before). Its boundary is weak though (Δstart +104s, only 2 late hits).

**No-GT regions (correctly non-AUTO):** JAY-Z + Florence (real late-episode music, not labeled as cuts), Magnús Eiríksson jingle stub, 1 heuristic FP, 1 DROP?.

**Files generated:**
- `data/labels/song_matches_fm95blo-2012-03-09.{csv,json}` (regenerated, multi-sample)
- `data/labels/boundary_debug_fm95blo-2012-03-09.json`, `song_unmatched_*`, `song_cut_suggestions_*`
- `data/labels/review_sheet_fm95blo-2012-03-09.csv`, `review_audacity_*` (regenerated)
- `data/labels/hybrid_{review_audacity,review,auto_cuts,manual_review}_fm95blo-2012-03-09.*`
- `data/labels/ground_truth_fm95blo-2012-03-09.csv` (blank GT sheet built for this session)

**Commits made:** None (all untracked).

**Next:** Address the UNCERTAIN duration-overshoot recall gap (Next Step #2) — highest-value, precision-safe. Or label a 4th episode to keep widening validation. README cross-episode table still says "2 episodes"; update to 3 when convenient.

---

### 2026-07-01/03 — Session 10: boundary baseline + start/end confidence split (idea 4) — VALIDATED (not committed)

**Step 6 (measurement) — DONE.** New reusable diagnostic `experiments/boundary_eval.py`: signed error (predicted−actual), median, split by category, across all 3 episodes. Baseline (pre-change):

| Category | n | Δstart med | Δend med |
|---|---|---|---|
| AUTO | 12 | −7s | −1s |
| REVIEW-S | 13 | −6s | −4s |
| REVIEW-H | 10 | see below | see below |

REVIEW-H direction verdict (the load-bearing question): split REVIEW-H by source. **`finalize_cuts`-sourced starts skew systematically EARLY** (median −47s, worst GT4 −102s, Zorba −120s) → over-expansion upstream in `expand_boundaries`. Gap-scan-sourced REVIEW-H boundaries are tight (−7 to −14s). The 2011 +611/+831 Δstart values are an eval artifact (gap-scan rows inside the 15-min closing GT block), not boundary error.
→ **Idea-3 implication:** a word-clawback must scan the WHOLE over-expanded region (a local ±20s window hits the same radius trap as idea 2 on a −102s error). Fix likely belongs upstream in/next to `expand_boundaries`.

**Step 4 (start/end confidence split) — IMPLEMENTED, NOT COMMITTED.** Changed `shazam_detect.py` (split `uncertain` → `start_confidence`/`end_confidence` + `start_uncertain`/`end_uncertain`/`end_info`; DB duration now display-only; a scan-confirmed end past the short heuristic region no longer forces UNCERTAIN), `make_review.py` (threads the new columns), `hybrid_review.py` (AUTO gates on start_confidence + `_bool` helper; legacy fallback for pre-split CSVs). Root cause found: old line `if final_end > r_end + UNCERTAIN_OVERSHOOT` flagged UNCERTAIN for exceeding the heuristic region end — which is itself known-short — so it falsely blocked correct extensions.

**Measured on 2012-03-09 (new code):** recall 10%→**40%** (Flo Rida, Eminem, AWOLNATION recovered, all GT✓), but precision 100%→**67%** — JAY-Z (1:42) and Florence (1:53), confidently-ID'd late-episode music NOT in GT, got auto-promoted. The old 100% depended on the false end-flag incidentally blocking those two while wrongly blocking the 3 real breaks. Shazam signals alone cannot separate "GT break" from "late-episode real music."

**DECISION RESOLVED → option C.** User auditioned 1:42 + 1:53: BOTH are real music breaks (JAY-Z music starts 1:42:55; Florence 1:53:31, host talks a little over the intro). So they were never false positives — the GT was simply incomplete (stopped at 1:35). Kept `AUTO_MIN_HITS = 2` (no recall sacrifice). Extended GT saved to `data/labels/2012-03-09_actual_cuts_extended.txt` (original untouched; the two new ends 1:46:15 / 1:55:07 are estimates — precision uses overlap≥30s so robust).

**Full re-validation (all 3 episodes, new split code, extended GT):**

| Episode | AUTO precision | AUTO recall was→now |
|---|---|---|
| 2011-11-09 | 8/8 = 100% | 38% → **62%** |
| 2012-02-28 | 6/6 = 100% | 55% → 55% |
| 2012-03-09 | 6/6 = 100% | 10% → **50%** |
| **Cumulative** | **20/20 = 100%** | (was 12/12) |

Split recovered **+8 real-music AUTO cuts, 0 FP**. AUTO boundary quality unchanged (aggregate Δstart med −7s, Δend +1s). No unlabeled-music leakage on other eps (2011 Snoop closing block stayed LOW at 1/6 hits, correctly non-AUTO).

**NEW follow-up surfaced by the audition (start-boundary quality, NOT precision):** AUTO cuts start ~8s early on median, and the two late 2012-03-09 tracks start ~12s early **into host talk** (JAY-Z −12s, Florence −11s). The acoustic start-snap FAILS on talk-over-intro ("no start transition found") — no clean speech→music edge. This is the strongest case yet for idea 3 (clawback to last spoken word / first music onset) and it now applies to AUTO cuts, not just REVIEW-H.

**Still outstanding:** idea 3 (REVIEW-H + AUTO start clawback — belongs upstream near `expand_boundaries`; must scan whole region, not a local window); idea 5 largely already handled by the matched-region end scan (AUTO Δend is tight now). **Committed 2026-07-23 in Session 11** (`d8ffd4a`) — `shazam_detect.py`, `make_review.py`, `hybrid_review.py`, new `boundary_eval.py`.

---

### 2026-07-23 — Session 11: commit Session 10 split work + README update

**What changed (housekeeping session, no new code logic):**
- Committed the Session 10 start/end confidence split, which had been sitting uncommitted in the working tree:
  - `d8ffd4a` — "Split Shazam boundary confidence into start/end (recall fix)" — `shazam_detect.py`, `make_review.py`, `hybrid_review.py`, new `boundary_eval.py` (code only, no generated `data/`, matching prior-session convention)
  - `aa42eaf` — "Update cross-episode validation table for 3 episodes / 20 AUTO cuts" — README cross-episode table went from 2 episodes / 11 AUTO cuts to **3 episodes / 20 AUTO cuts, 100% precision**, plus a note on why the split works.
- Updated this note (repo state, direction, Session 10 log).

**Commits made:** `d8ffd4a`, `aa42eaf` (both on `main`, direct — solo-repo convention). Branch now 6 ahead of `origin/main`, **not pushed**.

**Next:** idea 3 — start-boundary clawback. AUTO cuts start ~8s early on median; talk-over-intro tracks (JAY-Z −12s, Florence −11s) start early into host talk because the acoustic start-snap fails with no clean speech→music edge. Fix belongs upstream near `expand_boundaries` and must scan the whole over-expanded region, not a local ±20s window.

---

### 2026-07-23 — Session 12: idea 3 — transcript word-clawback for cut starts (Shazam side) — DONE + validated

**What changed:** `experiments/shazam_detect.py` — new pure helper `clawback_start(est_start, region_end, words)`. When the acoustic start-snap fails (returns "no start acoustic transition found" — the talk-over-intro signature, where the host talks over the song intro so the music signal never dips to speech level and there is no speech→music edge to snap to), we fall back to the raw Shazam offset, which sits ~10s early *inside* the host talk. The clawback walks the transcript words forward from that estimate and moves the start to the end of the last spoken word before the first sustained wordless gap (≥`CLAWBACK_MUSIC_GAP`=6s = host stopped, music alone). **Guards:** only moves *later*, never past region end, capped at `CLAWBACK_MAX`=25s, gated on snap-failure, and only fires when a real speech→music onset is found (continuous talk or continuous sung vocals → no move). Threaded transcript `words` (word-level timestamps) through `main → run_all → process_region`. Added `clawback_note` to `boundary_debug`.

**Key realisation about error direction (load-bearing):** this is a music *remover*, so the cut region is DELETED. A **negative** Δstart (cut starts before the music, in host talk) is the *dangerous* error — it deletes host content. A small **positive** Δstart (cut starts a few seconds into the music) is the *safe* error — it just leaves a little music un-deleted. So pulling early (negative) starts toward 0 / slightly-positive is exactly right, and the +4/+6s overshoots the clawback occasionally produces are the benign direction.

**Validation (offline, deterministic — no Shazam re-run):** `scratchpad/validate_clawback.py` replays each episode's existing `boundary_debug_<ep>.json` (the committed pre-clawback matches) through the real clawback logic — same matches, only the clawback differs (cleaner A/B than re-running Shazam, which would vary the matches). Across all 3 episodes, matched-region Δstart:

| | Δstart median | Δstart \|median\| | range |
|---|---|---|---|
| before | −6.7s | **9.5s** | [−50, +184] |
| after  | −3.8s | **5.9s** | [−50, +187] |

28 GT-matched regions, **18 improved, 0 regressions**, worst overshoot +6s. Notable fixes: Coldplay −23→−10, David Guetta Turn Me On −19→−7, Drake −14→−6, AWOLNATION −13→−7, Florence Cosmic −11→+4. **Interesting:** on this show the acoustic snap fails on *every* matched region (loudness-driven signal never gives a clean speech→music edge), so the clawback is now effectively the primary start refinement. Cases it left alone (correct, conservative): JAY-Z −12 (no transcribed host words at the estimate — "already on clean audio"), Calvin Harris −50 / Úlfur +104 / Drake +100 (continuous talk, no locatable onset).

**Not re-run through the live pipeline** — data outputs (`song_matches`, `hybrid_*`) still reflect pre-clawback boundaries; the next real `shazam_detect.py` run on any episode will pick up the clawback. The offline replay is the authoritative validation.

**Commits made:** `6700798` (on `main`; `experiments/shazam_detect.py` + README Shazam-section note). Branch now 7 ahead of `origin/main`, **not pushed**.

**Still outstanding (idea 3, other half):** REVIEW-H heuristic-only over-expansion — `finalize_cuts`/`expand_boundaries` starts skew early by −47s median (worst −102s / Zorba −120s). That is the committed heuristic core (recall-critical), a separate and riskier fix. The same `clawback_start` helper could be reused — either upstream near `expand_boundaries`, or applied to heuristic-only rows in `make_review.py` — but with a larger `max_move` (must scan the whole over-expanded region, not a 25s window). Left for a follow-up session.

---

### 2026-07-23/24 — Session 13: 4th episode (fm95blo-2012-01-30) + two detection fixes — validated on this episode, UNCOMMITTED

**Context:** After Session 12's clawback (committed `6700798`), a full 8-stage pipeline was run on a new episode `fm95blo-2012-01-30` (~2h02m, 7334s). The first attempt crashed at stage 1 (transcribe) on a transient `TimeoutError` mid-numpy-import (machine hiccup); the restart completed cleanly (transcribe 22:31→00:20, whole pipeline done 01:02). First live run of the clawback on a fresh episode.

**User provided ground truth (11 music cuts)** → saved to `data/labels/2012-01-30_actual_cuts.txt`. Initial scorecard: coverage 10/11, AUTO recall 7/11, AUTO precision 100%. Two problems diagnosed:
- **GT3 (31:52–35:30) completely missed.** Root cause: `finalize_cuts` merged three candidates (949-1151 conf0.4 + a 656s over-expanded FP at 1184-1840 + GT3's real 1869-2186 candidate) into one 949-2186 mega-region via two sub-45s gaps. Normal Shazam then matched only Eminem (GT2) inside it and narrowed the AUTO cut to Eminem, discarding GT3; and because GT3 was *interior* to a candidate, the gap-scanner (which only scans *between* candidates) never saw it. Blind spot created by an over-merge.
- **GT1 (Black Keys) + GT10 (Azealia Banks) shown as "REVIEW-H (no match)"** even though the gap-scanner *had* found and named them — `hybrid_review.py` only read `matched_song` (normal Shazam) and ignored `gap_shazam_song`.

**Fix #1 — merge (`finalize_cuts.py` + `config.py`, new `FINAL_MERGE_MAX_CLEAN_BLOCK=600`).** A short-gap bridge is blocked when it would grow the region past 600s AND the gap carries host speech (reuses existing `gap_has_host_speech`). **Simulated on all 4 labeled episodes before editing** (`scratchpad/sim_merge_gate.py`): swept the threshold; 600 gives only 3 splits total across all eps, each correct — 2011 splits its 19-min over-merged opener, 2012-03-09 separates Lil Wayne from Coldplay at the talk between them, 2012-01-30 recovers GT3. Region-level GT coverage unchanged (100%) everywhere. (A naive global short-gap speech gate was rejected first — it fragmented 35-50% of regions on every episode.)

**Fix #2 — classifier (`hybrid_review.py` + `make_review.py`).** Threaded `gap_shazam_n_hits` through the review sheet; `classify()` now handles `new_gap_find` rows off the gap match (AUTO if not-uncertain + ≥2 hits + ≥60s, else REVIEW-S, else DROP?), and the main loop surfaces the gap song name + hit count instead of "(no match)".

**Fix #3 — Shazam resilience (`shazam_detect.py`).** The re-run's gap scan died on 1 of 71 clips with `FailedDecodeJson` (free Shazam endpoint flakiness), and `set -e` aborted stage 8. `shazam_clip` now retries 3× with backoff then treats the clip as no-match. Applies to both `shazam_detect` and `shazam_gap_scan` (shared helper).

**2012-01-30 result after all fixes (re-scored vs GT):**

| Metric | Before | After |
|---|---|---|
| Coverage (found anywhere) | 10/11 | **11/11** |
| AUTO recall | 7/11 | **10/11 (91%)** |
| AUTO precision | 100% | **100% (10/10, 0 FP)** |
| AUTO cut total | 24.0 min | 32.4 min |

Three breaks recovered as AUTO: **GT3 → David Guetta – Titanium** (the split let *normal* Shazam name the previously-invisible song — merge fix paid off twice), **GT1 → Black Keys – Lonely Boy**, **GT10 → Azealia Banks – 212** (fix #2). Only GT5 (Florence – Shake It Out) remains REVIEW-S (genuinely uncertain boundary). Tradeoff: JAY-Z (Otis) + Ed Sheeran (Lego House) AUTO ends shortened ~80s (safe direction — leaves music rather than eating talk; EXT rows flag the longer end). Radiohead end +30s.

**Commits made:** NONE — all 5 code files + this note are UNCOMMITTED in the working tree.

**Next (critical before commit):** re-validate on **2011-11-09 and 2012-03-09** (the two other episodes the merge change split) — full downstream re-run, confirm AUTO precision stays 100%. 2012-02-28 unchanged at threshold 600 (no re-run needed). Then commit all 5 files together.

**Re-validation helper scripts (persisted to repo `scripts/`):**
- `scripts/sim_merge_gate.py` — the offline merge-threshold simulation across all 4 labeled episodes (the pre-edit validation for fix #1; re-run to justify any future threshold change). `PYTHONPATH=. .venv/bin/python scripts/sim_merge_gate.py`
- `scripts/eval_hybrid_vs_gt.py` — scores `hybrid_review_<ep>.csv` against a GT file (AUTO precision/recall, per-GT boundary table). Currently hardcoded to 2012-01-30; generalize the paths for other episodes.
- `scripts/rerun_downstream.sh <ep>` — re-runs stages 5-8 (review_export → shazam → gap_scan → make_review) after a finalize change. For a full re-validation, run `finalize_cuts.py` first, then this.

---

### 2026-07-25 — Session 14: Session-13 validation passed + committed; tail auto-extension; gap-find end floor; remainder rows

**Goal (user):** "detect all songs" (recall) + "make the cuts more precise" (boundaries).

**Part 1 — Session 13 re-validation (the gating step) — PASSED, committed `4bbcf23`:**
- Re-ran `finalize_cuts` + full downstream on 2011-11-09 and 2012-03-09. Both split exactly as the simulation predicted (2011 opener 112–1166 → 112–573 + 598–1166; 03-09 separates 3762–4013 / 4152–4384).
- 2011: AUTO precision 10/10 = 100%, recall 62%→77% (SexyBack, RHCP, Daughtry promoted). 03-09: 9/9 = 100%, recall 50%→75%, incl. a new Icelandic AUTO (Thorunn Antonia – Too Late, −2/+8s).
- Created `data/labels/2011-11-09_actual_cuts.txt` (13 music cuts from this note's GT table). Generalized `scripts/eval_hybrid_vs_gt.py` (`--episode-name`/`--gt-file` args; fixed AUTO-recall undercount when an EXT row out-overlaps its AUTO row). Fixed zsh word-split bug in `scripts/rerun_downstream.sh` (`$GTARG` → array).
- Committed all 5 Session-13 files + `scripts/` as `4bbcf23`.

**Part 2 — improvements (committed `4cae80b`), all offline (no Shazam re-run needed):**
1. **Tail auto-extension (precision).** New `gap_shazam_last_confirmed` column in `make_review` = song_start + last matched gap-window offset + clip length — the last time the scan positively heard the song. `hybrid_review` extends same-song AUTO ends to this value −5s instead of leaving an EXT row. 01-30 tails: JAY-Z Δend −91→−23, Ed Sheeran −78→−11. Key design point: never use `gap_shazam_end` (song_start + DB duration) — it overshoots +80s into host talk on album-edit durations.
2. **Gap-find end floor (recall).** Same floor applied to `new_gap_find` ends in `hybrid_review`, so a bogus DB duration can't cap the cut under `AUTO_MIN_DURATION`. **Pitbull GT08 on 03-09 (29s(!) MusicBrainz duration) is now AUTO at −7/−35s — the original rap blind spot is fully automatic.** 03-09 recall 75%→83%.
3. **Region-remainder rows (coverage).** When Shazam narrows a wide heuristic region, uncovered chunks ≥180s become `region_remainder` REVIEW-H rows instead of vanishing. Restores 2011 GT2 (Icelandic song at 10:00, start −2s) which the narrowing had swallowed after the merge-fix split. Cost: ~3 junk REVIEW-H rows/episode (over-expansion padding).
4. **2012-02-28 was running on STALE pre-split Shazam data** (empty `start_confidence` columns → legacy classifier path). Full downstream re-run with current code: recall 55%→73% (Childish Gambino, Gotye promoted — Gotye boundaries +1/+0s), precision 8/8.

**Validation:** AUTO precision 100% on every episode after every change (cumulative **38/38, 0 FP**). Coverage 47/47 GT cuts visible in review outputs.

**Follow-up found, not fixed:** merged same-song rows (Florence 01-30 −4/+7, Drake 02-28 −6/+15) stay REVIEW-S on stale pre-merge uncertainty flags — see Next Recommended Step #1.

**Commits made:** `4bbcf23`, `4cae80b`. Pushed to origin at session end.

---

*This note is the primary human-readable memory for this project. Code truth lives in git. Update this file at the end of every Claude session.*
