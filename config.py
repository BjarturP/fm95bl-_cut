"""
Tunable thresholds and keyword list for break detection.

This file is meant to be edited as you calibrate against labeled episodes
(see calibrate.py) -- nothing here is final. Start conservative (biased
toward flagging too much, since the review step in review_export.py is
where you filter false positives).
"""

# --- Transcription ---
WHISPER_MODEL = "small"  # large-v3/medium too slow on 8GB RAM (medium measured ~0.24x realtime); revisit on bigger hardware
WHISPER_LANGUAGE = "is"

# --- Acoustic feature extraction (audio_features.py) ---
FEATURE_HOP_SECONDS = 1.0       # window size for rolling features
FEATURE_CHUNK_SECONDS = 300.0   # process audio in chunks this long to bound memory use on long episodes
SILENCE_RMS_DB = -40.0          # below this RMS (dBFS) counts as "silent"
SILENCE_MIN_DURATION = 8.0      # seconds; shorter silences are likely just pauses in speech
MUSIC_SMOOTHING_SECONDS = 10.0  # rolling-average half-width applied to music_score
LOUDNESS_JUMP_DB = 10.0         # sudden change between consecutive windows

# --- Combined music detection (detect_breaks.py) ---
# Calibrated against fm95blo_2011_11_09: acoustic music_score alone misses
# percussive/guitar-heavy songs (~0% recall). Combining it with how sparse
# the transcript's words are in the same window (music/song sections have
# far fewer recognized words than the host talking) got recall to ~92% with
# 1 false positive across a 2h05m episode. Re-validate as more episodes are
# labeled -- this was tuned on a single example.
WORD_RATE_WINDOW = 15.0        # seconds, +/- window for rolling word-rate
WORD_RATE_NORM_CAP = 3.0       # words/sec treated as "fully speech" (typical fast conversational pace)
MUSIC_COMBINE_WEIGHT_ACOUSTIC = 0.5
MUSIC_COMBINE_WEIGHT_WORDRATE = 0.5
# Hysteresis thresholds on the combined score (0..1): a run starts once the
# score crosses MUSIC_ENTER_THRESHOLD and keeps going through dips (talk-over,
# quiet bridges) until it drops below MUSIC_EXIT_THRESHOLD. A single shared
# threshold fragmented real songs into many sub-20s pieces.
MUSIC_ENTER_THRESHOLD = 0.60
MUSIC_EXIT_THRESHOLD = 0.40
MUSIC_MIN_DURATION = 20.0        # seconds; real songs are long, this filters brief blips

# --- Post-processing: merge nearby candidates, expand boundaries ---
# Per-window detection above gives the most-confident *core* of a break;
# real ad/music breaks on this show are continuous multi-minute blocks, so
# raw per-window candidates get bridged and grown outward before being
# treated as final cuts (see merge_nearby_candidates / expand_boundaries in
# detect_breaks.py). Calibrated against fm95blo_2011_11_09 user feedback:
# the detector was finding the right general areas but splitting real
# breaks into many small pieces and stopping short of the true edges.
MERGE_GAP_ALWAYS_BELOW = 10.0     # seconds; always bridge gaps this short, no speech check needed
MERGE_GAP_MAX = 75.0              # seconds; bridge gaps up to this long IF no real speech is in between
MERGE_GAP_SPEECH_RATE_NORM = 0.5  # gap average rate_norm at/above this = "real host speech", don't bridge
EXPAND_CLEAR_SPEECH_THRESHOLD = 0.35  # combined score below this = clearly talk, eligible to stop expansion
EXPAND_CLEAR_SPEECH_SUSTAIN_SECONDS = 8.0  # need this many consecutive seconds of clear speech to stop
EXPAND_MAX_SECONDS = 180.0        # cap on how far a boundary can grow outward in one direction

# --- Stage 3: final cuts (finalize_cuts.py) ---
# Stage 2 above is deliberately generous -- merge + expand exist to avoid
# missing real breaks. Calibration showed that generosity costs precision:
# blocks expanded into ambiguous audio further than is safe to actually cut
# on, and standalone weak keyword hits got treated as real candidates. This
# stage is the precision pass that turns "review candidates" (stage 2's
# output) into cuts you'd actually trust by default.
FINAL_STRONG_CONFIDENCE = 0.5          # candidates at/above this count as "strong" for bridging/anchoring decisions
FINAL_MERGE_GAP_DEFAULT = 45.0         # seconds; always bridge gaps this short between surviving candidates
FINAL_MERGE_GAP_LONG_BLOCK = 90.0      # seconds; bridge gaps up to this long ONLY between long blocks with no real talk between them
FINAL_MERGE_LONG_BLOCK_MIN_DURATION = 90.0  # a candidate counts as "long" (eligible for the wider gap above) once it's at least this long
FINAL_MERGE_GAP_SPEECH_RATE_NORM = 0.45     # gap avg rate_norm at/above this = real host conversation, blocks bridging
FINAL_EOF_TAIL_SECONDS = 30.0          # if the last candidate ends within this long of the file's end, treat the next gap as "near EOF"
FINAL_EOF_MERGE_GAP_MAX = 220.0        # near EOF, allow bridging a much larger gap (no-speech check still applies) -- end-of-show breaks often run straight through to the literal end of the recording
FINAL_CONTINUOUS_EVIDENCE_THRESHOLD = 0.45  # combined score counting as "real" music/ad evidence when trimming boundaries
FINAL_MAX_LEAD_IN_SECONDS = 25.0       # don't let a boundary sit more than this far from the nearest real evidence unless that whole stretch is itself continuously music/ad-like
FINAL_TRIM_LOW_CONFIDENCE = 0.6        # candidates below this confidence get their boundaries trimmed inward instead of trusted as-is
FINAL_TRIM_SKIP_NEAR_START_SECONDS = 60.0  # never trim a candidate's start boundary if it already starts this close to t=0 -- there's no real speech before the recording begins to protect against

# --- Keyword phrases (Icelandic) ---
# Phrases/vocabulary that tend to appear during ad reads or break announcements.
# Seeded from fm95blo_2011_11_09's actual ad blocks (prices, brand/sponsor
# mentions, discount language, station-ID sweeper). Spoken ads in this show
# are NOT acoustically distinct from normal talk, so this keyword signal is
# the main detector for them -- expect to need manual boundary review more
# often for ad candidates than for music candidates.
BREAK_KEYWORDS = [
    "afsláttur", "afslætti", "tilboð", "krónur", "kr.",
    "prósent", "facebook", "sími", "síminn", "fm95", "fm 95",
]
KEYWORD_PREROLL = 5.0    # seconds before keyword occurrence to include in candidate
KEYWORD_POSTROLL = 45.0  # seconds after keyword occurrence to include in candidate
KEYWORD_SUPPORT_WINDOW = 15.0  # how close a keyword must be to a non-speech run to "support" it

# --- Candidate scoring weights ---
# Final confidence = weighted combination of which signals fired.
WEIGHT_SILENCE = 0.3
WEIGHT_MUSIC = 0.4
WEIGHT_KEYWORD = 0.4
WEIGHT_LOUDNESS_JUMP = 0.1
KEYWORD_ONLY_CONFIDENCE_CAP = 0.35  # keyword with no acoustic support stays low-confidence
# review_export.py default decision is "remove" only at/above this. Set high
# (effectively disabling auto-remove) until calibrated on more episodes --
# on fm95blo_2011_11_09, a conf=0.8 candidate (silence+music+loudness-jump,
# the strongest combination the detector can produce) turned out to be a
# quiet talk-show pause, not music. With one labeled episode, no confidence
# level is proven safe to auto-remove; every candidate should default to
# "review" and you decide per-row in the CSV.
AUTO_REMOVE_CONFIDENCE = 0.85

# --- Export ---
CROSSFADE_MS = 80
