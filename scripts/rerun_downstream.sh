#!/bin/zsh
# Re-run stages 5-8 (post-finalize) for an episode after the merge/classifier fixes.
set -e
cd /Users/bjarturpall/Documents/BjarturPall/Haskoli/drasl/Projects/claudetest2
EP=$1
AUDIO=episodes/${EP}-mp3.mp3
[ -f "$AUDIO" ] || AUDIO=episodes/${EP}.mp3
PY=.venv/bin/python
TR=data/transcripts/$EP.json
FE=data/features/$EP.json
FINAL=data/candidates/${EP}_finalcuts.json
GT=data/labels/${EP#fm95blo-}_actual_cuts.txt

stage () { echo "\n===== [$(date '+%H:%M:%S')] $1 ($EP) ====="; }

stage "5 review_export"
$PY review_export.py --candidates $FINAL --out-csv data/labels/${EP}_review.csv --out-audacity data/labels/${EP}_audacity.txt

stage "6 shazam_detect"
$PY experiments/shazam_detect.py --audio $AUDIO --candidates $FINAL --features $FE --transcript $TR --episode-name $EP

stage "7 shazam_gap_scan"
$PY experiments/shazam_gap_scan.py --audio $AUDIO --candidates $FINAL --episode-name $EP

stage "8 make_review (+hybrid)"
GTARG=()
[ -f "$GT" ] && GTARG=(--gt-file "$GT")
$PY experiments/make_review.py \
    --heuristic-csv    data/labels/${EP}_review.csv \
    --shazam-json      data/labels/song_matches_${EP}.json \
    --shazam-unmatched data/labels/song_unmatched_${EP}.csv \
    --gap-shazam-json  data/labels/gap_shazam_matches_${EP}.json \
    --episode-name     $EP "${GTARG[@]}"

stage "DONE $EP"
