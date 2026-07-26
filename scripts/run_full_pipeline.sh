#!/bin/zsh
set -e
cd /Users/bjarturpall/Documents/BjarturPall/Haskoli/drasl/Projects/claudetest2
EP=${1:?episode name required}
AUDIO=episodes/${EP}-mp3.mp3
PY=.venv/bin/python

stage () { echo "\n===== [$(date '+%H:%M:%S')] $1 ($EP) ====="; }

stage "1 transcribe"
$PY transcribe.py $AUDIO --out data/transcripts/$EP.json

stage "2 audio_features"
$PY audio_features.py $AUDIO --out data/features/$EP.json

stage "3 detect_breaks"
$PY detect_breaks.py --transcript data/transcripts/$EP.json --features data/features/$EP.json --out data/candidates/$EP.json

stage "4 finalize_cuts"
$PY finalize_cuts.py --candidates data/candidates/$EP.json --transcript data/transcripts/$EP.json --features data/features/$EP.json

stage "5-8 downstream"
zsh scripts/rerun_downstream.sh $EP

stage "ALL DONE"
