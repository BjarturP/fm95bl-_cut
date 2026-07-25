"""Simulate adding a host-speech gate to the DEFAULT (short-gap) merge tier.
Compare current vs proposed finalize merge on all labeled episodes."""
import json, re, copy
import config, detect_breaks as db

EPS = {
    'fm95blo_2011_11_09': None,   # GT from memory note (hardcoded below)
    'fm95blo-2012-02-28': 'data/labels/2012-02-08_actual_cuts.txt',
    'fm95blo-2012-03-09': 'data/labels/2012-03-09_actual_cuts_extended.txt',
    'fm95blo-2012-01-30': 'data/labels/2012-01-30_actual_cuts.txt',
}
# 2011 GT (music-only) in seconds, from the memory note table
GT_2011 = [(600,760),(1535,1728),(1525,1728),(1955,2098),(2694,2938),(3335,3538),
           (3936,4258),(5175,5557),(4676,4778)]  # approximate; we only count region-level

def to_s(t):
    p=[float(x) for x in t.strip().split(':')]
    while len(p)<3: p=[0]+p
    return p[0]*3600+p[1]*60+p[2]

def load_gt(path):
    gt=[]
    for ln in open(path):
        m=re.match(r'\w+\s*-\s*([\d:]+)\s*-\s*([\d:]+)', ln.strip())
        if m: gt.append((to_s(m.group(1)),to_s(m.group(2))))
    return gt

def overlap(a,b,c,d): return max(0,min(b,d)-max(a,c))

def gap_has_speech(t0,t1,ts,rn):
    return db._avg_in_range(t0,t1,ts,rn) >= config.FINAL_MERGE_GAP_SPEECH_RATE_NORM

def merge(cands, ts, rn, dur, gate_default, max_clean=360.0):
    cands=sorted(cands,key=lambda c:c['start'])
    merged=[copy.deepcopy(cands[0])]
    for c in cands[1:]:
        last=merged[-1]; gap=c['start']-last['end']
        near_eof=(c is cands[-1]) and (dur-c['end'])<=config.FINAL_EOF_TAIL_SECONDS
        if gap<=0: bridge=True
        elif gap<=config.FINAL_MERGE_GAP_DEFAULT:
            # PROPOSED (targeted): only gate a short gap on host speech when the
            # resulting merged region would be over-long (a normal song block is
            # a few min; bridging speech gaps into a 15-min region is the anomaly)
            would_len = max(last['end'],c['end']) - last['start']
            if gate_default and would_len > max_clean and gap_has_speech(last['end'],c['start'],ts,rn):
                bridge = False
            else:
                bridge = True
        elif near_eof and gap<=config.FINAL_EOF_MERGE_GAP_MAX: bridge=True
        elif gap<=config.FINAL_MERGE_GAP_LONG_BLOCK:
            lb=((last['end']-last['start'])>=config.FINAL_MERGE_LONG_BLOCK_MIN_DURATION
                or (c['end']-c['start'])>=config.FINAL_MERGE_LONG_BLOCK_MIN_DURATION)
            bridge=lb and not gap_has_speech(last['end'],c['start'],ts,rn)
        else: bridge=False
        if bridge: last['end']=max(last['end'],c['end'])
        else: merged.append(copy.deepcopy(c))
    return merged

for ep,gtpath in EPS.items():
    tr=json.load(open(f'data/transcripts/{ep}.json'))
    feat=json.load(open(f'data/features/{ep}.json'))
    cands=json.load(open(f'data/candidates/{ep}.json'))
    if isinstance(cands,dict): cands=cands.get('candidates',cands.get('cuts',[]))
    ts,rn,combined=db.build_combined_timeline(tr,feat)
    dur=feat['duration']
    # apply the same pre-merge steps finalize does (drop + trim) — approximate: use cands as-is
    cur=merge(cands,ts,rn,dur,gate_default=False)
    new=merge(cands,ts,rn,dur,gate_default=True)
    gt=load_gt(gtpath) if gtpath else GT_2011
    def recall(regions):
        return sum(1 for g in gt if any(overlap(*g,r['start'],r['end'])>=30 for r in regions))
    print(f"\n=== {ep} ({len(gt)} GT) ===")
    print(f"  regions: current={len(cur)}  proposed={len(new)}   GT-covered: cur={recall(cur)} new={recall(new)}")
    # show regions that differ
    cur_set={(round(r['start']),round(r['end'])) for r in cur}
    new_set={(round(r['start']),round(r['end'])) for r in new}
    only_new=sorted(new_set-cur_set)
    only_cur=sorted(cur_set-new_set)
    if only_cur: print(f"  REMOVED (current only): {only_cur}")
    if only_new: print(f"  ADDED (proposed only) : {only_new}")
