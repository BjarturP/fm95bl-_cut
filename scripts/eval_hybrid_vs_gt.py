"""Score hybrid_review_<ep>.csv against a ground-truth cuts file.

Usage:
    python scripts/eval_hybrid_vs_gt.py --episode-name fm95blo-2012-01-30 \
        --gt-file data/labels/2012-01-30_actual_cuts.txt
"""
import argparse, csv, re

def to_s(t):
    p = [float(x) for x in t.strip().split(':')]
    while len(p) < 3: p = [0]+p
    return p[0]*3600+p[1]*60+p[2]

def overlap(a,b,c,d):
    return max(0, min(b,d)-max(a,c))

def getf(r, *names):
    for n in names:
        if n in r and r[n] not in (None,''): return r[n]
    return ''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episode-name', required=True)
    ap.add_argument('--gt-file', required=True)
    ap.add_argument('--hybrid-csv', default=None,
                    help='defaults to data/labels/hybrid_review_<ep>.csv')
    ap.add_argument('--min-overlap', type=float, default=30.0)
    args = ap.parse_args()

    hybrid_csv = args.hybrid_csv or f'data/labels/hybrid_review_{args.episode_name}.csv'

    gt = []
    with open(args.gt_file) as f:
        for ln in f:
            m = re.match(r'music\s*-\s*([\d:]+)\s*-\s*([\d:]+)', ln.strip())
            if m: gt.append((to_s(m.group(1)), to_s(m.group(2))))

    rows = []
    with open(hybrid_csv) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    cuts = []
    for r in rows:
        s = getf(r,'start','start_s','pred_start','region_start')
        e = getf(r,'end','end_s','pred_end','region_end')
        try: s=float(s); e=float(e)
        except (ValueError, TypeError): continue
        lab = getf(r,'label','category','hybrid_label','class')
        song = getf(r,'song','song_title','title','note')
        cuts.append((s,e,lab,song))

    print(f"{args.episode_name}: {len(gt)} GT cuts, {len(cuts)} pipeline rows\n")
    print("GT match table:")
    print(f"{'#':<3}{'GT start':>9}{'GT end':>9}  {'best row':<12}{'Δstart':>8}{'Δend':>8}  song")
    matched_auto=0
    covered=0
    for i,(gs,ge) in enumerate(gt,1):
        best=None; bo=0
        for (s,e,lab,song) in cuts:
            o=overlap(gs,ge,s,e)
            if o>bo: bo=o; best=(s,e,lab,song)
        # recall counts any AUTO row overlapping this GT cut, not just the
        # best-overlap row (an EXT row can out-overlap the AUTO row it extends)
        auto_best=None; abo=0
        for (s,e,lab,song) in cuts:
            if 'AUTO' not in lab.upper(): continue
            o=overlap(gs,ge,s,e)
            if o>abo: abo=o; auto_best=(s,e,lab,song)
        if auto_best and abo>=args.min_overlap:
            matched_auto+=1
            best=auto_best; bo=abo
        if best and bo>=args.min_overlap:
            covered+=1
            s,e,lab,song=best
            print(f"{i:<3}{gs:>9.0f}{ge:>9.0f}  {lab:<12}{s-gs:>+8.0f}{e-ge:>+8.0f}  {song[:40]}")
        else:
            print(f"{i:<3}{gs:>9.0f}{ge:>9.0f}  {'*** MISSED':<12}{'':>8}{'':>8}")

    print(f"\nCoverage (any row ≥{args.min_overlap:.0f}s overlap): {covered}/{len(gt)}")
    print(f"AUTO recall: {matched_auto}/{len(gt)} = {100*matched_auto/len(gt):.0f}%")

    # AUTO precision: every AUTO row must overlap some GT cut
    auto_rows = [(s,e,lab,song) for (s,e,lab,song) in cuts if 'AUTO' in lab.upper()]
    auto_fp = [(s,e,lab,song) for (s,e,lab,song) in auto_rows
               if not any(overlap(gs,ge,s,e)>=args.min_overlap for gs,ge in gt)]
    n_auto = len(auto_rows)
    print(f"AUTO precision: {n_auto-len(auto_fp)}/{n_auto}"
          + (f" = {100*(n_auto-len(auto_fp))/n_auto:.0f}%" if n_auto else ""))

    print("\nNon-GT rows (potential false positives):")
    for (s,e,lab,song) in cuts:
        if any(overlap(gs,ge,s,e)>=args.min_overlap for gs,ge in gt): continue
        print(f"  {lab:<12}{s:>8.0f}{e:>8.0f}  {song[:45]}")

if __name__ == '__main__':
    main()
