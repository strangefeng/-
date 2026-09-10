"""Audit source labels, group related exports, and materialize a reproducible split.

Group identity is inferred from original export names and perceptual duplicates,
NOT verified aircraft/flight identity. Raw files are never modified.
"""
from pathlib import Path
import argparse, csv, hashlib, json, re, shutil
from collections import Counter, defaultdict
import numpy as np
from PIL import Image
import cv2, yaml

NAMES = ['missing-rivet', 'rust', 'scratch', 'crack', 'dent']
ROOT = Path(__file__).resolve().parents[1]

def dump_csv(path, rows):
    if not rows:
        path.write_text('path,issue\n',encoding='utf-8-sig'); return
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source', type=Path, default=ROOT.parent/'开源数据集'); args=ap.parse_args()
    out=ROOT/'data'/'public_v2'; reports=ROOT/'reports'; reports.mkdir(exist_ok=True)
    if (out/'data.yaml').exists(): raise SystemExit('Frozen output exists; use a new version, do not overwrite.')
    rows=[]; issues=[]; pixel_seen={}; parent=[]; names_seen={}; hashes=[]
    def find(i):
        while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[max(a,b)]=min(a,b)
    # Use original declared sources; derived merged copy audited separately, not added twice.
    sources=[p for p in sorted(args.source.iterdir()) if (p/'data.yaml').exists() and 'plus_external' not in p.name]
    for src in sources:
        cfg=yaml.safe_load((src/'data.yaml').read_text(encoding='utf-8-sig'))
        local=cfg['names']; local=list(local.values()) if isinstance(local,dict) else local
        mapping={i:NAMES.index(n.replace('scractch','scratch')) for i,n in enumerate(local)}
        for p in sorted(src.rglob('*.jpg')):
            lab=p.parent.parent/'labels'/(p.stem+'.txt')
            try:
                if not lab.exists(): raise ValueError('missing label; not treated as negative')
                a=np.asarray(Image.open(p).convert('RGB')); h,w=a.shape[:2]
                boxes=[]
                for line in lab.read_text().splitlines():
                    vals=list(map(float,line.split()))
                    if len(vals)>5 and len(vals)%2==1:
                        points=np.asarray(vals[1:]).reshape(-1,2)
                        if not np.isfinite(points).all() or (points<0).any() or (points>1).any(): raise ValueError('invalid polygon')
                        low=points.min(0); high=points.max(0); center=(low+high)/2; size=high-low
                        vals=[vals[0],*center,*size]
                    if len(vals)!=5: raise ValueError('expected box or polygon')
                    c,x,y,bw,bh=vals
                    if not np.isfinite(vals).all() or c!=int(c) or int(c) not in mapping: raise ValueError('invalid class or nonfinite')
                    if not (bw>0 and bh>0 and x-bw/2>=-1e-5 and y-bh/2>=-1e-5 and x+bw/2<=1.00001 and y+bh/2<=1.00001): raise ValueError('invalid box bounds')
                    boxes.append([mapping[int(c)],x,y,bw,bh])
                pixel=hashlib.sha256(a.tobytes()+str(a.shape).encode()).hexdigest()
                origin=re.split(r'\.rf\.',p.stem)[0].lower()
                sig=hashlib.sha256(json.dumps(sorted(boxes)).encode()).hexdigest()
                gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY); d=cv2.dct(cv2.resize(gray,(32,32)).astype(np.float32))[:8,:8].flatten()[1:]
                ph=sum(int(v) << j for j,v in enumerate(d>np.median(d)))
                idx=len(rows); parent.append(idx)
                if origin in names_seen: union(idx,names_seen[origin])
                else: names_seen[origin]=idx
                duplicate=False
                if pixel in pixel_seen:
                    old=pixel_seen[pixel]; union(idx,old)
                    duplicate=rows[old]['label_sha']==sig
                    if not duplicate:
                        issues.append({'path':str(p),'issue':'conflicting annotations for identical pixels; both excluded'})
                        rows[old]['conflict']=True
                else: pixel_seen[pixel]=idx
                conflict=pixel in pixel_seen and rows[pixel_seen[pixel]]['label_sha']!=sig if pixel_seen[pixel]!=idx else False
                rows.append(dict(path=str(p.resolve()),source=src.name,original_split=p.parent.parent.name,origin=origin,pixel_sha=pixel,label_sha=sig,width=w,height=h,boxes=boxes,duplicate=duplicate,conflict=conflict))
                hashes.append(ph)
            except Exception as e: issues.append({'path':str(p),'issue':str(e)})
    # Conservative near-duplicate grouping; grouping is not label transfer or deduplication.
    for i in range(len(rows)):
        for j in range(i):
            if (hashes[i]^hashes[j]).bit_count()<=4: union(i,j)
    groups=defaultdict(list)
    for i,r in enumerate(rows): groups[find(i)].append(i)
    cross=sum(len({rows[i]['original_split'] for i in ids})>1 for ids in groups.values())
    split_counts=Counter(); class_counts=defaultdict(Counter); manifests=[]
    for ids in groups.values():
        gid=hashlib.sha256('|'.join(sorted({rows[i]['pixel_sha'] for i in ids})).encode()).hexdigest()
        fraction=int(hashlib.sha256(('42:'+gid).encode()).hexdigest()[:8],16)/2**32
        split='train' if fraction<.7 else 'val' if fraction<.85 else 'test'
        for i in ids:
            r=rows[i]; status='excluded_conflict' if r['conflict'] else 'excluded_duplicate' if r['duplicate'] else 'included'
            manifests.append({k:v for k,v in r.items() if k!='boxes'}|{'group':gid,'split':split,'status':status})
            if status!='included': continue
            dest=out/'images'/split; labels=out/'labels'/split; dest.mkdir(parents=True,exist_ok=True); labels.mkdir(parents=True,exist_ok=True)
            stem=hashlib.sha256(r['path'].encode()).hexdigest()[:20]
            shutil.copy2(r['path'],dest/(stem+'.jpg'))
            (labels/(stem+'.txt')).write_text('\n'.join(f'{int(b[0])} '+' '.join(f'{v:.8f}' for v in b[1:]) for b in r['boxes']),encoding='utf-8')
            split_counts[split]+=1; class_counts[split].update(NAMES[b[0]] for b in r['boxes'])
    cfg={'path':out.as_posix(),'train':'images/train','val':'images/val','test':'images/test','names':NAMES}
    (out/'data.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8')
    dump_csv(reports/'public_manifest.csv',manifests); dump_csv(reports/'label_issues.csv',issues)
    summary={'input_images':len(rows),'issues':len(issues),'groups':len(groups),'old_split_crossing_groups':cross,'split_images':dict(split_counts),'split_instances':{s:dict(c) for s,c in class_counts.items()},'excluded':dict(Counter(m['status'] for m in manifests)), 'group_limit':'Source-image names and pHash inferred groups; aircraft/flight identity unknown. Not a verified aircraft-held-out benchmark.','excluded_source':'Aircraft_AI_3cls_plus_external_scratch_v1: derived copy with external provenance unresolved','manifest_sha256':hashlib.sha256((reports/'public_manifest.csv').read_bytes()).hexdigest()}
    if any(class_counts[s][c]==0 for s in ('train','val','test') for c in NAMES): raise RuntimeError('Missing class in split; audit before training')
    assert all(len({m['split'] for m in manifests if m['group']==g})==1 for g in {m['group'] for m in manifests})
    (reports/'data_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
