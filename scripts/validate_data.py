from pathlib import Path
import csv, hashlib, json
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
with (ROOT/'reports/public_manifest.csv').open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
for key in ('group','pixel_sha'):
    assignments=defaultdict(set)
    for r in rows: assignments[r[key]].add(r['split'])
    assert all(len(v)==1 for v in assignments.values()),key
root=ROOT/'data/public_v2'
count=0
for r in rows:
    if r['status']!='included': continue
    stem=hashlib.sha256(r['path'].encode()).hexdigest()[:20]
    image=root/'images'/r['split']/(stem+'.jpg'); label=root/'labels'/r['split']/(stem+'.txt')
    assert image.exists() and label.exists()
    assert hashlib.sha256(image.read_bytes()).digest()==hashlib.sha256(Path(r['path']).read_bytes()).digest()
    for line in label.read_text().splitlines():
        c,x,y,w,h=map(float,line.split()); assert c in range(5) and w>0 and h>0
        assert min(x-w/2,y-h/2)>=-1e-5 and max(x+w/2,y+h/2)<=1.00001
    count+=1
result={'status':'passed','images_checked':count,'checks':['group_disjoint','pixel_hash_disjoint','copied_images_equal_raw','labels_exist_and_valid'],'limitation':'Does not establish semantic label correctness or unseen aircraft independence.'}
(ROOT/'reports/data_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result)
