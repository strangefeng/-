"""Create a validation-only synthetic stress set with transformed GT and lineage.

Foreshortening is an affine image-space proxy, NOT measured camera pose.
Scale reduction with neutral padding is NOT a calibrated distance simulation.
"""
from pathlib import Path
import csv, hashlib, itertools, json
import cv2, numpy as np
from PIL import Image
import yaml
ROOT=Path(__file__).resolve().parents[1]

def transform_boxes(boxes,sx,sy):
    return [[int(c),.5+(x-.5)*sx,.5+(y-.5)*sy,w*sx,h*sy] for c,x,y,w,h in boxes]

def main():
    src=ROOT/'data/public_v2'; out=ROOT/'data/degradation_val_v1'
    if out.exists(): raise RuntimeError('Use a new output version; do not overwrite lineage')
    with (ROOT/'reports/public_manifest.csv').open(encoding='utf-8-sig') as f:
        candidates=[r for r in csv.DictReader(f) if r['status']=='included' and r['split']=='val']
    unique={}
    for r in sorted(candidates,key=lambda r:hashlib.sha256(r['pixel_sha'].encode()).hexdigest()): unique.setdefault(r['group'],r)
    selected=list(unique.values())[:100]; records=[]
    for scale,foreshorten,sigma in itertools.product((1.,.75,.5),(1.,.866,.5),(0.,1.,2.)):
        condition=f's{scale}_f{foreshorten}_b{sigma}'
        image_dir=out/condition/'images'; label_dir=out/condition/'labels'; image_dir.mkdir(parents=True); label_dir.mkdir()
        for row in selected:
            stem=hashlib.sha256(row['path'].encode()).hexdigest()[:20]; p=src/'images/val'/f'{stem}.jpg'
            im=np.asarray(Image.open(p).convert('RGB')); h,w=im.shape[:2]; sx=scale*foreshorten; sy=scale
            affine=np.array([[sx,0,(1-sx)*w/2],[0,sy,(1-sy)*h/2]],np.float32)
            transformed=cv2.warpAffine(im,affine,(w,h),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT,borderValue=(114,114,114))
            if sigma: transformed=cv2.GaussianBlur(transformed,(0,0),sigma)
            Image.fromarray(transformed).save(image_dir/p.name,quality=95)
            boxes=[list(map(float,line.split())) for line in (src/'labels/val'/f'{stem}.txt').read_text().splitlines()]
            boxes=transform_boxes(boxes,sx,sy)
            (label_dir/f'{stem}.txt').write_text('\n'.join(f'{b[0]} '+' '.join(f'{x:.8f}' for x in b[1:]) for b in boxes),encoding='utf-8')
            records.append({'image':str((image_dir/p.name).relative_to(out)),'source_stem':stem,'source_group':row['group'],'source_split':'val','condition':condition,'synthetic':True,'scale_factor':scale,'horizontal_foreshortening':foreshorten,'blur_sigma_pixels':sigma,'physical_distance_m':'','physical_angle_deg':'','padding':'constant_114','transform':'centered_affine_image_space'})
        cfg={'path':(out/condition).as_posix(),'train':'images','val':'images','names':['missing-rivet','rust','scratch','crack','dent']}
        (out/condition/'eval.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    with (out/'manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    summary={'images':len(records),'source_groups':len(selected),'conditions':27,'use':'validation stress development only, not held-out test','simulation_limit':'Affine foreshortening and neutral-padded resizing; no camera calibration, true physical distances, reflection, IMU or VIO. JPEG recompression shared across conditions.'}
    (ROOT/'reports/degradation_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(summary)

if __name__=='__main__': main()
