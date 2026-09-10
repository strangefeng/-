"""Fixed-threshold, per-GT validation diagnostics. This is NOT COCO AP.

Greedy confidence-ordered, class-aware matching at IoU >= 0.5. FP kept separately.
GT box quality/size are retrospective explanatory variables, not deployment inputs.
"""
from pathlib import Path
import argparse, csv, json
from collections import defaultdict
import cv2, numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[1]

def match(pred, gt, threshold=.5):
    matched={}; used=set()
    for j in sorted(range(len(pred)),key=lambda j:-pred[j][4]):
        p=pred[j]; candidates=[]
        for k,g in enumerate(gt):
            if k in used or int(p[5])!=int(g[4]): continue
            lt=np.maximum(p[:2],g[:2]); rb=np.minimum(p[2:4],g[2:4]); wh=np.maximum(0,rb-lt); inter=float(wh.prod())
            union=float(np.prod(p[2:4]-p[:2])+np.prod(g[2:4]-g[:2])-inter)
            iou=inter/union if union>0 else 0
            candidates.append((iou,k))
        if candidates:
            iou,k=max(candidates)
            if iou>=threshold: matched[k]=(j,iou); used.add(k)
    return matched

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--weights',required=True); ap.add_argument('--imgsz',type=int,required=True); ap.add_argument('--conf',type=float,default=.25); ap.add_argument('--device',default='0'); args=ap.parse_args()
    model=YOLO(args.weights); root=ROOT/'data/public_v2'; rows=[]; fps=[]
    with (ROOT/'reports/public_manifest.csv').open(encoding='utf-8-sig') as f:
        import hashlib
        groups={hashlib.sha256(r['path'].encode()).hexdigest()[:20]:r['group'] for r in csv.DictReader(f)}
    for p in sorted((root/'images/val').glob('*.jpg')):
        im=np.asarray(Image.open(p).convert('RGB')); h,w=im.shape[:2]; gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
        gt=[]
        for line in (root/'labels/val'/f'{p.stem}.txt').read_text().splitlines():
            c,x,y,bw,bh=map(float,line.split()); gt.append([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h,c])
        gt=np.array(gt).reshape(-1,5)
        pred=model.predict(im[:,:,::-1].copy(),imgsz=args.imgsz,conf=args.conf,device=args.device,end2end=False,verbose=False)[0].boxes.data.cpu().numpy()
        matched=match(pred,gt); taken={v[0] for v in matched.values()}
        fps.extend({'image':p.name,'class':model.names[int(v[5])],'confidence':float(v[4])} for j,v in enumerate(pred) if j not in taken)
        for k,g in enumerate(gt):
            x1,y1,x2,y2,c=g; crop=gray[max(0,int(y1)):min(h,int(np.ceil(y2))),max(0,int(x1)):min(w,int(np.ceil(x2)))]
            area=(x2-x1)*(y2-y1); hit=matched.get(k)
            rows.append({'image':p.name,'group':groups[p.stem],'class':model.names[int(c)],'gt_index':k,'bbox_width':x2-x1,'bbox_height':y2-y1,'area_original_pixels':area,'size':'small' if area<32**2 else 'medium' if area<96**2 else 'large','detected':int(hit is not None),'confidence':float(pred[hit[0],4]) if hit else 0.,'iou':hit[1] if hit else 0.,'local_contrast':float(crop.std()),'local_brightness':float(crop.mean()),'highlight_proxy':float((crop>=250).mean()),'image_blur_laplacian_var':float(cv2.Laplacian(gray,cv2.CV_64F).var()),'distance_m':'','view_angle_deg':''})
    out=ROOT/'reports'/('failure_'+Path(args.weights).parents[1].name); out.mkdir(exist_ok=True)
    for name,records in [('per_gt',rows),('false_positives',fps)]:
        with (out/f'{name}.csv').open('w',newline='',encoding='utf-8-sig') as f:
            if records:
                wr=csv.DictWriter(f,fieldnames=list(records[0])); wr.writeheader(); wr.writerows(records)
    summary=[]
    for field in ['class','size']:
        buckets=defaultdict(list)
        for r in rows: buckets[r[field]].append(r['detected'])
        for key,vals in buckets.items(): summary.append({'field':field,'bucket':key,'GT':len(vals),'TP':sum(vals),'FN':len(vals)-sum(vals),'recall':sum(vals)/len(vals)})
    (out/'summary.json').write_text(json.dumps({'split':'val','confidence':args.conf,'iou':.5,'groups':summary,'note':'Retrospective GT-based diagnostics. Highlight fraction is a proxy, not verified reflection. No AP_small claim.'},indent=2),encoding='utf-8')

if __name__=='__main__': main()
