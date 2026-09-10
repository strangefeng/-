"""Generate separate machine proposals; never promote them to ground truth."""
from pathlib import Path
import argparse, csv, json
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--weights',required=True); ap.add_argument('--imgsz',type=int,required=True); args=ap.parse_args()
    data=ROOT/'data/custom_unannotated_v1'; out=data/'machine_proposals'
    if out.exists(): raise RuntimeError('Proposals already exist: preserve human review and use a new version')
    out.mkdir()
    with (data/'manifest.csv').open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    model=YOLO(args.weights); images=[]; annotations=[]; aid=1
    for i,r in enumerate(rows,1):
        if r['curation_status']!='retain_for_annotation_review': continue
        result=model.predict(str(data/'images'/r['image']),imgsz=args.imgsz,conf=.1,device=0,end2end=False,verbose=False)[0]
        images.append({'id':i,'file_name':r['image'],'width':int(r['width']),'height':int(r['height']),'annotation_status':'machine_proposals_need_expert_review','no_detections_is_negative':False})
        for x1,y1,x2,y2,conf,c in result.boxes.data.cpu().tolist():
            annotations.append({'id':aid,'image_id':i,'category_id':int(c),'bbox':[x1,y1,x2-x1,y2-y1],'area':(x2-x1)*(y2-y1),'iscrowd':0,'score':conf,'provenance':'model_prediction_not_ground_truth','review_status':'pending'}); aid+=1
    (out/'proposals.coco.json').write_text(json.dumps({'info':{'status':'NOT_GROUND_TRUTH','checkpoint':args.weights,'purpose':'Human review only. No training or evaluation on these labels until reviewed.'},'images':images,'annotations':annotations,'categories':[{'id':int(i),'name':n} for i,n in model.names.items()]},ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'{len(images)} images, {len(annotations)} unreviewed proposals')

if __name__=='__main__': main()
