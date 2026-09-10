"""Extract real captured stills, retaining source membership and annotation state."""
from pathlib import Path
import csv, hashlib, io, json, zipfile
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageDraw
import pillow_heif
pillow_heif.register_heif_opener()
ROOT=Path(__file__).resolve().parents[1]

def main():
    src=ROOT.parent/'自采集数据集'; out=ROOT/'data'/'custom_unannotated_v1'; out.mkdir(parents=True,exist_ok=True)
    images=out/'images'; images.mkdir(exist_ok=True); rows=[]; seen={}; failures=[]
    for p in sorted(src.rglob('*')):
        if not p.is_file(): continue
        try:
            members=[('',p.read_bytes())] if p.suffix.lower() in ('.jpg','.jpeg','.heic','.png') else []
            if p.suffix.lower()=='.livp':
                with zipfile.ZipFile(p) as z:
                    members=[(n,z.read(n)) for n in z.namelist() if Path(n).suffix.lower() in ('.heic','.jpg','.jpeg')]
            for member,data in members:
                im=ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert('RGB'); a=np.array(im)
                sha=hashlib.sha256(a.tobytes()+str(im.size).encode()).hexdigest(); name=sha[:20]+'.jpg'
                duplicate=sha in seen
                if not duplicate: im.save(images/name,quality=95); seen[sha]=name
                gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
                rows.append({'image':name,'source':str(p.resolve()),'member':member,'pixel_sha256':sha,'duplicate':duplicate,'width':im.width,'height':im.height,'aircraft_id':'unassigned_glider_or_transport','group':'capture_session_unknown_01','split':'unassigned','annotation_status':'unannotated','blur_laplacian_var':float(cv2.Laplacian(gray,cv2.CV_64F).var()),'brightness':float(gray.mean()),'contrast':float(gray.std()),'saturation_fraction':float((gray>=250).mean()),'distance_m':'','view_angle_deg':'','imu_file':'','pose_provenance':'unavailable'})
        except Exception as e: failures.append({'source':str(p),'error':str(e)})
    with (out/'manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # COCO-compatible labeling starter: no annotations means UNANNOTATED, never negatives.
    unique=list(seen.values())
    (out/'annotation_tasks.json').write_text(json.dumps({'status':'UNANNOTATED_DO_NOT_EVALUATE','images':[{'id':i+1,'file_name':r['image'],'width':r['width'],'height':r['height']} for i,r in enumerate([r for r in rows if not r['duplicate']])],'categories':[{'id':i,'name':n} for i,n in enumerate(['missing-rivet','rust','scratch','crack','dent'])],'annotations':[]},indent=2),encoding='utf-8')
    for start in range(0,len(unique),48):
        sheet=Image.new('RGB',(1200,8*135),'white'); draw=ImageDraw.Draw(sheet)
        for j,name in enumerate(unique[start:start+48]):
            im=Image.open(images/name); im.thumbnail((196,110)); x=j%6*200; y=j//6*135; sheet.paste(im,(x,y)); draw.text((x,y+112),f'{start+j+1}: {name[:12]}',fill='black')
        sheet.save(out/f'contact_{start//48+1:02d}.jpg')
    summary={'input_records':len(rows),'unique_stills':len(unique),'duplicates':len(rows)-len(unique),'failures':failures,'annotation_status':'unannotated','pose':'not supplied; no fabricated measurements'}
    (ROOT/'reports'/'custom_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
