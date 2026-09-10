"""Sequential GPU campaign. Select on validation only; never open test images.

Stage E01 screens scales at 800; E02 screens resolution; B0 retrains the winner.
P2 is a control, not a proposed innovation. Run failure analysis before M1/M2.
"""
from pathlib import Path
import argparse, csv, gc, hashlib, json, subprocess, sys, time, traceback
import torch, ultralytics
from ultralytics import YOLO
from otm_trainer import OneToManyTrainer

ROOT=Path(__file__).resolve().parents[1]

def save(path,obj):
    path.parent.mkdir(exist_ok=True,parents=True)
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8'); tmp.replace(path)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--epochs',type=int,default=60); ap.add_argument('--final-epochs',type=int,default=120); ap.add_argument('--batch',type=int,default=1); args=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable: use project GPU environment')
    torch.set_num_threads(4)
    manifest=ROOT/'reports/public_manifest.csv'; fingerprint=hashlib.sha256(manifest.read_bytes()).hexdigest()
    data=ROOT/'data/public_v2/data.yaml'
    if ultralytics.__version__!='8.4.22': raise RuntimeError('OTM validator adapter requires ultralytics 8.4.22')
    env={'python':sys.version,'torch':torch.__version__,'ultralytics':ultralytics.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'manifest_sha256':fingerprint,'head':'one-to-many + NMS, train-time adapter v1','epochs':args.epochs,'final_epochs':args.final_epochs,'batch':args.batch}
    save(ROOT/'reports/environment.json',env)
    def run(name,scale,size,epochs,seed=42,p2=False,smoke=False):
        spec={**env,'scale':scale,'imgsz':size,'epochs_run':epochs,'seed':seed,'p2':p2,'smoke':smoke}
        resultfile=ROOT/'reports'/f'{name}.json'
        if resultfile.exists():
            old=json.loads(resultfile.read_text(encoding='utf-8'))
            if old['spec']!=spec: raise RuntimeError(f'{name}: configuration changed; choose a new run name')
            return old
        save(ROOT/'reports/campaign_status.json',{'status':'running','experiment':name,'started':time.time(),'spec':spec})
        weights=ROOT/'weights'; weights.mkdir(exist_ok=True)
        checkpoint=weights/f'yolo26{scale}.pt'
        if not checkpoint.exists():
            temporary=checkpoint.with_suffix('.part')
            subprocess.run(['curl.exe','-L','--fail','--retry','3','--connect-timeout','20','--max-time','1200','--output',str(temporary),f'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26{scale}.pt'],check=True)
            temporary.replace(checkpoint)
        model=YOLO(f'yolo26{scale}-p2.yaml').load(str(checkpoint)) if p2 else YOLO(str(checkpoint))
        runpath=ROOT/'runs'/name
        last=runpath/'weights/last.pt'
        if last.exists() and not smoke:
            model=YOLO(str(last)); model.train(trainer=OneToManyTrainer,resume=True)
        else:
            model.train(trainer=OneToManyTrainer,data=str(data),epochs=epochs,imgsz=size,batch=args.batch,nbs=64,device=0,workers=0,seed=seed,deterministic=True,optimizer='SGD',lr0=.01,lrf=.01,momentum=.937,weight_decay=.0005,warmup_epochs=3,patience=0,amp=True,cache=False,project=str(ROOT/'runs'),name=name,exist_ok=False,plots=False,save=True,val=True,end2end=False,close_mosaic=10 if epochs>=10 else 0,fraction=.01 if smoke else 1.)
        best=runpath/'weights/best.pt'
        if not best.exists(): raise RuntimeError('Training ended without best checkpoint')
        model=YOLO(str(best))
        metrics=model.val(data=str(data),split='val',imgsz=size,batch=args.batch,device=0,workers=0,end2end=False,plots=False,project=str(ROOT/'runs'),name=name+'_val',exist_ok=True)
        result={'name':name,'spec':spec,'checkpoint':str(best),'checkpoint_sha256':hashlib.sha256(best.read_bytes()).hexdigest(),'metrics':{k:float(v) for k,v in metrics.results_dict.items()},'per_class_ap50_95':dict(zip(model.names.values(),map(float,metrics.box.maps))),'completed':time.time()}
        save(resultfile,result); del model,metrics; gc.collect(); torch.cuda.empty_cache(); return result
    if args.smoke:
        run('smoke_otm_s_800','s',800,1,smoke=True); save(ROOT/'reports/smoke_status.json',{'status':'complete','claim':'execution check only; not a paper result'}); return
    candidates=[run(f'E01_{s}_800',s,800,args.epochs) for s in 'smlx']
    key=lambda r:r['metrics']['metrics/mAP50-95(B)']
    winner=max(candidates,key=key); scale=winner['spec']['scale']
    resolutions=[winner]+[run(f'E02_{scale}_{sz}',scale,sz,args.epochs) for sz in (640,1024,1280)]
    selected=max(resolutions,key=key); size=selected['spec']['imgsz']
    save(ROOT/'reports/baseline_selection.json',{'scale':scale,'imgsz':size,'selection_split':'val','selection_metric':'mAP50-95','scale_candidates':[r['name'] for r in candidates],'resolution_candidates':[r['name'] for r in resolutions],'screening_epochs':args.epochs,'note':'Sequential scale/resolution screening, not exhaustive joint search'})
    for seed in (42,3407,2026):
        run(f'B0_{scale}_{size}_{seed}',scale,size,args.final_epochs,seed)
    if not (ROOT/'data/custom_unannotated_v1/machine_proposals').exists():
        subprocess.run([sys.executable,str(ROOT/'scripts/prelabel_custom.py'),'--weights',str(ROOT/'runs'/f'B0_{scale}_{size}_42'/'weights/best.pt'),'--imgsz',str(size)],check=True)
    run(f'P2_{scale}_{size}_42',scale,size,args.final_epochs,p2=True)
    for expname in (f'B0_{scale}_{size}_42',f'P2_{scale}_{size}_42'):
        subprocess.run([sys.executable,str(ROOT/'scripts/failure_analysis.py'),'--weights',str(ROOT/'runs'/expname/'weights/best.pt'),'--imgsz',str(size)],check=True)
    save(ROOT/'reports/campaign_status.json',{'status':'baseline_and_p2_complete','next':'Validation failure analysis, then define M1; test remains unopened. No improvement claim yet.'})

if __name__=='__main__':
    try: main()
    except Exception:
        save(ROOT/'reports/campaign_status.json',{'status':'failed','traceback':traceback.format_exc(),'time':time.time()}); raise
