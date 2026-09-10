import sys
from pathlib import Path
import torch
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from otm_trainer import decode_one_to_many

def main():
    torch.set_num_threads(4)
    model=YOLO(str(ROOT/'weights/yolo26s.pt')).model.eval()
    head=model.model[-1]; x=torch.rand(1,3,128,128)
    with torch.no_grad():
        head.end2end=True
        raw=model(x)[1]['one2many']; decoded=decode_one_to_many(head,raw)
        assert head.end2end is True
        head.end2end=False; direct=model(x)[0]
    torch.testing.assert_close(decoded,direct,rtol=1e-5,atol=1e-5)
    print('OTM adapter equals direct OTM inference; dual-head state restored')

if __name__=='__main__': main()
