import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from failure_analysis import match

def test_duplicate_prediction_and_class_mismatch():
    gt=np.array([[0,0,10,10,0],[20,20,30,30,1]])
    pred=np.array([[0,0,10,10,.9,0],[0,0,10,10,.8,0],[20,20,30,30,.7,0]])
    assert match(pred,gt)=={0:(0,1.)}

def test_empty_and_nonoverlap():
    gt=np.array([[0,0,10,10,0]])
    assert match(np.empty((0,6)),gt)=={}
    assert match(np.array([[20,20,30,30,.9,0]]),gt)=={}

if __name__=='__main__':
    test_duplicate_prediction_and_class_mismatch(); test_empty_and_nonoverlap(); print('matching checks passed')
