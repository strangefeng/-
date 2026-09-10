import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_degradation import transform_boxes

boxes=[[2,.3,.7,.2,.1],[0,.5,.5,1.,1.]]
np.testing.assert_allclose(transform_boxes(boxes,1,1),boxes)
result=transform_boxes(boxes,.5,.75)
np.testing.assert_allclose(result[0],[2,.4,.65,.1,.075])
for _,x,y,w,h in result:
    assert 0<=x-w/2<=x+w/2<=1 and 0<=y-h/2<=y+h/2<=1
print('Synthetic box geometry checks passed')
