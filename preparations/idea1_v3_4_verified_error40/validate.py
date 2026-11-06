import json
from types import SimpleNamespace
import numpy as np
import torch
from . import model as m
from .common import OUT
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4_E2_comp_error_cost.verify import check
from preparations.idea1_v3_4_E2_adjacency_200.common import choose

def main():
    proof=check()
    class Fake:
        graph=None
        def __init__(self,b):self.clear()
        def clear(self):self.graph=None
        def run(self,c,seed,coeff,*args):
            self.graph=True;k=args[-1]
            value=np.arange(17,dtype=np.float32) if len(args)==1 else args[0].numpy().mean((1,2))
            g=np.tile(value,(4,1));score=np.sqrt(np.mean(g*g,0))
            return choose(score,k,'spaced'),g,score,dict(selector_total_ms=0.,selector_capture_ms=0.,selector_hot_ms=0.)
    m.OriginalSelector=m.ErrorSelector=Fake
    b=SimpleNamespace(adapter=SimpleNamespace(measure=lambda fn:(fn(),0.)))
    owner=SimpleNamespace(history=[],step=0);sel=m.CompensatedSelector(b,owner);records=[]
    for index,step in enumerate([0,5,20,30]):
        current=dict(step=step,residuals={i:torch.full((2,2),float(step*step)) for i in range(1,18)})
        owner.step=step;owner.history=(owner.history+[current])[-2:]
        sel.run({},100,np.ones((5,7)),6);records.append(dict(sel.record))
        if index:assert torch.allclose(sel.error_buffer,torch.full_like(sel.error_buffer,[0,25,300,250][index]))
    assert [r['scoring_mode'] for r in records]==['warm_original','comp_error_hold','comp_error_extrapolate','comp_error_extrapolate']
    assert records[2]['score_source_full_steps']==[0,5,20] and records[2]['score_error_ratio']==3.
    assert records[3]['score_source_full_steps']==[5,20,30] and np.isclose(records[3]['score_error_ratio'],2/3)
    sel.clear();assert sel.history==[] and sel.error_buffer is None and sel.record is None
    weights=m.projection_weights(4);assert weights.shape==(4,5,7) and np.linalg.matrix_rank(weights.reshape(4,-1))==4
    atomic(OUT/'cpu_gate.json',dict(passed=True,error_model=proof,history_flow=records,reset=True,rank4=True,source='own Full histories; second Full hold error; later actual elapsed-step extrapolation'))
    print(json.dumps(dict(passed=True,history_modes=[r['scoring_mode'] for r in records])))

if __name__=='__main__':main()
