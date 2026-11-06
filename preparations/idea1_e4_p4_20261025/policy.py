"""Xiaomi-only timing/warmup wrapper; period_policy differs by one comparison."""
import hashlib
import time
from pathlib import Path
import numpy as np
from .period_policy import Policy as PeriodPolicy, XMASK


def array_hash(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


class Policy(PeriodPolicy):
    def __init__(self,backend):
        super().__init__(backend)
        self.protocol.update(experiment='idea1_e4_p4_20261025',policy_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cost_boundary='server predict plus entire ack wall; warmup separately timed')

    def configure(self,config):
        if config.get('_control')=='common_warm':return self.common_warm(config)
        if config.get('model')!='xiaomi' or config.get('refresh_period') not in [2,4]:raise ValueError('Xiaomi P2/P4 only')
        return super().configure(config)

    def common_warm(self,req):
        start=time.perf_counter();self.reset();b=self.backend
        obs,text,seed,suite=req['obs'],req['instruction'],req['seed'],req['suite']
        full,_=b.prepare(obs,text,());b.sample(full,seed,2)
        draft,_=b.prepare(obs,text,XMASK);_,raw,_=b.sample(draft,seed,2)
        b.verify(full,draft,raw,seed,suite)
        signatures=[]
        for _ in range(3):
            c,_=b.prepare(obs,text,());a,r,_=b.sample(c,seed,2)
            signatures.append(dict(raw_sha256=array_hash(r.float().cpu().numpy()),decoded_sha256=array_hash(a)))
        if signatures[-1]!=signatures[-2]:raise ValueError('Common Xiaomi warm did not stabilize Full-D2')
        return dict(common_warm_ms=(time.perf_counter()-start)*1000,canonical_signatures=signatures,canonical=signatures[-1],warmup_only=True,
            schedule='Full D2,skip12 D2,dual q=.10/.05 verify with native 1-q,3 Full D2; no Pi metadata/Kalman warmup')

    def predict(self,*args,**kwargs):
        start=time.perf_counter();before=self.draft_rounds
        r=super().predict(*args,**kwargs)
        r['refresh_state']=dict(period=self.config['refresh_period'],draft_rounds_before=before,draft_rounds_after=self.draft_rounds)
        r['timing']['policy_wall_ms']=(time.perf_counter()-start)*1000
        return r

    def acknowledge(self,*args,**kwargs):
        start=time.perf_counter();r=super().acknowledge(*args,**kwargs)
        r['ack_policy_wall_ms']=(time.perf_counter()-start)*1000
        return r
