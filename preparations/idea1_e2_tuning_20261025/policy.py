"""Routing-only subclass of the frozen v3.3 policy; filter math unchanged."""
import hashlib
import time
from pathlib import Path
import numpy as np
from preparations.idea1_v3_3 import policy as original
from preparations.idea1_v3_3.metadata_r2 import Policy as MetadataPolicy
from preparations.idea1_v3_3.policy import V14
from .features import decide,metrics,feature_after_observe


def array_hash(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


class Policy(MetadataPolicy):
    def __init__(self,backend):
        super().__init__(backend)
        self.protocol.update(experiment='idea1_e2_tuning_20261025',policy_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            cost_boundary='server predict wall + entire server acknowledge wall; common initial-input warmup separately timed',Kalman_implementation='Unmodified scripts.idea1_v2_1.filter.CommandKalman')

    def reset(self):
        out=super().reset();self.feature_history=[];self.last_decision=None
        return out

    def configure(self,config):
        if config.get('_control')=='common_warm':return self.common_warm(config)
        return super().configure(config)

    def common_warm(self,req):
        start=time.perf_counter();self.reset();b=self.backend;obs=req['obs'];text=req['instruction'];seed=req['seed'];suite=req['suite']
        full,_=b.prepare(obs,text,());_,_,_=b.sample(full,seed,10)
        draft,_=b.prepare(obs,text,V14);actions,raw,_=b.sample(draft,seed,2)
        b.verify(full,draft,raw,seed,suite)
        meta=b.prepare_metadata(obs,text);encoded=b.encode_commands(meta,actions)
        b.verify(full,meta,encoded,seed,suite)
        signatures=[]
        for _ in range(3):
            c,_=b.prepare(obs,text,());a,r,_=b.sample(c,seed,10)
            signatures.append(dict(raw_sha256=array_hash(r.float().cpu().numpy()),decoded_sha256=array_hash(a)))
        if signatures[-1]!=signatures[-2]:raise ValueError('Fixed common warmup did not yield two bitwise-identical canonical Full outputs')
        # Reuse v3.3 padding diagnosis; identical encode/verify shapes were
        # warmed above. No extra diagnostic calls inside measured episodes.
        self.padding_diagnosed=True
        return dict(common_warm_ms=(time.perf_counter()-start)*1000,canonical_signatures=signatures,
            canonical=signatures[-1],padding='same inverse decoder and zeros25 as frozen v3.3',warmup_only=True)

    def _gate(self,commands,quats,*unused):
        allowed,blocked,m=decide(self.config,commands,quats,self.feature_history)
        self.last_decision=dict(metrics=m,common_and_candidate_allowed=allowed,blocked=blocked)
        return allowed,blocked

    def predict(self,*args,**kwargs):
        start=time.perf_counter();self.last_decision=None;previous=original.gate;original.gate=self._gate
        try:r=super().predict(*args,**kwargs)
        finally:original.gate=previous
        r['route']['tuning_features']=self.last_decision or dict(not_a_gate_decision=True,runtime_new_metric_collection=self.config.get('kind')=='metric')
        r['timing']['frozen_predict_wall_ms']=r['timing']['policy_wall_ms']
        r['timing']['policy_wall_ms']=(time.perf_counter()-start)*1000-r['timing'].get('diagnostic_excluded_ms',0)
        return r

    def acknowledge(self,*args,**kwargs):
        start=time.perf_counter();observe=self.kalman.observe
        if self.config.get('kind')!='metric':
            result=super().acknowledge(*args,**kwargs);result['ack_policy_wall_ms']=(time.perf_counter()-start)*1000
            return result
        def capture(action):
            prior=self.kalman.H@self.kalman.F_forecast@self.kalman.x
            observe(action)
            error=np.clip(np.asarray(action,float)[:3],-1,1)-prior
            self.feature_history.append(dict(forecast_error_rms=float(np.linalg.norm(error)/np.sqrt(3))))
            self.feature_history=self.feature_history[-5:]
        self.kalman.observe=capture
        try:result=super().acknowledge(*args,**kwargs)
        finally:self.kalman.observe=observe
        result['ack_policy_wall_ms']=(time.perf_counter()-start)*1000
        return result
