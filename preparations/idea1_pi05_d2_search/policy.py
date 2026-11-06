"""Native pi0.5 constructor with real sample_kwargs, not relabelled D10."""
import time
import numpy as np
from preparations.idea1_v3.pi05 import CHECKPOINT,Pi05Policy as ProtocolMethods
from preparations.idea1_v3_supplement.pi05 import install

REVISION='pi05-d2-search-native-scan-v1'


def validate_skip(skip):
    if not isinstance(skip,(list,tuple)) or len(skip)>16 or len(set(skip))!=len(skip) or any(type(i) is not int or not 3<=i<=18 for i in skip):
        raise ValueError('Explicit unique original layers3..18, count0..16; layers1/2 retained')
    return tuple(sorted(skip))


class Pi05Policy(ProtocolMethods):
    def __init__(self,config):
        from openpi.training import config as training
        from openpi.policies import policy_config
        self.config=dict(config); skip=validate_skip(config['skip']); steps=config['denoise_steps']
        if type(steps) is not int or steps not in (2,10): raise ValueError('D2 science or explicit D10 timing reference only')
        if steps==10 and not config.get('timing_reference'): raise ValueError('No new D10 quality runs')
        install(skip)
        train=training.get_config('pi05_libero')
        if not train.model.pi05 or train.model.action_horizon!=10: raise ValueError('Native pi05 H10 required')
        self.H=10
        self.policy=policy_config.create_trained_policy(train,CHECKPOINT,sample_kwargs=dict(num_steps=steps))
        if self.policy._sample_kwargs.get('num_steps')!=steps: raise RuntimeError('Native sampler kwargs mismatch')
        self.protocol=dict(model='pi05_libero',H=10,R=5,D=steps,denoise_steps=steps,
            sample_kwargs=dict(self.policy._sample_kwargs),denoise_nodes=[1-i/steps for i in range(steps+1)],
            denoise_evaluation_times=[1-i/steps for i in range(steps)],native_sampler='openpi.models.pi0.Pi0.sample_actions',
            backend='native OpenPI JAX',revision=REVISION,skip=list(skip),mask=config['mask'],depth=18,
            checkpoint=str(CHECKPOINT),normalization_asset=str(CHECKPOINT/'assets/physical-intelligence/libero/norm_stats.json'),
            ae_structure='real original native scanned suffix, all18 layers',
            mask_semantics='prefix hidden-update bypass, original-index KV shell; layers1/2 retained')
        self.reset()

    def predict(self,obs,instruction,suite,seed):
        if self.pending is not None: raise RuntimeError('Execution acknowledgement missing')
        from preparations.idea1_v2_1.runtime_models import native_observation
        images,state=native_observation(obs)
        data={'observation/image':np.asarray(images[0]),'observation/wrist_image':np.asarray(images[1]),'observation/state':state,'prompt':instruction}
        noise=np.random.default_rng(seed).standard_normal((10,32),dtype=np.float32)
        tick=time.perf_counter(); response=self.policy.infer(data,noise=noise)
        actions=np.asarray(response['actions'],np.float32)
        if actions.shape!=(10,7) or not np.isfinite(actions).all(): raise ValueError('Invalid native H10 actions')
        result=dict(actions=actions,execute_length=5,raw=np.empty((0,32),np.float32),arrays={},
            route=dict(path='pi05_native',mask=self.config['mask'],skip=self.protocol['skip'],denoise_steps=self.protocol['D'],denoise_nodes=self.protocol['denoise_nodes']),
            counts=dict(full=self.round+1,verify=0,draft=0),seed=seed,round=self.round,step=self.step,
            timing=dict(policy_wall_ms=(time.perf_counter()-tick)*1000,**response.get('policy_timing',{})))
        self.pending=result; self.round+=1
        return result
