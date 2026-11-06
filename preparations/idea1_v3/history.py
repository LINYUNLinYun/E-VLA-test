"""E3 previous-round anchor, invertible decoder mapping and clean history prefix.

The deployed pi0_libero config has extra_delta_transform=True: decoded controller
commands include state-dependent re-anchoring. This module inverts the actual
loaded affine decoder at the previous anchor instead of concatenating raw tensors
from distinct states. Unused 25 padded model dimensions retain their model values.
"""
import time
import numpy as np

from preparations.idea1_v3.policy import Policy, TritonBackend, V14, acceptance, verification_seed


class DecoderCoordinates:
    """Finite affine inverse of the released output transform; fail closed on nonlinearity."""
    def __init__(self, transform, state):
        self.transform=transform; self.state=np.asarray(state).copy()
        probe=np.zeros((50,32),np.float32)
        self.offset=self.decode(probe)[0]
        self.matrix=np.empty((7,7),np.float32)
        for axis in range(7):
            value=probe.copy(); value[:,axis]=1
            self.matrix[axis]=self.decode(value)[0]-self.offset
        if np.linalg.cond(self.matrix)>1e6: raise ValueError('E3 decoder not stably invertible')
        self.inverse=np.linalg.inv(self.matrix)
        validation=np.linspace(-.7,.8,50*32,dtype=np.float32).reshape(50,32)
        expected=self.offset+validation[:,:7]@self.matrix
        if not np.allclose(self.decode(validation),expected,atol=2e-5,rtol=2e-5):
            raise ValueError('E3 decoder is not the expected affine LIBERO mapping')

    def decode(self,raw):
        return np.asarray(self.transform(dict(state=self.state.copy(),actions=np.asarray(raw).copy()))['actions'],np.float32)

    def encode_commands(self,commands,raw_template):
        commands=np.asarray(commands,np.float32)
        raw=np.asarray(raw_template,np.float32).copy()
        raw[:len(commands),:7]=(commands-self.offset)@self.inverse
        padded=np.zeros((50,32),np.float32); padded[:len(raw)]=raw
        if not np.allclose(self.decode(padded)[:len(commands)],commands,atol=2e-5,rtol=2e-5):
            raise ValueError('E3 command re-anchoring round-trip failed')
        return raw


def make_history_chunk(history_commands,history_raw,current_commands,current_raw,coordinates):
    r=len(history_commands)
    if not 1<=r<=5 or np.shape(current_raw)!=(50,32): raise ValueError('E3 requires actual r1..5 and H50')
    historical=coordinates.encode_commands(history_commands,history_raw[:r])
    future=coordinates.encode_commands(current_commands[:50-r],current_raw[:50-r])
    return np.concatenate([historical,future],axis=0),future,r


class HistoryBackend(TritonBackend):
    def verify_history(self,full,current,raw,seed,suite,previous):
        from scripts.spec.triton.pi0_spec_infer import FullCacheSnapshot
        coordinates=DecoderCoordinates(self.adapter.backend.output_transform,previous['condition']['transformed_state'])
        current_commands=self.decode(current,raw)
        raw_np=raw.float().cpu().numpy()
        chunk,future,r=make_history_chunk(previous['commands'],previous['raw'],current_commands,raw_np,coordinates)
        s=self.spec(current,suite)
        if not s._verify_exact_input_supported(weights=current['runtime'].weights):
            raise RuntimeError('E3 exact time/input weights unavailable')
        snapshot=FullCacheSnapshot(encoder_seq_len=full['seq'],encoder_k=full['K'],encoder_v=full['V'])
        chunk_t=self.torch.as_tensor(chunk,device='cuda',dtype=self.torch.float32)
        noise=self.torch.as_tensor(np.random.default_rng(verification_seed(seed)).standard_normal((50,32),dtype=np.float32),device='cuda')
        x_t,timestep=s._build_verify_batch(noise=noise,x0_draft=chunk_t,t_list=(.10,.05))
        x_t[:,:r].copy_(chunk_t[None,:r])
        def run():
            return s._run_batched_verify(full_runtime=current['runtime'],cache_snapshot=snapshot,
                observation_state_normalized=previous['condition']['state'],x_t_bk=x_t,timestep_bk=timestep)
        hats,ms=self.adapter.measure(run)
        hats=hats.reshape(2,50,32).float().cpu().numpy()
        hats[:,:r]=chunk[:r]  # Fixed historical constraints are never acceptance samples.
        return hats[:,r:r+5],future[:5],dict(verify_ae_ms=ms),dict(history_length=r,
            anchor_step=previous['step'],history_commands=previous['commands'].tolist(),
            history_normalized=chunk[:r].tolist(),future_reanchored=future[:5].tolist(),
            coordinate_definition='inverse actual affine output_transform at previous pre-execution state',
            clean_history_chunk_time_condition='original chunk-level t retained; mixed-noise heuristic')


class HistoryPolicy(Policy):
    def __init__(self,config,backend=None):
        if config.get('draft_steps',2)!=2: raise ValueError('E3 parent must be D2')
        parent=dict(config,experiment='E1')
        super().__init__(parent,backend or HistoryBackend())
        self.config=dict(config,experiment='E3')
        self.protocol.update(config=self.config)

    def reset(self):
        result=super().reset(); self.history=None; self.pending_condition=None
        return result

    def acknowledge(self,executed):
        pending=self.pending
        anchor=self.pending_condition
        previous_step=self.step
        result=super().acknowledge(executed)
        sent=np.asarray(executed,np.float32).copy()
        effective=sent.copy(); effective[:,:6]=np.clip(effective[:,:6],-1.,1.)
        self.history=dict(condition=anchor,commands=effective,sent_commands=sent,
            raw=np.asarray(pending['raw'])[:len(executed)].copy(),step=previous_step)
        self.pending_condition=None
        return result

    def predict(self,obs,instruction,suite,seed):
        if self.pending is not None: raise RuntimeError('Acknowledge previous executed commands first')
        tick=time.perf_counter(); timing={}; arrays={}; route={}; anchor=None
        age=None if self.full is None else dict(rounds=self.round-self.full['round'],env_steps=self.step-self.full['step'],wall_s=time.perf_counter()-self.full['time'])
        def full(reason):
            nonlocal anchor
            anchor,cost=self.backend.prepare(obs,instruction,())
            actions,raw,costs=self.backend.sample(anchor,seed,10)
            timing.update({f'full_{k}':v for k,v in dict(cost,**costs).items()})
            self.full=dict(anchor,round=self.round,step=self.step,time=time.perf_counter())
            self.draft_rounds=0; self.pending_gripper=False; self.counts['full']+=1
            route.update(path='full',full_reason=reason)
            return actions,raw,5
        if self.full is None: actions,raw,length=full('startup')
        elif self.pending_gripper: actions,raw,length=full('gripper_window')
        elif self.draft_rounds>=self.config.get('period',2): actions,raw,length=full('periodic')
        else:
            anchor,cost=self.backend.prepare(obs,instruction,V14)
            actions,raw,costs=self.backend.sample(anchor,seed,2)
            timing.update({f'draft_{k}':v for k,v in dict(cost,**costs).items()}); self.counts['draft']+=1
            if self.history:
                hats,comparison,costs,history=self.backend.verify_history(self.full,anchor,raw,seed,suite,self.history)
                route.update(history)
            else:
                hats,costs=self.backend.verify(self.full,anchor,raw,seed,suite)
                comparison=raw.float().cpu().numpy()
                route.update(history_length=0,anchor_step=self.step)
            timing.update(costs); self.counts['verify']+=1
            if not np.isfinite(hats).all(): raise ValueError('nonfinite E3 reconstruction')
            length,details=acceptance(comparison,hats,self.config.get('tau',.15),self.last_gripper)
            arrays.update(draft_future=comparison[:5],reconstructed_future=hats[:,:5])
            route.update(path='history_verified',accepted_length=length,**details); self.draft_rounds+=1
            switch=details['verify_gripper'] or details['post_verify_gripper']
            if length==0: actions,raw,length=full('gripper' if switch else 'zero_accept')
            elif switch: self.pending_gripper=True
        raw_np=raw.float().cpu().numpy() if hasattr(raw,'cpu') else np.asarray(raw)
        result=dict(actions=np.asarray(actions,np.float32),raw=raw_np,execute_length=min(length,5),route=route,
            timing=timing,arrays=arrays,counts=dict(self.counts),full_cache_age=age,seed=seed,
            verification_seed=verification_seed(seed),round=self.round,step=self.step)
        timing['policy_wall_ms']=(time.perf_counter()-tick)*1000
        self.pending=result; self.pending_condition=anchor; self.round+=1
        return result
