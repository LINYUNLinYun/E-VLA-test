"""One AE with causal shifted future action priors, in current decoder coordinates."""
import hashlib
import time
import numpy as np
from preparations.idea1_v3.history import DecoderCoordinates
from preparations.idea1_v4_1_warmstart.sampling import sample_gaussian, sample_initial
from preparations.idea1_v4_1_warmstart.schedule import mix
from preparations.idea1_v4_3_kalman6.rules import KalmanPrior
from .common import MASK, conditions


def shift_prediction(commands,executed):
    commands=np.asarray(commands,np.float32)
    if commands.shape!=(50,7) or not np.isfinite(commands).all() or not 1<=executed<=5:
        raise ValueError('Finite H50 decoded prediction and actual r1..5 required')
    return np.concatenate((commands[executed:],np.repeat(commands[-1:],executed,axis=0)),axis=0)


class Policy:
    def __init__(self,backend,gaussian_sampler=sample_gaussian,initial_sampler=sample_initial,coordinates=DecoderCoordinates):
        self.b=backend
        self.gaussian_sampler,self.initial_sampler=gaussian_sampler,initial_sampler
        self.coordinates=coordinates
        self.config={}
        self.reset()

    def reset(self):
        self.pending=None
        self.future=None
        self.future_source_round=self.future_source_step=None
        self.shift_length=0
        self.round=self.step=0
        self.history=[]
        self.kalman=KalmanPrior() if self.config.get('initial')=='kalman' else None
        self.counts=dict(full=0,draft=0,verify=0,residual_refresh=0,kalman=0,shifted_prior=0)

    def configure(self,config):
        if config not in conditions():raise ValueError('Frozen tau.1 future-prior A/B conditions required')
        self.config=dict(config);self.reset();return {}

    def predict(self,obs,instruction,suite,seed):
        if self.pending is not None:raise RuntimeError('Previous execution must be acknowledged')
        start=time.perf_counter();refresh=self.round%3==0
        timing,arrays={},{}
        prior=None
        if self.future is not None:
            tick=time.perf_counter()
            if self.kalman is not None:
                assert self.kalman.filter.count==self.kalman.rotation_filter.count==self.step
                prior=self.kalman.forecast().copy()
                prior[:,6]=self.future[:,6]
                timing['kalman_rotation_forecast_ms']=self.kalman.rotation_forecast_ms
                timing['kalman_predict_ms']=(time.perf_counter()-tick)*1000
            else:prior=self.future.copy()
        if refresh:
            context,costs=self.b.prepare_residual(obs,instruction,capture=True)
            self.history.append(dict(step=self.step,residuals=context['residuals']));self.history=self.history[-2:]
            self.counts['residual_refresh']+=1
            timing.update({f'refresh_{k}':v for k,v in costs.items()});del context
        last=self.history[-1];previous=self.history[-2] if len(self.history)==2 else None
        age=self.step-last['step'];gap=None if previous is None else last['step']-previous['step']
        ratio=0. if gap is None else age/gap
        context,costs=self.b.prepare_residual(obs,instruction,MASK,history=last['residuals'],older=None if previous is None else previous['residuals'],ratio=ratio)
        timing.update(costs)
        initial_name,actual_start='gaussian',1.
        if prior is None:
            actions,raw,costs=self.gaussian_sampler(self.b,context,seed,1)
        else:
            tick=time.perf_counter()
            coordinates=self.coordinates(self.b.adapter.backend.output_transform,context['transformed_state'])
            encoded=coordinates.encode_commands(prior,np.zeros((50,32),np.float32))
            timing['kalman_encode_ms']=(time.perf_counter()-tick)*1000
            initial=mix(encoded,seed,self.config['start_time'],1.)
            arrays.update(prior_commands=prior.copy(),prior_command_prefix=prior[:5].copy(),prior_encoded_prefix=encoded[:5].copy(),prior_initial_prefix=initial[:5].copy(),decoder_offset=coordinates.offset.copy(),decoder_inverse=coordinates.inverse.copy())
            initial_name='kalman6' if self.kalman is not None else 'shifted_chunk'
            actual_start=self.config['start_time']
            actions,raw,costs=self.initial_sampler(self.b,context,initial,1,actual_start)
            self.counts['shifted_prior']+=1
            if self.kalman is not None:self.counts['kalman']+=1
        timing.update(costs)
        actions=np.asarray(actions,np.float32);raw=raw.float().cpu().numpy()
        if actions.shape!=(50,7) or raw.shape!=(50,32) or not np.isfinite(actions).all() or not np.isfinite(raw).all():raise ValueError('Invalid D1 action output')
        self.counts['draft']+=1
        timing['policy_wall_ms']=(time.perf_counter()-start)*1000
        result=dict(actions=actions,raw=raw,arrays=arrays,execute_length=5,round=self.round,step=self.step,seed=seed,timing=timing,counts=dict(self.counts),full_cache_age=age,
            route=dict(path='v12_d1',action_source='v12_d1',D=1,skip=list(MASK),compensation='extrapolate',residual_refresh=refresh,full_ae_executed=False,full_action_executed=False,verifier=False,
                history_steps=[h['step'] for h in self.history],has_second_history=previous is not None,extrapolation_ratio=ratio,
                initial_state=initial_name,actual_start_time=actual_start,prior_source='gaussian' if prior is None else self.config['prior_source'],prior_source_round=self.future_source_round,shift_source_step=self.future_source_step,shift_executed_length=self.shift_length,future_chunk_available=self.future is not None,
                gripper_prior_source='gaussian' if prior is None else 'previous_prediction_shifted',prior_weight=0. if prior is None else 1-actual_start,
                kalman_bootstrap=self.kalman is not None and prior is None,kalman_dimensions=6 if self.kalman is not None else 0,kalman_observations=0 if self.kalman is None else self.kalman.filter.count,kalman_rotation_observations=0 if self.kalman is None else self.kalman.rotation_filter.count,kalman_source_last_step=self.step-1 if self.kalman is not None and prior is not None else None,
                warmstart_time=actual_start,warmstart_noise_scale=1.,actual_noise_std=actual_start,kalman_prior_weight=1-actual_start if self.kalman is not None and prior is not None else 0.,
                motion_noise_std=actual_start,actual_gripper_noise_std=actual_start,gripper_denoising_span=actual_start,gripper_prior_weight=0. if prior is None else 1-actual_start,heterogeneous_noise=False,
                fast_time_embedding='openpi_linspace',native_encoder_graph=False,native_ae_graph=False,verify_graph=False),
            model_noise_sha256=hashlib.sha256(np.random.default_rng(seed).standard_normal((50,32),dtype=np.float32).tobytes()).hexdigest())
        self.pending=result;self.round+=1;return result

    def acknowledge(self,executed,quaternions=None):
        if self.pending is None:raise RuntimeError('No pending actions')
        executed=np.asarray(executed,np.float32)
        if executed.ndim!=2 or executed.shape[1]!=7 or not 0<len(executed)<=5 or not np.array_equal(executed,self.pending['actions'][:len(executed)]):raise ValueError('Actual actions differ from proposed prefix')
        tick=time.perf_counter()
        if self.kalman is not None:
            for command in executed:
                effective=command.copy();effective[:6]=np.clip(effective[:6],-1,1);self.kalman.observe(effective)
        self.future=shift_prediction(self.pending['actions'],len(executed))
        self.future_source_round=self.pending['round'];self.future_source_step=self.pending['step'];self.shift_length=len(executed)
        self.step+=len(executed);self.pending=None
        return dict(ack_policy_ms=(time.perf_counter()-tick)*1000)
