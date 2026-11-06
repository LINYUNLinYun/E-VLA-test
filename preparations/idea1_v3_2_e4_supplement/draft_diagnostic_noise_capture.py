"""Record official corruption input without changing RNG calls or observation sampling."""
from contextlib import contextmanager
import hashlib
import inspect
import sys
from pathlib import Path
import numpy as np


def image_hash(value):
    array=np.ascontiguousarray(value)
    return dict(sha256=hashlib.sha256(array.tobytes()).hexdigest(),shape=list(array.shape),dtype=str(array.dtype))


def operator_for(noise):
    if not 1<=noise<=50:raise ValueError('Official noise ID1..50 required')
    return ('motion_blur','gaussian_blur','zoom_blur','fog','glass_blur')[(noise-1)//10],(noise-1)%10+1


class Capture:
    def __init__(self,noise):
        self.noise=int(noise);self.operator,self.severity=operator_for(self.noise)
        self.calls=[];self.clean=None;self.output=None;self.source={}

    def metadata(self):
        return dict(noise_id=self.noise,operator=self.operator,severity=self.severity,call_count=len(self.calls),
            calls=self.calls,operator_source=self.source,
            clean_agentview=image_hash(self.clean),last_operator_output=image_hash(self.output),
            scope='Only copies existing PIL operator input, invokes original once with original arguments. No extra random calls, physics steps, renders or observation updates.')


@contextmanager
def install_capture(control_env,wrapper=None):
    capture=Capture(control_env.noise)
    if wrapper is None:wrapper=sys.modules[type(control_env).__module__]
    original=getattr(wrapper,capture.operator)
    path=Path(inspect.getsourcefile(original))
    capture.source=dict(path=str(path),file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        function_sha256=hashlib.sha256(inspect.getsource(original).encode()).hexdigest())
    def record(image,*args,**kwargs):
        clean=np.asarray(image).copy()
        result=original(image,*args,**kwargs)
        capture.clean=clean;capture.output=np.asarray(result).copy()
        severity=kwargs.get('severity',args[0] if args else 1)
        capture.calls.append(dict(index=len(capture.calls),severity=int(severity),input=image_hash(clean),output=image_hash(capture.output)))
        return result
    setattr(wrapper,capture.operator,record)
    try:yield capture
    finally:setattr(wrapper,capture.operator,original)


def clean_counterfactual(env):
    capture=env._diagnostic_noise_capture
    if capture.clean is None or len(capture.calls)<5:raise RuntimeError('Final settle operator input was not captured')
    actual=env._diagnostic_actual_observation
    if capture.clean.shape!=actual['agentview_image'].shape or capture.clean.dtype!=actual['agentview_image'].dtype:
        raise RuntimeError('Captured clean image contract mismatch')
    counter={key:np.asarray(value).copy() for key,value in actual.items()}
    counter['agentview_image']=capture.clean.copy();env.obs=counter
    return counter


def captured_environment(base):
    class Captured(base):
        def reset(self):
            if self.env.noise:
                with install_capture(self.env) as capture:
                    result=super().reset()
                self._diagnostic_noise_capture=capture
            else:result=super().reset()
            self._diagnostic_actual_observation={key:np.asarray(value).copy() for key,value in self.obs.items() if isinstance(value,np.ndarray)}
            return result
    return Captured
