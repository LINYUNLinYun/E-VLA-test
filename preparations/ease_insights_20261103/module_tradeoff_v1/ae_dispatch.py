"""Scoped action-expert bypass around the unchanged eager sampler arithmetic.

No CUDA, torch, model or module import occurs here. A future backend supplies
its own process-local native helper namespace. A real numerical gate is still
required before any scientific use.
"""
from contextlib import contextmanager
from dataclasses import dataclass,field
import threading


def checked_skip(skip):
    value=tuple(skip)
    if (len(value)>6 or any(type(l) is not int or not 1<=l<=18 for l in value)
        or value!=tuple(sorted(set(value)))):
        raise ValueError('AE bypass requires zero to six unique sorted one-based layers')
    return value


@dataclass
class Visits:
    skip:tuple
    attempted:list=field(default_factory=list)
    executed:list=field(default_factory=list)
    bypassed:list=field(default_factory=list)
    original_restored:bool=False

    def receipt(self,actual_nfe):
        if type(actual_nfe) is not int or actual_nfe not in (2,10):
            raise ValueError('Only the existing actualD2 orD10 schedule is supported')
        expected=list(range(1,19))*actual_nfe
        kept=[l for l in range(1,19) if l not in self.skip]
        if (not self.original_restored or self.attempted!=expected or self.executed!=kept*actual_nfe
            or self.bypassed!=list(self.skip)*actual_nfe):
            raise RuntimeError('Native sampler did not visit the exact ordered AE layer schedule')
        return dict(actual_NFE=actual_nfe,attempted_layers_per_step=list(range(1,19)),
            executed_layers_per_step=kept,bypassed_layers_per_step=list(self.skip),
            actual_layer_calls=len(self.executed),attempted_layer_calls=len(self.attempted),
            bypassed_layer_calls=len(self.bypassed),original_helper_restored=True,
            scope='Observed eager helper dispatch, not FLOPs, latency or quality')


class ScopedLayers:
    def __init__(self,namespace):
        self.namespace=namespace;self.original=namespace._decoder_layer;self.lock=threading.Lock()

    @contextmanager
    def use(self,skip):
        skip=checked_skip(skip)
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('AE layer dispatch is single-request and non-reentrant')
        visits=Visits(skip)
        def wrapped(weights,buffers,seq,index):
            if type(index) is not int or not 0<=index<18:
                raise ValueError('Unexpected native AE layer index')
            layer=index+1;visits.attempted.append(layer)
            if layer in skip:
                visits.bypassed.append(layer)
                return None
            result=self.original(weights,buffers,seq,index)
            visits.executed.append(layer)
            return result
        installed=False
        try:
            if self.namespace._decoder_layer is not self.original:
                raise RuntimeError('Native helper changed before scoped dispatch')
            self.namespace._decoder_layer=wrapped;installed=True
            yield visits
        finally:
            try:
                if installed:
                    if self.namespace._decoder_layer is not wrapped:
                        raise RuntimeError('Native helper changed inside scoped dispatch')
                    self.namespace._decoder_layer=self.original
                    visits.original_restored=True
            finally:self.lock.release()


def sample_with_dispatch(adapter,dispatch,context,seed,steps,ae_skip=()):
    """Call the original sampler once; observe its real ordered helper calls."""
    skip=checked_skip(ae_skip)
    if type(steps) is not int or steps not in (2,10) or (steps==10 and skip):
        raise ValueError('FullD10 must retain allAE18; optional AE bypass applies only to FastD2')
    with dispatch.use(skip) as visits:
        result=adapter.sample(context,seed,dict(kind='native',m=None,num_steps=steps),False)
    if len(result)!=6 or result[3].get('actual_NFE')!=steps:
        raise RuntimeError('Original sampler return or actualNFE differs')
    return result,visits.receipt(result[3]['actual_NFE'])
