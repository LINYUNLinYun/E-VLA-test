"""Experimental selector backend; native Full/Fast/verifier stay unchanged."""
import os
from pathlib import Path
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4_selection_async import core,policy as original_policy
from preparations.idea1_v3_4_E2_E4_single import torch_model as tm
from reports import idea1_v3_4_selection_async_batched_attention as direct
from reports import idea1_v3_4_selection_async_rms_backward as rms
from reports import idea1_v3_4_selection_async_elementwise as element

ORIGINAL_ATTENTION=core.attention
ORIGINAL_RMS=tm.rms
ACTIVE='original'


class Graph(core.Graph):
    def __init__(self,backend,context,seed,q=4,mode='original',batch=True):
        element.restore();tm.rms=ORIGINAL_RMS;core.attention=ORIGINAL_ATTENTION
        if ACTIVE!='original':
            assert q==4 and batch and mode=='original'
            direct.install(False);rms.install()
            if ACTIVE!='rms':element.install(glu=ACTIVE=='rms_rope_glu')
        super().__init__(backend,context,seed,q,mode,batch)


original_policy.Graph=Graph


class Policy(original_policy.Policy):
    def configure(self,config):
        global ACTIVE
        selected=config.get('gradient_backend','original')
        assert selected in ('original','rms','rms_rope','rms_rope_glu')
        if ACTIVE!=selected:self.clear()
        ACTIVE=selected
        return super().configure(config)


def main():
    from preparations.idea1_v3_4_selection_async import server
    server.Policy=Policy;old_setup=server.setup
    def setup(out):
        backend=old_setup(out)
        from reports.idea1_v3_4_selection_async_gpu_units import run as unit_checks
        unit_checks(Path(out)/'gpu_operator_checks.json')
        backend.protocol['experimental_selector_backends']=['rms','rms_rope','rms_rope_glu']
        atomic(Path(out)/'optimized_backend_capability.json',dict(backends=backend.protocol['experimental_selector_backends'],
            activation='Only config.gradient_backend activates the experimental branch'))
        return backend
    server.setup=setup;server.main()


if __name__=='__main__':main()
