"""Versioned native CALVIN server with a final idle guard before model CUDA move.

Does not alter frozen v3 modules or any installed dependency. No GPU operation
occurs on import; all normal server launch/manifest/quality gates still apply.
"""
from contextlib import contextmanager


@contextmanager
def final_allocation_guard():
    import torch
    from preparations.idea1_v3_2.residual import require_idle_gpu
    original=torch.nn.Module.to
    receipt={"checked":False}
    def checked(module,*args,**kwargs):
        destination=kwargs.get("device")
        if destination is None and args and isinstance(args[0],(str,torch.device)):
            destination=args[0]
        if destination is not None and torch.device(destination).type=="cuda" and not receipt["checked"]:
            receipt.update(checked=True,module_class=type(module).__name__,gpu_preflight=require_idle_gpu())
        return original(module,*args,**kwargs)
    torch.nn.Module.to=checked
    try:yield receipt
    finally:torch.nn.Module.to=original


def main():
    from . import policy_server
    from .policy_native import NativeCalvinPolicy
    class SafeNativeCalvinPolicy(NativeCalvinPolicy):
        def __init__(self,*args,**kwargs):
            with final_allocation_guard() as guard:
                super().__init__(*args,**kwargs)
            if not guard["checked"]: raise RuntimeError("Native model CUDA allocation guard did not execute")
            self.protocol["final_model_cuda_allocation_guard"]=guard
    policy_server.NativeCalvinPolicy=SafeNativeCalvinPolicy
    policy_server.main()


if __name__=="__main__":main()
