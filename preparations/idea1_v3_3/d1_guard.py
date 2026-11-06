"""Restore the stage gate after bounded isolated native D1 verification."""
import json
import subprocess
import time
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import environment
from .dispatch import OUT,ROOT,PY
from .campaign import retired_gpu


def main():
    gate=OUT/'cost_selection_gate.json';d=json.loads(gate.read_text());d.update(pi0_screen_ready=False,D1_native_probe_pending=True);atomic(gate,d)
    child=None
    try:
        retired_gpu(0,9340)
        env=environment(0);env['PYTHONPATH']=str(ROOT/'policies/realtime-vla-flash/src')+':'+env['PYTHONPATH']
        with (OUT/'D1_native_probe.log').open('a') as log:
            child=subprocess.Popen([str(PY['pi0']),'-m','preparations.idea1_v3_3.d1_native_probe'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            child.wait(timeout=180)
            if child.returncode:raise RuntimeError('Native D1 probe failed; isolated log retained')
        retired_gpu(0,9340)
    except Exception as exc:
        atomic(OUT/'D1_native_probe_error.json',dict(time=time.time(),error=repr(exc)))
        if child is not None and child.poll() is None:child.terminate();child.wait(timeout=30)
        raise
    finally:
        d=json.loads(gate.read_text());d.update(pi0_screen_ready=True,D1_native_probe_pending=False);atomic(gate,d)


if __name__=='__main__':main()
