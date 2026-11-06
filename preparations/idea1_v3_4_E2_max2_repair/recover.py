"""Execution-only recovery of case0 after GPU0 was occupied during loading."""
import json,subprocess,time,traceback
from .common import ROOT,OUT,sha
from .controller import MODEL,EVAL
from preparations.idea1_v3_1.common import environment
from preparations.idea1_v3_4.dispatch import render_environment
from preparations.idea1_v2_1.runtime_client import rpc
from preparations.idea1_v3.io import atomic
def main():
    atomic(OUT/'execution_recovery.json',dict(reason='external GPU0 occupancy caused startup OOM; 0 valid episodes',
        scientific_changes=False,physical_gpu=1,case_shard=0,source_sha256=sha(__file__),time=time.time()))
    while True:
        apps=subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if not apps:break
        time.sleep(10)
    wd=OUT/'workers/recovery_gpu1_case0';wd.mkdir(parents=True,exist_ok=True);env=environment(1);env['PYTHONPATH']=str(ROOT/'policies/realtime-vla-flash/src')+':'+env['PYTHONPATH'];render,_=render_environment(env,1);port=20821
    server=None
    try:
        with (wd/'service.log').open('w') as f:
            server=subprocess.Popen([str(MODEL),'-u','-m','preparations.idea1_v3_4_E2_max2_repair.server','--port',str(port),'--output',str(wd)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            t=time.monotonic()
            while True:
                if server.poll() is not None:raise RuntimeError('Recovery model exited')
                if time.monotonic()-t>600:raise TimeoutError()
                try:rpc(port,dict(op='health'));break
                except OSError:time.sleep(1)
            with (wd/'evaluate.log').open('w') as log:
                subprocess.run([str(EVAL),'-u','-m','preparations.idea1_v3_4_E2_max2_repair.evaluate','--port',str(port),'--shard','0'],cwd=ROOT,env=render,stdout=log,stderr=subprocess.STDOUT,check=True)
        # The failed original scheduler is allowed to collect its other workers.
        for _ in range(60):
            if (OUT/'failed.json').exists():break
            time.sleep(1)
        from .finish import main as finish
        finish()
    finally:
        if server is not None and server.poll() is None:server.terminate();server.wait(timeout=30)
if __name__=='__main__':
    try:main()
    except Exception:
        atomic(OUT/'recovery_failed.json',dict(error=traceback.format_exc(),time=time.time()));raise
