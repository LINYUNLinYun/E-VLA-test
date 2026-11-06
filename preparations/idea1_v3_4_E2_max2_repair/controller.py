import concurrent.futures,json,os,subprocess,time,traceback
from .common import ROOT,OUT
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import environment
from preparations.idea1_v3_4.dispatch import render_environment
from preparations.idea1_v2_1.runtime_client import rpc
MODEL=ROOT/'policies/realtime-vla-flash/.venv/bin/python'
EVAL=ROOT/'policies/libero-eval-v21/.venv/bin/python'
def worker(gpu):
    wd=OUT/'workers'/f'gpu{gpu}';wd.mkdir(parents=True,exist_ok=True);port=20810+gpu
    env=environment(gpu);env['PYTHONPATH']=str(ROOT/'policies/realtime-vla-flash/src')+':'+env['PYTHONPATH'];render,_=render_environment(env,gpu)
    server=None
    try:
        with (wd/'service.log').open('w') as log:
            server=subprocess.Popen([str(MODEL),'-u','-m','preparations.idea1_v3_4_E2_max2_repair.server','--port',str(port),'--output',str(wd)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            t=time.monotonic()
            while True:
                if server.poll() is not None:raise RuntimeError('Server exited')
                if time.monotonic()-t>600:raise TimeoutError()
                try:rpc(port,dict(op='health'));break
                except OSError:time.sleep(1)
            with (wd/'evaluate.log').open('w') as f:
                code=subprocess.call([str(EVAL),'-u','-m','preparations.idea1_v3_4_E2_max2_repair.evaluate','--port',str(port),'--shard',str(gpu)],cwd=ROOT,env=render,stdout=f,stderr=subprocess.STDOUT)
            if code:raise RuntimeError(f'Evaluator exited {code}')
    finally:
        if server is not None and server.poll() is None:server.terminate();server.wait(timeout=30)
def main():
    atomic(OUT/'process.json',dict(pid=os.getpid(),time=time.time()))
    with concurrent.futures.ThreadPoolExecutor(4) as ex:
        fs=[ex.submit(worker,g) for g in range(4)]
        while not all(f.done() for f in fs):
            atomic(OUT/'status.json',dict(status='repair_running',completed=len(list(OUT.glob('*/episodes/*/task*/init*/complete.json'))),target=32,time=time.time()));time.sleep(5)
        for f in fs:f.result()
    from .finish import main as finish
    finish()
if __name__=='__main__':
    try:main()
    except Exception:
        atomic(OUT/'failed.json',dict(error=traceback.format_exc(),time=time.time()));raise
