"""One independent 40-case condition for the fastest validated fused VJP."""
import json,os,socket,subprocess,time,traceback
from pathlib import Path
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v3_1.common import environment,lock
from preparations.idea1_v3_4.dispatch import render_environment
from preparations.idea1_v2_1.runtime_client import rpc
from preparations.idea1_v3_4_selection_async.common import ROOT,OUT,sha,verify
from preparations.idea1_v3_4_selection_async import controller


def freeze():
    path=OUT/'optimized_manifest.json'
    if path.exists():return json.loads(path.read_text())
    base=json.loads((OUT/'closed_loop_manifest.json').read_text());verify(base)
    probe=json.loads((OUT/'elementwise_batch_probe/complete.json').read_text())
    choices=[r for r in probe['summary'] if r['n']==40 and r['raw_passes']==8]
    assert choices,'No fused candidate preserves the original differentiable forward'
    best=min(choices,key=lambda r:(r['mean_ms'],r['p95_ms'],r['method']))
    template=next(c for c in base['conditions'] if c['id']=='v12_p3_q4_batch_async')
    cfg=dict(template,id='v12_p3_q4_optimized_async',gradient_backend=best['method'],n=40)
    m=dict(condition=cfg,cases=base['cases'],selected_probe=best,
        selection_rule='Minimum mean hot latency among completed eight-input candidates with original forward passing the fixed tolerance.',
        numerical_classification='Changed reverse rounding; gradients failed original serial tolerance. Separate scientific condition.',
        parent_manifest_hash=base['manifest_hash'],
        source_files={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'reports').glob('idea1_v3_4_selection_async*.py')},
        probe_sha256=sha(OUT/'elementwise_batch_probe/complete.json'),time=time.time())
    m['manifest_hash']=digest(m);atomic(path,m);return m


def main():
    wd=OUT/'optimized_worker';wd.mkdir(exist_ok=True)
    atomic(wd/'controller_process.json',dict(pid=os.getpid(),time=time.time()))
    while not (OUT/'elementwise_batch_probe/released.json').exists():time.sleep(3)
    assert (OUT/'elementwise_batch_probe/complete.json').exists(),'Bounded elementwise probe did not finish'
    manifest=freeze();controller.OUT=wd
    preferred=json.loads((OUT/'elementwise_batch_probe/launch.json').read_text())['gpu']
    chosen=None
    while chosen is None:
        for gpu,uuid in sorted(controller.available(),key=lambda x:(x[0]!=preferred,x[0])):
            lease=lock(Path('/tmp')/f'embodied-benchmark-gpu-{uuid}.lock',blocking=False)
            try:lease.__enter__()
            except BlockingIOError:continue
            if (gpu,uuid) not in controller.available():lease.__exit__(None,None,None);continue
            chosen=(gpu,uuid,lease);break
        if chosen is None:time.sleep(30)
    gpu,uuid,lease=chosen;service=gate=evaluation=None
    try:
        env=environment(gpu);env['PYTHONPATH']=str(ROOT/'policies/realtime-vla-flash/src')+':'+env['PYTHONPATH']
        render,mapping=render_environment(env,gpu)
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        atomic(wd/'identity.json',dict(gpu=gpu,uuid=uuid,port=port,egl_mapping=mapping,time=time.time()))
        with (wd/'service.log').open('a') as stream:
            service=subprocess.Popen([str(controller.MODEL),'-u','-m','reports.idea1_v3_4_selection_async_optimized_server',
                '--port',str(port),'--output',str(wd)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        atomic(wd/'service_process.json',dict(pid=service.pid,port=port,time=time.time()))
        start=time.time()
        while True:
            if service.poll() is not None:raise RuntimeError('Optimized service exited')
            try:rpc(port,dict(op='health'));break
            except (OSError,ConnectionError):
                if time.time()-start>600:raise TimeoutError('Optimized model startup')
                time.sleep(2)
        with (wd/'gate.log').open('a') as stream:
            gate=subprocess.Popen([str(controller.MODEL),'-u','-m','reports.idea1_v3_4_selection_async_optimized_gate',
                '--port',str(port),'--output',str(wd/'gate')],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        gate.wait(timeout=600)
        if gate.returncode:raise RuntimeError('Fused sync/async gate failed')
        with (wd/'evaluate.log').open('a') as stream:
            evaluation=subprocess.Popen([str(controller.EVAL),'-u','-m','reports.idea1_v3_4_selection_async_optimized_evaluate',
                '--port',str(port)],cwd=ROOT,env=render,stdout=stream,stderr=subprocess.STDOUT)
        atomic(wd/'evaluation_process.json',dict(pid=evaluation.pid,time=time.time()))
        while evaluation.poll() is None:time.sleep(3)
        if evaluation.returncode:raise RuntimeError('Fused closed-loop evaluation failed')
        rpc(port,dict(op='stop'));service.wait(timeout=30)
    finally:
        for process in (evaluation,gate,service):
            if process is not None and process.poll() is None:
                process.terminate()
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:process.kill();process.wait()
        lease.__exit__(None,None,None);atomic(wd/'exits.json',dict(released=True,time=time.time()))


if __name__=='__main__':
    try:main()
    except Exception:atomic(OUT/'optimized_failure.json',dict(error=traceback.format_exc(),time=time.time()));raise
