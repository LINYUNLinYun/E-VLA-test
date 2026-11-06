"""Finish CPU-only audit/report after all scientific workers and supervisor exit."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
from preparations.idea1_v3.io import atomic
from .dispatch import OUT,ROOT,PY
from .campaign import alive,retired_gpu


def main():
    status=OUT/'finalization_status.json';start=time.time()
    while True:
        state=json.loads((OUT/'campaign_state.json').read_text())
        if (OUT/'campaign_alert.json').exists():raise RuntimeError('Scientific supervisor alert; diagnose before finalization')
        if state['phase']=='scientific_complete' and not alive(99382):break
        if time.time()-start>10800:raise TimeoutError('Bounded confirmation wait exceeded3hours')
        atomic(status,dict(time=time.time(),state='waiting_science_and_final_supervisor_analysis',phase=state['phase'],science_valid=state['complete']))
        time.sleep(30)
    devices={}
    for gpu in (0,2,3):devices[str(gpu)]=retired_gpu(gpu,9350+gpu)
    leases=json.loads((OUT/'external_gpu_leases.json').read_text())
    atomic(OUT/'final_resource_release.json',dict(time=time.time(),idea1_compute_render_processes_all_exited=True,checked_physical_gpus=[0,2,3],RPC_ports_free=[9350,9352,9353],nvidia_smi=devices,
        external_gpu1_lease=leases.get('1'),GPU1_untouched=True,scope='Only idea1 assigned GPUs checked for retirement; externalGPU1 is not terminated or inferred released from idle.'))
    for name,args in [('execution_audit',['-m','preparations.idea1_v3_3.audit','--full']),('reports',['-m','preparations.idea1_v3_3.report']),('evidence_index',['-m','preparations.idea1_v3_3.evidence_index'])]:
        atomic(status,dict(time=time.time(),state='running_'+name,science_valid=state['complete']))
        with (OUT/('final_'+name+'.log')).open('a') as log:subprocess.run([str(PY['pi0']),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
    atomic(status,dict(time=time.time(),state='complete',science_valid=state['complete'],engineering_valid=400,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),evidence_index_sha256=hashlib.sha256((OUT/'execution_evidence_index.json').read_bytes()).hexdigest()))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        atomic(OUT/'finalization_error.json',dict(time=time.time(),error=repr(exc)));raise
