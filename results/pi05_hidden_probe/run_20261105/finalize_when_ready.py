"""Finish the auditable report after the persistent scientific runs finish."""
import json,os,subprocess,time
from pathlib import Path
from experiments.pi05_hidden_probe.common import write_json

base=Path(__file__).resolve().parent;root=base.parents[2]
while True:
    status=json.loads((base/'experiment_supervisor.json').read_text())
    if status['stage']=='all_runs_and_analysis_complete':break
    if status['stage']=='technical_error':raise RuntimeError('Scientific supervisor needs attention; final report not generated')
    time.sleep(10)
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(root),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
with (base/'finalize_report.log').open('w') as log:
    process=subprocess.Popen([str(root/'policies/openpi/.venv/bin/python'),'-u',str(base/'finalize_report.py')],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    write_json(base/'final_report_job.json',{'state':'running','pid':process.pid,'started_unix':time.time()})
    code=process.wait()
write_json(base/'final_report_job.json',{'state':'finished','pid':process.pid,'exit_code':code,'interpretation_review_remaining':True,'finished_unix':time.time()})
if code:raise SystemExit(code)
print('Report and matrix audit generated; primary-agent interpretation review remains',flush=True)
