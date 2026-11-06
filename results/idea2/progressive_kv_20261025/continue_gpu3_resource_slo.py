from pathlib import Path
import json,time,subprocess,sys
r=Path(__file__).resolve().parent
while True:
 p=r/'resource_confirmation_selection.json'
 if p.exists():
  selection=json.loads(p.read_text())
  if selection['decision']!='confirm':raise SystemExit(0)
  break
 time.sleep(5)
while True:
 decision=json.loads((r/'sm32_quality_selection.json').read_text())
 if decision['decision'].startswith('skip'):break
 p=r/'closedloop_sm32_gpu3_clean_port19087/execution.json'
 if p.exists():
  d=json.loads(p.read_text())
  if d['status']=='failed':raise RuntimeError('Quality technical failure')
  if d['status']=='passed':break
 time.sleep(5)
selected=selection['selected_ae_sms']
m=[]
for period in [33,62]:
 for reverse in [False,True]:
  m.append(dict(name=f'gpu3_sm{selected}_slo_{period}ms'+('_repeat' if reverse else ''),module='results.idea2.progressive_kv_20261025.resource_slo',args=['--bank',str(r/'same_task_bank.pkl'),'--ae-sm-min',str(selected),'--arrival-period-ms',str(period),'--conditions','3/encoder','--pairs','0','--continuous','100','--warmup','20','--verify-count','1']+(['--reverse'] if reverse else [])))
p=r/'gpu3_resource_slo_matrix.json';p.write_text(json.dumps(m,indent=2))
raise SystemExit(subprocess.call([sys.executable,str(r/'run_isolated_matrix.py'),str(p),'GPU-555cde36-5e41-bc44-0c69-ea47358fd703',','.join(map(str,range(96,112)))]))
