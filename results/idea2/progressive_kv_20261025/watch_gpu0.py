import datetime,json,os,signal,subprocess,time,xml.etree.ElementTree as ET
from pathlib import Path
r=Path(__file__).resolve().parent
receipt=r/'gpu0_resource_matrix.execution.json'
log=r/'gpu0_occupancy_watchdog.json'
state={'status':'watching','observations':[]}
while True:
 if receipt.exists():
  d=json.loads(receipt.read_text())
  if d['status']!='running':state['status']='finished';break
  cases=d.get('cases',[]);case=cases[-1] if cases else None
 else:case=None
 xml=subprocess.check_output(['nvidia-smi','-i','GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1','-q','-x'],text=True)
 g=ET.fromstring(xml).find('gpu')
 procs=[{'pid':int(p.findtext('pid')),'type':p.findtext('type')} for p in g.findall('processes/process_info')]
 conflict=len(procs)>1 or any('G' in str(p['type']) for p in procs)
 row={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'case':case.get('name') if case else None,'processes':procs,'conflict':conflict}
 state['observations'].append(row)
 if conflict:
  state['status']='interference_detected'
  if case and case.get('status')=='running':
   os.kill(case['pid'],signal.SIGTERM);state['terminated_only_own_container_pid']=case['pid']
  break
 if len(state['observations'])%5==0:log.write_text(json.dumps(state,indent=2)+'\n')
 time.sleep(1)
log.write_text(json.dumps(state,indent=2)+'\n')
