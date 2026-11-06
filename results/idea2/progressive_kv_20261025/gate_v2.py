import json,pickle,traceback,time
from pathlib import Path
from preparations.idea2.layer_bench import require_idle_gpu, require_same_processes, gpu_snapshot
UUID='GPU-599477fb-0507-40dc-14df-d491f7a7b202'
R=Path(__file__).resolve().parent
result={'status':'starting','gpu_before':require_idle_gpu(UUID),'performance_eligible':False}
(R/'gate_v2.result.json').write_text(json.dumps(result,indent=2))
from preparations.idea2.experiments import load_backend,native_raw
from preparations.idea2.progressive_kv import ProgressiveRuntime
backend=load_backend('shared/checkpoints/openpi/pi05_libero_pytorch_idea2',model_config='pi05_libero',num_steps=10)
bank=pickle.loads(Path('results/idea1_v3_1/input_bank.pkl').read_bytes())
p=[backend.prepare(native_raw(x)) for x in bank]
noise=backend.noise(1701)
engine=None
try:
 engine=ProgressiveRuntime(backend,p[0],noise)
 result.update(identity=backend.identity,graph_setup_ms=engine.graphs.setup_ms,kv_addresses=engine.graphs.kv_addresses,rows=[])
 engine.bootstrap(p[0])
 for index,lead in enumerate([0,1,3,6,18,0,3,6]):
  current=p[(index+1)%len(p)]
  output,row,nxt=engine.request(current,backend.noise(1701+(index+1)%len(p)),lead=lead,verify=True)
  print('gate',index,lead,row['frontiers'],row['numerical'],flush=True)
  assert row['numerical']['matched_schedule']['max_abs']<=1e-5 and row['numerical']['matched_schedule']['mae']<=1e-6
  result['rows'].append(row)
 engine.reset(p[0])
 current=p[1]
 for index in range(4):
  next_p=p[index+2] if index<3 else None
  output,row,nxt=engine.request(current,backend.noise(1702+index),lead=3,verify=True,next_encoder=next_p is not None,lookahead=(lambda v=next_p:v) if next_p is not None else None)
  print('ahead',index,row['frontiers'],row['numerical']['matched_schedule'],flush=True)
  assert row['numerical']['matched_schedule']['max_abs']<=1e-5 and row['numerical']['matched_schedule']['mae']<=1e-6
  result['rows'].append(row)
  current=nxt
 result.update(status='passed',tickets=engine.pool.ticket_diagnostics)
except BaseException:
 result.update(status='failed',error=traceback.format_exc())
 raise
finally:
 if engine:engine.close()
 (R/'gate_v2.result.json').write_text(json.dumps(result,indent=2)+'\n')
