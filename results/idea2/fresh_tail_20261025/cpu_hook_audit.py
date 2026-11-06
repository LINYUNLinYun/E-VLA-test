import sys,types,contextlib,importlib,json,pathlib,hashlib
A=importlib.import_module('results.idea2.fresh_tail_20261025.adapter');log=[];capture=False;fail=False
class Event:
 def record(self,s):log.append(('record',))
class Stream:
 def wait_event(self,e):log.append(('wait',))
class External:
 def wait(self,e,s):log.append(('external_wait',))
 def mark(self,e,s):log.append(('external_mark',))
def snapshot(f,h,step,**kw):log.append(('snapshot',step))
class Base:
 def _capture_ae(self,index):
  global capture
  for capture in (False,True):
   for i in range(self.steps):
    runtime.snapshot(None,None,i,override=None)
    if fail and i==3:raise RuntimeError('mock capture failure')
class Runtime:
 def request(self,*args,**kw):log.append(('base_request',));return None,{'frontiers':[18]*10,'slot':0},None
runtime=types.ModuleType('preparations.idea2.progressive_kv');runtime.ProgressiveGraphs=Base;runtime.ProgressiveRuntime=Runtime;runtime.snapshot=snapshot;runtime.initialized_events=lambda *args:[Event()]
resources=types.ModuleType('results.idea2.progressive_kv_20261025.resource_adapter')
def install(*args):runtime.ProgressiveGraphs=Base;return [],[]
resources.install=install
sys.modules['preparations.idea2.progressive_kv']=runtime;sys.modules['results.idea2.progressive_kv_20261025.resource_adapter']=resources
sys.modules['torch']=types.SimpleNamespace(cuda=types.SimpleNamespace(stream=lambda s:contextlib.nullcontext(),is_current_stream_capturing=lambda:capture,nvtx=types.SimpleNamespace(range=lambda s:contextlib.nullcontext())))
results=[]
for sms in (16,32):
 for tail in (0,1,2):
  A.install(sms,tail);g=runtime.ProgressiveGraphs();g.steps=10;g.ae_stream=Stream();g.external=External();g.slots=[types.SimpleNamespace(layer_ready=[Event()])];log.clear();g._capture_ae(0);assert runtime.snapshot is snapshot
  assert sum(x[0]=='snapshot' for x in log)==20
  for i,x in enumerate(log):
   if x[0] in ('wait','external_wait'):
    assert log[i+1][0] in ('record','external_mark');assert log[i+2]==('snapshot',10-tail)
  assert sum(x[0]=='wait' for x in log)==int(tail>0);assert sum(x[0]=='external_wait' for x in log)==int(tail>0)
  if tail:
   r=runtime.ProgressiveRuntime();log.clear()
   try:r.request(None,None,force_old=True)
   except ValueError:pass
   else:raise AssertionError('force_old accepted')
   assert not log
   fail=True
   try:g._capture_ae(0)
   except RuntimeError:pass
   else:raise AssertionError('failure injection did not run')
   assert runtime.snapshot is snapshot;fail=False
  results.append(dict(sm=sms,tail=tail,passed=True))
p=pathlib.Path('results/idea2/fresh_tail_20261025');out=dict(status='passed',scope='Pure Python fake torch/runtime/resource modules; no GPU, no real graph execution',adapter_sha256=hashlib.sha256((p/'adapter.py').read_bytes()).hexdigest(),cases=results,checks=['20 original snapshots preserved per warmup+capture simulation','only step8/9 wait before snapshot','warmup normal wait; capture external wait/mark','tail0 no wait/hook side effects','force_old rejected before base runtime','snapshot hook restored after success and injected failure'])
(p/'cpu_hook_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
