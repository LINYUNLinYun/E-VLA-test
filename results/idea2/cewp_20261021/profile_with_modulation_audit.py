"""Read-only AdaRMS attribution before any Graph setup/profile capture."""
import collections,gc,hashlib,json,runpy,sys
from pathlib import Path
import torch
from preparations.idea2 import experiments
original_loader=experiments.load_backend

def audited_loader(*args,**kwargs):
 backend=original_loader(*args,**kwargs)
 import pickle
 def option(name,default):return sys.argv[sys.argv.index(name)+1] if name in sys.argv else default
 dest=Path(option('--output','results/idea2/cewp_20261021/modulation_audit'))
 record=pickle.loads(Path(option('--bank','results/idea1_v3_1/input_bank.pkl')).read_bytes())[0]
 prepared=backend.prepare(experiments.native_raw(record));noise=backend.noise(1701)
 prefix=backend.build_prefix(prepared)
 expert=backend.model.paligemma_with_expert.gemma_expert.model
 sites={name+'.dense':module.dense for name,module in expert.named_modules() if type(module).__name__=='GemmaRMSNorm' and module.dense is not None}
 captured=[];hooks=[]
 def hook(name):
  def record(module,inputs,output):
   captured.append((name,inputs[0].detach().clone(),output.detach().clone(),tuple(module.weight.shape),str(module.weight.dtype)))
  return record
 for name,module in sites.items():hooks.append(module.register_forward_hook(hook(name)))
 try:
  with torch.no_grad():actions=backend.generate(prefix,prepared,noise)
  torch.cuda.current_stream(backend.device).synchronize()
 finally:
  for h in hooks:h.remove()
 expected=list(sites);counts=collections.Counter(x[0] for x in captured)
 same_condition_by_step=[];rows=[]
 assert len(captured)==len(sites)*backend.num_steps
 for step in range(backend.num_steps):
  chunk=captured[step*len(sites):(step+1)*len(sites)]
  assert [x[0] for x in chunk]==expected
  same_condition_by_step.append(all(torch.equal(x[1],chunk[0][1]) for x in chunk))
  for name,cond,out,weight_shape,weight_dtype in chunk:
   rows.append(dict(step=step,site=name,input_shape=list(cond.shape),input_dtype=str(cond.dtype),output_shape=list(out.shape),output_dtype=str(out.dtype),weight_shape=weight_shape,weight_dtype=weight_dtype,input_sha256=hashlib.sha256(cond.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()))
 result=dict(num_steps=backend.num_steps,sites=len(sites),calls=len(captured),site_calls=dict(counts),site_order=expected,same_condition_all_sites_each_step=same_condition_by_step,rows=rows,passed=all(same_condition_by_step) and len(sites)==37 and all(v==backend.num_steps for v in counts.values()),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Read-only forward hooks on ORIGINAL eager generate; extra clones only diagnostic, all outside Graph setup and profiler capture. Hooks removed and single prefix discarded before benchmark.')
 (dest/'modulation_attribution.json').write_text(json.dumps(result,indent=2)+'\n')
 assert result['passed']
 del captured,actions,prefix,prepared,noise,sites
 gc.collect()
 torch.cuda.current_stream(backend.device).synchronize()
 return backend

experiments.load_backend=audited_loader
runpy.run_module('preparations.idea2.cewp_bench',run_name='__main__')
