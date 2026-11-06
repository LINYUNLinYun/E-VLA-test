import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
root=Path('/root/nfs/embodied/benchmark')
out=root/'results/idea2/cewp_20261021'
files=['preparations/idea2/'+name for name in ['cewp_preparation.py','cewp_segmented.py','runtime_graph.py','policy.py','scheduler.py','submission.py','openpi_backend.py','graph_sampler.py','server.py','experiments.py','test_cewp_model.py','test_runtime_graph.py','test_submission.py','test_scheduler.py','test_sampling.py','test_nsys_profile.py','test_analysis_stats.py']]
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
review={
 'timestamp_utc':datetime.now(timezone.utc).isoformat(),
 'reviewer':'model_audit',
 'scope':'Independent final concurrency/lifecycle review of two-slot complete-old-KV CEWP runtime; no model structure/algorithm change.',
 'status':'passed_with_documented_operational_constraints',
 'source_sha256':{f:sha(root/f) for f in files},
 'cpu_tests':{'passed':True,'count':41,'elapsed_s':7.088,'log':'final_review_cpu_tests.log','log_sha256':sha(out/'final_review_cpu_tests.log'),'command':'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=.:policies/openpi/src policies/openpi/.venv/bin/python -m unittest preparations.idea2.test_cewp_model preparations.idea2.test_runtime_graph preparations.idea2.test_submission preparations.idea2.test_scheduler preparations.idea2.test_sampling preparations.idea2.test_nsys_profile preparations.idea2.test_analysis_stats'},
 'protocol_metadata_probe':json.loads((out/'final_review_probes.json').read_text())['protocol_metadata'],
 'findings':[
  {'id':'F1','severity':'medium','status':'fixed_and_cpu_regression_verified','file':'preparations/idea2/policy.py','description':'Completion wait or CPU output transform exception previously left a pinned old-KV Lease, causing reset/close to fail. Now recoverable failure drains both owned branch streams, releases Lease, drains/resets Pool and clears selective pending Event flags. Successful prediction path has no added synchronization or model operations.','before':'final_review_probes_before_fix.json','after':'final_review_probes.json','regression_tests':['test_output_failure_drains_pending_prefix_and_releases_lease','test_completion_wait_failure_releases_lease']},
  {'id':'F2','severity':'low','status':'documented_existing_recovery_contract','file':'preparations/idea2/policy.py','description':'Bootstrap Prefill failure after reservation can leave BUILDING with ready=None, but no Lease exists and submit() has drained partial GPU submission. Explicit reset/drain recovers; automatic retry without reset is not promised.','regression_test':'test_bootstrap_failure_can_recover_through_reset'}],
 'invariants':[
  {'name':'complete_old_kv','passed':True,'evidence':'One Lease is acquired before current observation branch dispatch and retained across all denoising steps and output copy; selective staging never introduces new-frame KV dependence. Previous-prefix wait only waits BUILDING observations with id < current request and calls Pool.poll after completion.'},
  {'name':'two_slots_no_third_full_cache','passed':True,'evidence':'Two GraphSlot instances only; independent captured KV storage per slot; warmup temporary KV deleted before capture; AE use_cache=False reads selected slot directly. Prepared condition/modulation tensors are not KV copies. Graph core metadata contains masks/positions, not extra full caches.'},
  {'name':'no_cross_stream_writable_scratch_alias','passed':True,'evidence':'Slot-owned Prepared/noise/core/actions and private Graph pools; standard Graph requires four distinct pools, segmented Graph six. Only immutable model parameters shared. Replay checks exclusive reservation/live Lease and dedicated stream; controller output cloned.'},
  {'name':'selective_event_dependencies','passed':True,'evidence':'vision_only requires ae_first and delays VLM behind recorded AE completion; vlm_only requires prefill_first and delays AE behind vision completion. Serialized host submission prevents re-record races and one-direction dependencies avoid cyclic waits. Reset/task switch drain Pool events before clearing pending flags; abort now also clears flags after drains.'},
  {'name':'temporary_forward_restore','passed':True,'evidence':'PreparedModulationCore patches norm dense.forward only in sequential cold warm/capture before any Lease; try/finally restores all forwards even if generation raises. Slot-specific modulation tensors retain original FP32 projection and norm math. Compiler combination rejected. CPU injected capture-body exception plus prior actual GPU parameter/hash/forward checks passed.'},
  {'name':'capture_and_layout_lifetime','passed':True,'evidence':'Constructors catch failures and close/drain both owned streams, clear registered slots; failed constructor is not assigned to policy.graphs. Layout mismatch drains Pool before closing old captures; task/reset preserve captures but invalidate Pool epoch and reset Event flags. Locals/traceback references may retain failed capture objects until exception references are dropped.'},
  {'name':'default_behavior','passed':True,'evidence':'Default remains pi0_libero, D10, eager, prefill_first, AE priority 0, no host workers, no optional schedule/modulation/selective/previous-prefix wait. New preparation_core is metadata only. Optional pi05 retains H10 and original noise/time/adarms numerical paths.'},
  {'name':'prefix_math','passed':True,'evidence':'Segmented embeddings preserve original image/token order, sqrt(hidden_dim) language scaling, pad masks, cumulative position IDs, attention mask and original full PaliGemma forward. No prompt-only caching of image-dependent hidden states/KV.'}],
 'prior_gpu_evidence':{
  'lifecycle_artifacts':['lifecycle_gpu0/result.json','lifecycle_wait_modulation_gpu0/result.json'],
  'configurations':13,'action_comparisons':507,'complete_prefix_hash_comparisons':364,
  'all_normalized_and_decoded_mae_and_max':0.0,'predeclared_tolerance':{'max_abs':1e-5,'mae':1e-6},
  'covered':'D1/D5/D10; graph-only/split-all/vision-only/vlm-only/prepared schedule/modulation/previous-prefix wait; slot rotation, same-address in-place input updates after own-stream drain, task/layout/reset and post-replay exception recovery.',
  'rpc':'server_rpc_gpu0/receipt.json: 13 dynamic predictions plus reset/configure/drain/shutdown, no steady recapture, process exited and listener closed.',
  'version_scope':'GPU artifacts predate the final output-error cleanup and preparation_core metadata changes in policy.py. They validate unchanged normal numerical/replay path; latest CPU tests validate final error handling and metadata. No new GPU/performance claim is inferred from this CPU review.'},
 'actual_risks_and_constraints':[
  {'kind':'single_host_contract','description':'Pool and policy API are single-owner and not reentrant. Temporary forward replacement is safe only with exclusive cold construction on this model; simultaneous graph creation or an external inference caller sharing the same model is unsupported. Local RPC server processes requests sequentially.'},
  {'kind':'immutable_parameters_and_configuration','description':'Prepared schedule/modulation is valid for immutable loaded weights and fixed num_steps. Changing graph_core_class/graph_engine_class/backend model settings in place after capture is unsupported; layout signature alone does not detect these changes. Existing server fixes them at startup.'},
  {'kind':'external_input_mutation','description':'record_stream protects source allocation lifetime, not concurrent in-place user writes. Caller must not mutate Prepared source tensors before asynchronous stage copies finish; correctness in-place audit explicitly drains owned streams before writes.'},
  {'kind':'cuda_fatal_errors','description':'Cleanup assumes owned stream synchronization remains usable. An illegal access/device-lost/capture-invalidated context can itself make synchronize fail; no automatic recovery from a poisoned CUDA context is claimed. CPU failure tests model recoverable exceptions, and prior GPU audit injects a Python exception after valid replay.'},
  {'kind':'bootstrap_failure','description':'After bootstrap submission failure issue reset/drain before retry. Explicit recovery is covered; bootstrap automatic rollback was not expanded during this minimal fix.'},
  {'kind':'measurement_scope','description':'Final CPU review does not establish GPU performance. GPU event spans include scheduling gaps/resource waits; actual overlap and performance require separate nsys and matched steady-state measurements.'}]
}
(out/'final_code_review.json').write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps({'status':review['status'],'cpu_tests':41,'files_hashed':len(files),'fixed_findings':['F1'],'artifact':str(out/'final_code_review.json')},indent=2))
