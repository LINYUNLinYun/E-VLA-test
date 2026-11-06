"""Bounded saved-evidence audit; no model, RPC, simulation or GPU sampling."""
import argparse
import json
from pathlib import Path
import numpy as np
from .native_envs import ROOT,sha,write_json
from .d2_cross_device_probe import comparison
from preparations.idea1_v3_2.transfer_prompts import instruction_tokens


def first(path):
    with Path(path).open() as stream:return json.loads(stream.readline())


def audit(probe_root,output):
    probe_root=Path(probe_root);manifest=json.loads((probe_root/'manifest.json').read_text());summaries=[];target_inputs=set();inputs={}
    for ordinal,ref in enumerate(manifest['references']):
        row=first(ref['infer']['path']);response=row['response']
        with np.load(ref['observation']['path'],allow_pickle=False) as saved:instruction=str(saved['instruction'])
        result=json.loads(Path(ref['result']['path']).read_text())
        gpu0=probe_root/'gpu0'/('reference%02d'%ordinal)/'repeat0.npz';gpu1=probe_root/'gpu1'/('reference%02d'%ordinal)/'repeat0.npz'
        with np.load(gpu0) as a,np.load(gpu1) as b:
            historical={key:comparison(a[key],a['historical_'+key]) for key in ('raw','decoded')}
            cross={key:comparison(a[key],b[key]) for key in ('raw','decoded')}
        count={category:{mode:{key:value for key,value in counters.items() if key in ('logical_requests','attempted_requests','constructor_graph_capture_suppressed')}
                for mode,counters in modes.items()} for category,modes in response['counters'].items()}
        summaries.append(dict(ordinal=ordinal,group=ref['group'],source_index=ref['source_index'],mother_suite=ref['mother_suite'],mode=ref['mode'],
            instruction=instruction,token_count=len(instruction_tokens(instruction)[1]),seed=row['seed'],input_hash=response['input_hash'],noise_hash=response['noise_hash'],
            counters_after_first_request=count,first_request_observed_calls=response['observed_calls'],startup_events=response.get('startup_events',[]),
            historical_result_pid=result['pid'],historical_result_requests=result['requests'],historical_source_files=result['source_files'],
            versus_historical=historical,gpu0_versus_gpu1=cross))
        if ordinal<2:target_inputs.add(response['input_hash'])
        for evidence in (ref['result'],ref['observation'],ref['infer']):inputs[evidence['path']]=evidence['sha256']
    scanned=0;matches=[]
    supplement=ROOT/'results/idea1_v3_2_e4_supplement'
    for path in supplement.rglob('infer.jsonl'):
        try:row=first(path)
        except (ValueError,FileNotFoundError):continue
        scanned+=1;response=row.get('response',{})
        if response.get('input_hash') in target_inputs:
            matches.append(dict(path=str(path.relative_to(ROOT)),mode=response.get('mode'),seed=row.get('seed'),request=row.get('request'),
                input_hash=response['input_hash'],noise_hash=response.get('noise_hash'),sha256=sha(path)))
    repeat=supplement/'plus_same_gpu_repeat_20261022T1224Z/receipt.json';repeat_value=json.loads(repeat.read_text())
    repeat_input={c['input_hash'] for c in repeat_value['calls']}
    repeat_info=dict(path=str(repeat.relative_to(ROOT)),sha256=sha(repeat),server_pid=repeat_value['server_pid'],gpu_uuid=repeat_value['gpu_uuid'],
        matches_case0_input=bool(repeat_input & target_inputs),input_hashes=sorted(repeat_input),noise_hashes=sorted({c['noise_hash'] for c in repeat_value['calls']}),
        comparisons=repeat_value['comparisons'])
    service=supplement/'libero_plus_20261022T074328Z/service_resume_20261022T112826Z.json';service_value=json.loads(service.read_text())
    service_info=dict(path=str(service.relative_to(ROOT)),sha256=sha(service),server_pid=service_value['server_pid'],gpu_preflight=service_value['gpu_preflight'],
        started_unix=service_value['started_unix'],ready_unix=service_value['ready_unix'],completed_unix=service_value['completed_unix'],
        startup_gate=dict(passed=service_value['gate']['passed'],bank=service_value['gate']['bank'],bank_sha256=service_value['gate']['bank_sha256'],
            mode_order=[c['mode'] for c in service_value['gate']['checks']]))
    half=supplement/'libero_plus_half_20261022T1216Z/live_ledger.json';ledger=json.loads(half.read_text())
    reused=[{k:r[k] for k in ('group','source_index','mode','path','sha256','reused_historical')} for r in ledger['selected_completed'] if r['group']=='perturbed350' and r['source_index']==0]
    metadata={}
    for card in (0,1):
        receipt=json.loads((probe_root/('gpu%d'%card)/'receipt.json').read_text());runtime=receipt['runtime_metadata']
        metadata[str(card)]={k:runtime[k] for k in ('versions','sys_executable','cuda_version','device_name','compute_capability','precision','allowlisted_environment')}
    report=dict(status='CPU_saved_evidence_only',quality_requests=0,gpu_calls=0,source_sha256=sha(__file__),probe_manifest_hash=manifest['manifest_hash'],references=summaries,
        independent_reference_search=dict(scope='First rows of all existing supplement infer.jsonl only; not a claim of scanning every later observation in every log',scanned_first_rows=scanned,matches=matches,
            conclusion='Only original case0 Full/V14/draft first rows share this input. Full/V14 each have one saved production output; matching draft is a different model route.'),
        old_same_gpu_repeat=repeat_info,historical_resume_service=service_info,half_reuse=dict(ledger_sha256=sha(half),rows=reused,
            conclusion='Half ledger reuses identical original result paths/hashes; it is not an independent re-inference.'),runtime_small_metadata=metadata,input_files=inputs,
        established=['GPU0 and GPU1 all16 repeated raw/decoded outputs agree bitwise. Only perturbed case0 Full/V14 disagree with historical outputs.',
            'Case0 prompt18 tokens is the only18-token prompt in this8case sample; length specificity remains a hypothesis, not a proven cause.',
            'Historical case0 Full follows six native legacy gate reference/candidate pairs, is first quality request, and constructs a new runtime. V14 first request follows19 Full quality requests and reuses that runtime.',
            'New probe begins case0 without the historical six-route legacy preflight and inserts repeats/capture/D2 before V14. Execution history is therefore not reproduced.',
            'Historical dependency versions, environment precision toggles and intermediate buffers were not saved; hardware-only attribution is unsupported.'],
        next_minimal_hypothesis='Historical legacy preflight/runtime history changes case0 output, while cold current production path is stable across cards.',
        proposed_next_action='A separately preregistered one-card diagnostic, no quality: fresh process, exact historical six-route legacy_preflight(bank), then case0 Full saved input/seed twice and V14 saved input/seed twice; compare complete raw/decoded to historical and current stable outputs. Capture case0 intermediates in an extra pass and compare with uncaptured output. Fix order/configuration in advance; no parameter/card search. If this does not reproduce history, stop and retain both references; do not discard case0 or relax bitwise gate.',
        important_limit='Replaying identical saved initial input twice does not recreate the19 historical Full requests on evolving observations: those complete RGB inputs were not preserved. The proposed history test isolates the legacy gate branch, not every historical allocation/state transition.')
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    write_json(output,report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probe-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    report=audit(a.probe_root,a.output);print(json.dumps(dict(status=report['status'],references=len(report['references']),matched_input_first_rows=len(report['independent_reference_search']['matches']),output=a.output)))


if __name__=='__main__':main()
