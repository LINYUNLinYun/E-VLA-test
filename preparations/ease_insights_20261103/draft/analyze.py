"""Paired target-fixed analysis of FLASH and pure draft source checkpoints."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import atomic_json

def mean(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return sum(values) / len(values) if values else None

def ratio(a, b):
    return a / b if a is not None and b else None

def wilson(success, n):
    if not n:
        return [None, None]
    z = 1.959963984540054
    center = (success / n + z*z/(2*n)) / (1+z*z/n)
    half = z * math.sqrt(success/n*(1-success/n)/n + z*z/(4*n*n)) / (1+z*z/n)
    return [center-half, center+half]

def exact_discordance_p(first_only, second_only):
    n = first_only + second_only
    if not n:
        return 1.
    return min(1., 2. * sum(math.comb(n, k) for k in range(min(first_only, second_only)+1)) / 2**n)

def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader(); writer.writerows(rows)

def latency_subset_tables(records, preparation):
    """Prospectively fixed held-out subsets, never classified as provably warm."""
    protocol_path=HERE/'latency_protocol.json'
    protocol=json.loads(protocol_path.read_text())
    heldout=[r for r in records if r['init_id'] in (21,22,23,24)]
    if not heldout:
        return dict(status='not_applicable_no_heldout_init21_24',summary=[],comparisons=[],preparation=[],
                    protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),warm_cold_labels_created=False)
    summaries=[];comparisons=[]
    def totals(rows):
        return {field:sum(r[field] for r in rows) for field in ('success','requests','env_steps',
                    'policy_total_ms','client_rpc_total_ms','wall_seconds')}
    for subset in protocol['fixed_subsets']:
        selected=[r for r in heldout if r['init_id'] in subset['init_ids']]
        groups=defaultdict(list)
        for row in selected:groups[(row['mode'],row['source_suite'],row['target_suite'])].append(row)
        index={(r['mode'],r['source_suite'],r['target_suite'],r['task_id'],r['init_id']):r for r in selected}
        full={(r['target_suite'],r['task_id'],r['init_id']):r for r in selected if r['mode']=='full'}
        for (mode,source,target),rows in sorted(groups.items()):
            total=totals(rows);n=len(rows)
            expected={(task,init) for task in subset['tasks'] for init in subset['init_ids']}
            actual={(r['task_id'],r['init_id']) for r in rows}
            if not actual.issubset(expected):raise ValueError('Unexpected case in preregistered latency subset')
            summaries.append(dict(subset=subset['name'],mode=mode,source_suite=source,target_suite=target,
                n=n,expected_n=subset['expected_episodes_per_condition'],status='complete' if actual==expected else 'incomplete',
                successes=total['success'],success_rate=ratio(total['success'],n),
                policy_ms_per_executed_action=ratio(total['policy_total_ms'],total['env_steps']),
                client_rpc_ms_per_executed_action=ratio(total['client_rpc_total_ms'],total['env_steps']),
                episode_wall_total_seconds=total['wall_seconds'],mean_episode_wall_seconds=ratio(total['wall_seconds'],n),
                policy_total_ms=total['policy_total_ms'],client_rpc_total_ms=total['client_rpc_total_ms'],
                executed_actions=total['env_steps'],requests=total['requests'],
                mean_policy_round_ms_not_actual_speedup=ratio(total['policy_total_ms'],total['requests']),
                label='repeat-task, not guaranteed compilation-free' if subset['name']=='repeat_task_init22_24' else subset['purpose'],
                warm_state_claim=False))
            if mode=='full':continue
            for reference_kind in ('Full','matching_source'):
                if reference_kind=='matching_source' and source==target:continue
                paired=[]
                for row in rows:
                    reference=(full.get((target,row['task_id'],row['init_id'])) if reference_kind=='Full' else
                               index.get((mode,target,target,row['task_id'],row['init_id'])))
                    if reference is not None:paired.append((reference,row))
                if not paired:continue
                reference=totals([x for x,y in paired]);condition=totals([y for x,y in paired])
                rc=ratio(reference['policy_total_ms'],reference['env_steps']);cc=ratio(condition['policy_total_ms'],condition['env_steps'])
                rr=ratio(reference['client_rpc_total_ms'],reference['env_steps']);cr=ratio(condition['client_rpc_total_ms'],condition['env_steps'])
                comparisons.append(dict(subset=subset['name'],mode=mode,source_suite=source,target_suite=target,
                    reference=reference_kind,n_paired=len(paired),expected_n=subset['expected_episodes_per_condition'],
                    reference_successes=reference['success'],condition_successes=condition['success'],
                    reference_policy_ms_per_executed_action=rc,condition_policy_ms_per_executed_action=cc,
                    policy_per_action_cost_ratio_reference_over_condition=ratio(rc,cc),
                    reference_rpc_ms_per_executed_action=rr,condition_rpc_ms_per_executed_action=cr,
                    rpc_per_action_cost_ratio_reference_over_condition=ratio(rr,cr),
                    reference_episode_wall_seconds=reference['wall_seconds'],condition_episode_wall_seconds=condition['wall_seconds'],
                    episode_wall_ratio_reference_over_condition=ratio(reference['wall_seconds'],condition['wall_seconds']),
                    paired_case_keys=['%s_t%02d_i%02d'%(target,y['task_id'],y['init_id']) for x,y in paired],
                    warm_state_claim=False))
    return dict(status='heldout_subsets',summary=summaries,comparisons=comparisons,preparation=preparation,
                protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),warm_cold_labels_created=False)


def analyze(roots, output):
    runs = []
    for root in roots:
        root = Path(root)
        if (root / 'manifest.json').exists() and (root / 'cases.json').exists():
            runs.append(root)
        else:
            runs += [p.parent for p in root.rglob('cases.json') if (p.parent / 'manifest.json').exists()]
    runs = sorted(set(runs))
    records, seen, audit = [], set(), []
    run_preparation=[]
    initial_inputs = defaultdict(list)
    for run in runs:
        manifest = json.loads((run / 'manifest.json').read_text())
        if 'protocol' not in manifest or manifest.get('mode') not in ('flash','draft','full'):
            continue
        mode, source = manifest['mode'], manifest['source_suite']
        reqs = defaultdict(list)
        request_path = run / 'requests.jsonl'
        if request_path.exists():
            with request_path.open() as f:
                for line in f:
                    r = json.loads(line)
                    reqs[(r['suite'], r['task_id'], r['init_id'])].append(r)
        expected = json.loads((run / 'cases.json').read_text())
        if any(c['init_id'] in (21,22,23,24) for c in expected):
            run_result_path=run/'result.json'
            run_result=json.loads(run_result_path.read_text()) if run_result_path.exists() else {}
            task_order=[];init_order=defaultdict(list)
            for case in expected:
                task=(case['suite'],case['task_id'])
                if not task_order or task_order[-1]!=task:task_order.append(task)
                init_order[task].append(case['init_id'])
            run_preparation.append(dict(run=str(run),mode=mode,source_suite=source,
                target_suites=list(dict.fromkeys(c['suite'] for c in expected)),
                status=run_result.get('status','unknown'),planned_cases=len(expected),
                planned_init_ids=sorted({c['init_id'] for c in expected}),
                each_task_contiguous=len(task_order)==len(set(task_order)),
                each_task_order_is_21_22_23_24=all(v==[21,22,23,24] for v in init_order.values()),
                preparation_seconds=run_result.get('preparation_seconds'),
                run_wall_seconds=run_result.get('wall_seconds'),closed_loop_seconds=run_result.get('closed_loop_seconds'),
                attribution='Once-per-run preparation remains separate; it is not assigned or subtracted from any fixed subset',
                request_level_cold_capture_marker_available=False))
        for c in expected:
            key = '%s_t%02d_i%02d' % (c['suite'],c['task_id'],c['init_id'])
            path = run / 'episodes' / key / 'result.json'
            if not path.exists():
                audit.append(dict(run=str(run), case=key, issue='missing_result')); continue
            receipt = json.loads(path.read_text())
            if receipt.get('status') != 'complete':
                audit.append(dict(run=str(run), case=key, issue='technical_failure')); continue
            identity = (mode, source, c['suite'],c['task_id'],c['init_id'])
            if identity in seen:
                raise ValueError('Duplicate scientific condition; choose one authoritative run: %r' % (identity,))
            seen.add(identity)
            requests = reqs[(c['suite'],c['task_id'],c['init_id'])]
            episode = receipt['episode']
            if len(requests) != episode['infer_calls']:
                raise ValueError('Request count mismatch in ' + str(path))
            infer_ids = [r['replan_id'] for r in requests]
            if infer_ids != list(range(len(requests))):
                raise ValueError('Noncontiguous inference ids in ' + str(path))
            if requests:
                first = requests[0]
                initial_inputs[(c['suite'],c['task_id'],c['init_id'])].append(
                    dict(mode=mode, source=source, input=first['input_sha256'], noise=first['noise_sha256']))
            draft = [r for r in requests if r['route_calls']['draft'] == 1]
            full = [r for r in requests if r['route_calls']['full'] == 1]
            verify = [r for r in requests if r['route_calls']['verify'] == 1]
            if mode == 'draft' and (full or verify or len(draft) != len(requests)):
                raise ValueError('Pure draft route violation in ' + str(path))
            if mode == 'full' and (draft or verify or len(full) != len(requests)):
                raise ValueError('Full route violation in ' + str(path))
            if mode == 'flash' and len(draft) != len(verify):
                raise ValueError('FLASH draft/verify mismatch in ' + str(path))
            accepted = sum(r['timing']['accepted_prefix_len'] for r in verify)
            latency = [r['timing']['total_ms'] for r in requests]
            warm_latency = [r['timing']['total_ms'] for r in requests[1:]]
            client_rpc_total = (float(receipt['client_roundtrip_sum_ms']) if 'client_roundtrip_sum_ms' in receipt
                                else float(receipt['mean_client_roundtrip_ms']) * len(requests))
            policy_total = sum(r['policy_wall_ms'] for r in requests)
            action_count = episode['executed_action_count']
            record = dict(mode=mode, source_suite=source, target_suite=c['suite'],task_id=c['task_id'],init_id=c['init_id'],
                success=int(receipt['success']), requests=len(requests), draft_rounds=len(draft), verified_rounds=len(verify),
                full_rounds=len(full), accepted_actions=accepted, acceptance_rate=ratio(accepted,12*len(verify)),
                zero_action_draft_rounds=sum(r['timing']['accepted_prefix_len'] == 0 for r in verify),
                full_round_fraction=ratio(len(full),len(requests)),
                rejection_full_rounds=sum(r['timing'].get('ease_rejection_full',0) for r in requests),
                periodic_full_rounds=sum(r['timing'].get('ease_periodic_full',0) for r in requests),
                initial_full_rounds=sum(r['timing'].get('ease_initial_full',0) for r in requests),
                scheduled_fallback_rounds=sum(r['timing'].get('scheduled_full_fallback',0) for r in requests),
                env_steps=episode['executed_action_count'], wall_seconds=receipt['wall_seconds'],
                model_total_ms=sum(latency), mean_model_round_ms=mean(latency),
                mean_model_round_after_first_ms=mean(warm_latency),
                mean_client_roundtrip_ms=receipt['mean_client_roundtrip_ms'],
                policy_total_ms=policy_total, client_rpc_total_ms=client_rpc_total,
                policy_ms_per_executed_action=ratio(policy_total,action_count),
                client_rpc_ms_per_executed_action=ratio(client_rpc_total,action_count),
                model_ms_per_executed_action=ratio(sum(latency),action_count),
                mean_policy_round_ms=mean([r['policy_wall_ms'] for r in requests]),
                run=str(run), result=str(path))
            records.append(record)
    groups = defaultdict(list)
    for row in records:
        groups[(row['mode'], row['source_suite'],row['target_suite'])].append(row)
    summary = []
    for (mode,source,target), rows in sorted(groups.items()):
        totals = {k:sum(r[k] for r in rows) for k in ('success','requests','draft_rounds','verified_rounds','full_rounds',
            'accepted_actions','rejection_full_rounds','periodic_full_rounds','initial_full_rounds','scheduled_fallback_rounds',
            'env_steps','wall_seconds','model_total_ms','policy_total_ms','client_rpc_total_ms','zero_action_draft_rounds')}
        n = len(rows); interval = wilson(totals['success'],n)
        summary.append(dict(mode=mode,source_suite=source,target_suite=target,n=n,
            successes=totals['success'],success_rate=totals['success']/n,success_ci_low=interval[0],success_ci_high=interval[1],
            acceptance_rate=ratio(totals['accepted_actions'],12*totals['verified_rounds']),
            full_round_fraction=ratio(totals['full_rounds'],totals['requests']),
            rejection_full_fraction=ratio(totals['rejection_full_rounds'],totals['requests']),
            periodic_full_fraction=ratio(totals['periodic_full_rounds'],totals['requests']),
            scheduled_fallback_fraction=ratio(totals['scheduled_fallback_rounds'],totals['verified_rounds']),
            mean_model_round_ms=ratio(totals['model_total_ms'],totals['requests']),
            mean_policy_round_ms=ratio(totals['policy_total_ms'],totals['requests']),
            mean_client_roundtrip_ms=ratio(totals['client_rpc_total_ms'],totals['requests']),
            policy_ms_per_executed_action=ratio(totals['policy_total_ms'],totals['env_steps']),
            client_rpc_ms_per_executed_action=ratio(totals['client_rpc_total_ms'],totals['env_steps']),
            model_ms_per_executed_action=ratio(totals['model_total_ms'],totals['env_steps']),
            mean_episode_wall_seconds=totals['wall_seconds']/n,
            episodes_per_hour=ratio(n*3600,totals['wall_seconds']), **{k:v for k,v in totals.items() if k != 'success'}))
    pairs = []
    index = {(r['mode'],r['source_suite'],r['target_suite'],r['task_id'],r['init_id']):r for r in records}
    for (mode, source,target), rows in sorted(groups.items()):
        if mode == 'full' or source == target:
            continue
        matched=[]
        for r in rows:
            baseline = index.get((mode,target,target,r['task_id'],r['init_id']))
            if baseline:
                matched.append((baseline,r))
        if not matched:
            continue
        native_only=sum(x['success'] and not y['success'] for x,y in matched)
        foreign_only=sum(y['success'] and not x['success'] for x,y in matched)
        pairs.append(dict(mode=mode,source_suite=source,target_suite=target,n_paired=len(matched),
            matched_successes=sum(x['success'] for x,y in matched),foreign_successes=sum(y['success'] for x,y in matched),
            matched_only=native_only,foreign_only=foreign_only,
            foreign_minus_matched_success_rate=(foreign_only-native_only)/len(matched),
            exact_mcnemar_p_uncorrected=exact_discordance_p(native_only,foreign_only),
            caution='exploratory multiple comparisons; independent init replication required'))
    full_records=[r for r in records if r['mode']=='full']
    full_cases={(r['target_suite'],r['task_id'],r['init_id']):r for r in full_records}
    if len(full_cases)!=len(full_records):
        raise ValueError('Multiple Full baselines for the same case; select one authoritative baseline before computing cost ratios')
    speed=[]; paired_action_costs=[]
    for (mode, source,target), rows in sorted(groups.items()):
        if mode=='full': continue
        matched=[(full_cases[(r['target_suite'],r['task_id'],r['init_id'])],r) for r in rows
                 if (r['target_suite'],r['task_id'],r['init_id']) in full_cases]
        if not matched: continue
        b=[x for x,y in matched]; v=[y for x,y in matched]
        for baseline,condition in matched:
            paired_action_costs.append(dict(mode=mode,source_suite=source,target_suite=target,
                task_id=condition['task_id'],init_id=condition['init_id'],
                full_success=baseline['success'],condition_success=condition['success'],
                full_executed_actions=baseline['env_steps'],condition_executed_actions=condition['env_steps'],
                full_wall_seconds=baseline['wall_seconds'],condition_wall_seconds=condition['wall_seconds'],
                full_policy_ms_per_executed_action=baseline['policy_ms_per_executed_action'],
                condition_policy_ms_per_executed_action=condition['policy_ms_per_executed_action'],
                policy_per_executed_action_speedup_vs_full=ratio(baseline['policy_ms_per_executed_action'],condition['policy_ms_per_executed_action']),
                full_client_rpc_ms_per_executed_action=baseline['client_rpc_ms_per_executed_action'],
                condition_client_rpc_ms_per_executed_action=condition['client_rpc_ms_per_executed_action'],
                client_rpc_per_executed_action_speedup_vs_full=ratio(baseline['client_rpc_ms_per_executed_action'],condition['client_rpc_ms_per_executed_action']),
                mean_policy_round_ratio_not_actual_speedup=ratio(baseline['mean_policy_round_ms'],condition['mean_policy_round_ms'])))
        def pooled_cost(data,field):
            return ratio(sum(r[field] for r in data),sum(r['env_steps'] for r in data))
        full_policy_cost,condition_policy_cost=pooled_cost(b,'policy_total_ms'),pooled_cost(v,'policy_total_ms')
        full_rpc_cost,condition_rpc_cost=pooled_cost(b,'client_rpc_total_ms'),pooled_cost(v,'client_rpc_total_ms')
        speed.append(dict(mode=mode,source_suite=source,target_suite=target,n_paired=len(matched),
            full_successes=sum(r['success'] for r in b),condition_successes=sum(r['success'] for r in v),
            full_episode_wall_seconds=sum(r['wall_seconds'] for r in b),condition_episode_wall_seconds=sum(r['wall_seconds'] for r in v),
            paired_episode_wall_speedup=ratio(sum(r['wall_seconds'] for r in b),sum(r['wall_seconds'] for r in v)),
            full_policy_ms_per_executed_action=full_policy_cost,condition_policy_ms_per_executed_action=condition_policy_cost,
            policy_per_executed_action_speedup_vs_full=ratio(full_policy_cost,condition_policy_cost),
            full_client_rpc_ms_per_executed_action=full_rpc_cost,condition_client_rpc_ms_per_executed_action=condition_rpc_cost,
            client_rpc_per_executed_action_speedup_vs_full=ratio(full_rpc_cost,condition_rpc_cost),
            gpu_stage_work_ratio_not_end_to_end=ratio(sum(r['model_total_ms'] for r in b),sum(r['model_total_ms'] for r in v)),
            mean_policy_round_ratio_not_actual_speedup=ratio(ratio(sum(r['policy_total_ms'] for r in b),sum(r['requests'] for r in b)),
                                              ratio(sum(r['policy_total_ms'] for r in v),sum(r['requests'] for r in v))),
            mean_gpu_stage_round_ratio_not_actual_speedup=ratio(ratio(sum(r['model_total_ms'] for r in b),sum(r['requests'] for r in b)),
                                              ratio(sum(r['model_total_ms'] for r in v),sum(r['requests'] for r in v))),
            caution='Primary service cost is total complete policy/RPC ms divided by executed actions, including zero-action rejection requests. Ratios above1 mean lower cost than Full. Mean request latency alone is not actual speedup. Different closed-loop trajectories; assess episode wall and successes alongside costs; GPU-stage timing is not end-to-end.'))
    input_audit=[]
    for key,rows in sorted(initial_inputs.items()):
        input_audit.append(dict(suite=key[0],task_id=key[1],init_id=key[2],conditions=len(rows),
            initial_input_hashes=len(set(r['input'] for r in rows)),noise_hashes=len(set(r['noise'] for r in rows)),
            matched=len(set(r['input'] for r in rows))==1 and len(set(r['noise'] for r in rows))==1))
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    for name,rows in [('episodes',records),('summary',summary),('paired_vs_matching_source',pairs),
                      ('speed_vs_full',speed),('paired_action_costs_vs_full',paired_action_costs),('initial_pairing_audit',input_audit)]:
        write_csv(output/(name+'.csv'),rows)
        atomic_json(output/(name+'.json'),rows)
    subset_outputs=latency_subset_tables(records,run_preparation)
    subset_status={k:v for k,v in subset_outputs.items() if k not in ('summary','comparisons','preparation')}
    atomic_json(output/'latency_subset_status.json',subset_status)
    for name,key in [('latency_subsets','summary'),('latency_subset_comparisons','comparisons'),('run_preparation_costs','preparation')]:
        if subset_outputs['status']=='heldout_subsets':
            write_csv(output/(name+'.csv'),subset_outputs[key]);atomic_json(output/(name+'.json'),subset_outputs[key])
        else:
            for suffix in ('.csv','.json'):
                stale=output/(name+suffix)
                if stale.exists():stale.unlink()
    result=dict(runs=len(runs),complete_conditions=len(records),uncompleted=audit,
                initial_pairing_mismatches=[r for r in input_audit if not r['matched']],
                latency_subsets=subset_status,
                primary_cost_metric='policy_ms_per_executed_action = sum(complete native policy-call wall ms) / executed actions; includes rejected zero-action requests',
                secondary_cost_metric='client_rpc_ms_per_executed_action = sum(complete client RPC ms) / executed actions',
                latency_warning='Mean request latency alone is not actual acceleration; GPU-stage model timing is not end-to-end. Report successful episodes and real episode wall alongside service costs.',
                no_pure_draft_acceptance_claim=True, prepared_costs='per-run result.json preparation_seconds; first request costs additionally in episode wall')
    atomic_json(output/'audit.json',result)
    report='# Draft generalization analysis\n\nPrimary service metric: complete native policy-call milliseconds per actually executed action. Client RPC cost per executed action is reported alongside it. All inference calls, including rejection rounds that execute zero actions, contribute to the numerator. Mean request latency is retained as a diagnostic and cannot alone establish actual acceleration. GPU-stage timing is not end-to-end.\n\n'
    report+='| Mode | Source | Target | Success / n | Policy ms / action | RPC ms / action | Episode wall mean (s) |\n|---|---|---|---:|---:|---:|---:|\n'
    for row in summary:
        fmt=lambda value: 'N/A' if value is None else '%.3f'%value
        report+='| %s | %s | %s | %d/%d | %s | %s | %s |\n'%(row['mode'],row['source_suite'],row['target_suite'],row['successes'],row['n'],fmt(row['policy_ms_per_executed_action']),fmt(row['client_rpc_ms_per_executed_action']),fmt(row['mean_episode_wall_seconds']))
    report+='\nSame-case comparisons are in `paired_action_costs_vs_full.json`; aggregate ratios are in `speed_vs_full.json`. Service-cost speedup is Full cost / condition cost; below1 means higher condition cost. Real episode-wall ratios and both success counts are retained because trajectories and episode lengths may differ.\n'
    if subset_outputs['status']=='heldout_subsets':
        report+='\nPreregistered held-out latency subsets are in `latency_subsets.json`: all init21–24, first-task init21, and fixed repeat-task init22–24. Same-case Full/matching-source cost ratios are in `latency_subset_comparisons.json`. Once-per-run readiness/preparation and run wall are separate in `run_preparation_costs.json`. Repeat-task is not guaranteed free of compilation/capture and is never labeled deployment warm-state acceleration. No success/latency-based filtering is used.\n'
    else:
        report+='\nThe init21–24 latency subset protocol does not apply to these discovery-only inputs. No warm-state or repeat-task cost table is generated.\n'
    (output/'README.md').write_text(report)
    print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args(); analyze(a.runs,a.output)
