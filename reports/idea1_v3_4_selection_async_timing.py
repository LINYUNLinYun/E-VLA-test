"""Critical-path accounting; overlapping GPU spans are never summed as wall."""
import csv,json
from collections import Counter
from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4_selection_async.common import OUT,write_csv


def run():
    follow=json.loads((OUT/'followup_manifest.json').read_text())
    selected=list(dict.fromkeys(['v12_p3_q4_serial_async','v12_p3_q4_batch_async','v12_p3_q4_stride2','v12_p3_q4_stride3']+follow['diagnostic_conditions']))
    summary=[];categories=[]
    req=list(csv.DictReader((OUT/'requests.csv').open()));episodes=list(csv.DictReader((OUT/'episodes.csv').open()))
    for condition in selected:
        current=[r for r in req if r['condition']==condition and int(r['request'])>=3]
        case_keys={r['case'] for r in current}
        base=[r for r in req if r['condition']=='full_d10' and int(r['request'])>=3 and r['case'] in case_keys]
        ee=[r for r in episodes if r['condition']==condition and r['case'] in case_keys]
        be=[r for r in episodes if r['condition']=='full_d10' and r['case'] in case_keys]
        policy_ms=sum(float(r['warm_tail_ms']) for r in ee);actions=sum(int(r['warm_tail_actions']) for r in ee)
        base_cost=sum(float(r['warm_tail_ms']) for r in be)/sum(int(r['warm_tail_actions']) for r in be)
        collect=sum(float(r['selector_collect_ms']) for r in current)
        launch=sum(float(r['selector_launch_ms']) for r in current)
        waits=sum(float(r['selector_wait_ms']) for r in current)
        full=[r for r in current if r['path']=='full'];draft=[r for r in current if float(r['draft_path_ms'])>0]
        fallback=[r for r in current if r['category'].startswith('full_reject')]
        components=dict(full_ms=sum(float(r['full_path_ms'])-float(r['capture_ms']) for r in current),
            draft_ms=sum(float(r['draft_path_ms']) for r in current),
            verify_ms=sum(float(r['verify_path_ms']) for r in current),selector_collect_ms=collect)
        request_policy=sum(float(r['warm_policy_ms']) for r in current)
        residual=request_policy-sum(components.values())
        terminal_ms=policy_ms-request_policy
        oracle=policy_ms-collect-launch-terminal_ms
        # Terminal flush is charged in the reported evaluation metric, but can
        # also be removed in this optimistic zero-selector accounting exercise.
        wasted=sum(float(r['draft_path_ms'])+float(r['verify_path_ms']) for r in fallback)
        stats=dict(condition=condition,requests=len(current),actions=actions,fulls=len(full),drafts=len(draft),
            fallback_requests=len(fallback),full_fraction=len(full)/max(len(current),1),
            fallback_rate_per_draft=len(fallback)/max(len(draft),1),
            policy_ms_per_action=policy_ms/actions,baseline_ms_per_action=base_cost,
            observed_pooled_speedup=base_cost/(policy_ms/actions),
            zero_exposed_selector_speedup=base_cost/(oracle/actions),
            exposed_selector_wait_ms_per_action=waits/actions,
            selector_collect_ms_per_action=collect/actions,selector_launch_ms_per_action=launch/actions,
            terminal_flush_ms_per_action=terminal_ms/actions,
            wasted_rejected_proposal_ms_per_action=wasted/actions,
            full_mean_ms=float(np.mean([float(r['full_path_ms'])-float(r['capture_ms']) for r in full])),
            draft_mean_ms=float(np.mean([float(r['draft_path_ms']) for r in draft])) if draft else 0.,
            verify_mean_ms=float(np.mean([float(r['verify_path_ms']) for r in draft])) if draft else 0.,
            ratio_gt_one=sum(float(r['extrapolation_ratio'])>1 for r in draft if r['extrapolation_ratio']!=''),
            components_ms_per_action={k:v/actions for k,v in components.items()},
            other_request_cpu_ms_per_action=residual/actions)
        summary.append(stats)
        cc=Counter(r['category'] for r in current)
        for category,count in cc.items():categories.append(dict(condition=condition,category=category,count=count,fraction=count/len(current)))
    write_csv(OUT/'timing_diagnosis.csv',[{k:v for k,v in r.items() if k!='components_ms_per_action'} for r in summary])
    write_csv(OUT/'timing_categories.csv',categories)
    result=dict(summary=summary,categories=categories,
        note='Zero-exposed-selector is optimistic accounting with measured native path times, not an implemented policy or proof of a hardware lower bound.')
    atomic(OUT/'timing_diagnosis.json',result);return result


if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False))
