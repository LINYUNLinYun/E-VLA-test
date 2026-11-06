"""Same-boundary replay ratios; never substitute for closed-loop speedup."""
import argparse
import json
from pathlib import Path
from preparations.idea1_v3.analysis import write_csv


def compare(reference,candidate):
    for key in ('revision','input_sha256','thread_environment'):
        if reference[key]!=candidate[key]: raise ValueError(f'Different replay boundary: {key}')
    if reference['gpu']['uuid']!=candidate['gpu']['uuid']: raise ValueError('Different physical GPUs')
    if reference['config']['experiment']!=candidate['config']['experiment']: raise ValueError('No cross-backend ratios')
    if reference['measured_requests']!=candidate['measured_requests']: raise ValueError('Different request counts')
    return dict(reference=reference['config'].get('reference'),candidate=json.dumps(candidate['config'],sort_keys=True),
        request_ratio=reference['mean_request_ms']/candidate['mean_request_ms'],
        executed_action_ratio=reference['replay_ms_per_executed_action']/candidate['replay_ms_per_executed_action'],
        reference_ms_per_action=reference['replay_ms_per_executed_action'],candidate_ms_per_action=candidate['replay_ms_per_executed_action'],
        reference_requests=reference['measured_requests'],candidate_requests=candidate['measured_requests'],candidate_replay_actions=candidate['replay_executed_actions'],
        interpretation='fixed same-input virtual replay ratio only; not actual closed-loop route distribution or deployment speedup',
        reference_runtime_setup=reference['measured_requests_with_runtime_setup'],candidate_runtime_setup=candidate['measured_requests_with_runtime_setup'])


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--reference',type=Path,action='append',required=True); p.add_argument('--candidate',type=Path,action='append',required=True); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); rows=[]
    for rp in args.reference:
        r=json.loads((rp/'benchmark_summary.json').read_text())
        if not r['config'].get('reference'): raise ValueError('Explicit same-revision Full benchmark required')
        for cp in args.candidate:
            c=json.loads((cp/'benchmark_summary.json').read_text()); rows.append(dict(compare(r,c),reference_path=str(rp),candidate_path=str(cp)))
    args.output.parent.mkdir(parents=True,exist_ok=True); write_csv(args.output,rows)


if __name__=='__main__': main()
