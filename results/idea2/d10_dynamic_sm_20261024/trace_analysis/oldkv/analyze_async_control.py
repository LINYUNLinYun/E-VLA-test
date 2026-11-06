"""Optional CPU-only AE -> Vision -> VLM early-return control audit.

Independent of analyze_oldkv.py's three default traces. Missing control never
blocks that entrypoint. This script reads/exports completed profiles only.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess

from analyze_oldkv import ROOT, analyze, compare, save, signatures, write_csv

HERE = Path(__file__).resolve().parent
CONTROL = 'd10_async_serial_profile_20261024_v1'
OVERLAP = 'd10_oldkv_graph_encoder_overlap_20261024_v1'


def load_case(name, args, destination):
    sqlite = args.trace_dir / (name + '.sqlite')
    rep = args.trace_dir / (name + '.nsys-rep')
    reused = sqlite.exists()
    if not reused:
        if args.no_export or not rep.exists():
            return None, dict(name=name, missing=True, sqlite=str(sqlite), trace=str(rep))
        destination.mkdir(parents=True, exist_ok=True)
        with (destination/'export.log').open('w') as log:
            subprocess.run(['nsys','export','--type','sqlite','--output',str(sqlite),str(rep)],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
    report = analyze(sqlite, destination)
    return report, dict(name=name, missing=False, source=str(sqlite), export_reused=reused)


def subset(report, folder, mode, phase):
    groups = {k:v for k,v in report['groups'].items()
              if k.split('/')[1:3] == [phase, mode]}
    if not groups:
        raise ValueError(f'No measured group for {phase}/{mode}; observed {list(report["groups"])}')
    kernels=[]
    with (folder/'kernels.csv').open() as handle:
        for row in csv.DictReader(handle):
            if row['group'] not in groups or not row['stage'] or not row['request_id']:
                continue
            for key in ('start','end','host_end'):
                row[key]=int(row[key])
            row['duration_us']=float(row['duration_us'])
            row['geometry']=json.loads(row['geometry'])
            kernels.append(row)
    selected = dict(report)
    selected.update(groups=groups,
        requests=[r for r in report['requests'] if r['group'] in groups],
        request_modules=[r for r in report['request_modules'] if r['group'] in groups],
        signatures=signatures(kernels))
    return selected,kernels


def audit_control(report, kernels):
    requests=defaultdict(list)
    for row in kernels:
        requests[row['request_id']].append(row)
    rows=[]
    for request,selected in sorted(requests.items()):
        stages={stage:[r for r in selected if r['stage']==stage] for stage in ('AE','Vision','VLM')}
        present=all(stages.values())
        ae_end=max((r['end'] for r in stages['AE']),default=None)
        host_end=selected[0]['host_end']
        prefill=[r for r in selected if r['branch']=='Prefill']
        prefill_end=max((r['end'] for r in prefill),default=None)
        vision_start=min((r['start'] for r in stages['Vision']),default=None)
        vlm_start=min((r['start'] for r in stages['VLM']),default=None)
        vision_end=max((r['end'] for r in stages['Vision']),default=None)
        rows.append(dict(request_id=request,group=selected[0]['group'],all_three_stages_present=present,
            ae_completed_by_api_return=ae_end<=host_end if ae_end is not None else None,
            prefill_continues_after_api_return=prefill_end>host_end if prefill_end is not None else None,
            prefill_end_after_api_return_ms=(prefill_end-host_end)/1e6 if prefill_end is not None else None,
            prefill_kernels_after_api_return=sum(r['end']>host_end for r in prefill),
            vision_after_same_request_ae=vision_start>=ae_end if present else None,
            vision_after_ae_gap_us=(vision_start-ae_end)/1000 if present else None,
            vlm_after_same_request_vision=vlm_start>=vision_end if present else None,
            vlm_after_vision_gap_us=(vlm_start-vision_end)/1000 if present else None))
    pair_totals={name:sum(group['pairs'][name]['overlap_ms'] for group in report['groups'].values())
                 for name in ('AE_x_Vision','AE_x_VLM','AE_x_Prefill')}
    a=report['attribution']
    checks=dict(nonempty=bool(rows),all_requests_have_ae_vision_vlm=bool(rows) and all(r['all_three_stages_present'] for r in rows),
        ae_x_vision_exactly_zero=pair_totals['AE_x_Vision']==0,
        ae_x_vlm_exactly_zero=pair_totals['AE_x_VLM']==0,
        ae_x_all_prefill_exactly_zero=pair_totals['AE_x_Prefill']==0,
        ae_completed_by_api_return=bool(rows) and all(r['ae_completed_by_api_return'] for r in rows),
        every_request_prefill_continues_after_api_return=bool(rows) and all(r['prefill_continues_after_api_return'] for r in rows),
        vision_waits_same_request_ae=bool(rows) and all(r['vision_after_same_request_ae'] for r in rows),
        vlm_follows_same_request_vision=bool(rows) and all(r['vlm_after_same_request_vision'] for r in rows),
        graph_module_attribution_complete=a['all_scoped_graph_modules_mapped'],
        launch_correlation_complete=a['missing_launch_correlation']==0 and a['ambiguous_launch_correlation']==0)
    return dict(passed=all(checks.values()),checks=checks,requests=len(rows),pair_overlap_total_ms=pair_totals,
        requests_with_prefill_after_return=sum(bool(r['prefill_continues_after_api_return']) for r in rows),
        definition='Exact kernel-interval intersections across the whole selected group, including cross-request work. No epsilon hides small overlaps. Host return is the end of cewp request NVTX, not the GPU end. Ordering compares first/last measured model kernels, excludes copies/memsets and is not a substitute for Event/Lease source protocol verification.'),rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control',default=CONTROL)
    parser.add_argument('--overlap',default=OVERLAP)
    parser.add_argument('--control-mode',default='graph_async_serial')
    parser.add_argument('--overlap-mode',default='graph_encoder_overlap')
    parser.add_argument('--phase',default='continuous')
    parser.add_argument('--trace-dir',type=Path,default=ROOT/'nsys/idea2_experiment')
    parser.add_argument('--output',type=Path,default=HERE/'async_control_analysis')
    parser.add_argument('--no-export',action='store_true')
    parser.add_argument('--skip-missing',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    # Check presence before exporting either input. No waits and no GPU activity.
    missing=[name for name in dict.fromkeys((args.control,args.overlap))
             if not (args.trace_dir/(name+'.sqlite')).exists()
             and (args.no_export or not (args.trace_dir/(name+'.nsys-rep')).exists())]
    if missing:
        save(args.output/'receipt.json',dict(status='missing_input',missing=missing,cpu_only=True))
        if args.skip_missing:
            print(json.dumps(dict(status='skipped_missing_input',missing=missing)))
            return
        parser.error('Missing completed inputs: '+', '.join(missing))
    control_full,control_receipt=load_case(args.control,args,args.output/'control')
    overlap_full,overlap_receipt=load_case(args.overlap,args,args.output/'encoder_overlap')
    control,control_kernels=subset(control_full,args.output/'control',args.control_mode,args.phase)
    overlap,overlap_kernels=subset(overlap_full,args.output/'encoder_overlap',args.overlap_mode,args.phase)
    checks,requests=audit_control(control,control_kernels)
    comparison=compare(control,overlap)
    checks['signature_comparison']=dict(
        direction='graph_encoder_overlap elapsed / graph_async_serial elapsed',
        matched_types=len(comparison['matches']),baseline_unmatched_types=len(comparison['baseline_unmatched']),
        candidate_unmatched_types=len(comparison['candidate_unmatched']),
        count_mismatched_types=sum(r['baseline_calls']!=r['candidate_calls'] for r in comparison['matches']),
        exact_signatures_and_counts_match=bool(comparison['matches']) and not comparison['baseline_unmatched']
             and not comparison['candidate_unmatched']
             and all(r['baseline_calls']==r['candidate_calls'] for r in comparison['matches']))
    checks['matched_comparison_eligible']=checks['passed'] and checks['signature_comparison']['exact_signatures_and_counts_match']
    save(args.output/'control_checks.json',checks)
    write_csv(args.output/'control_requests.csv',requests)
    save(args.output/'signature_comparison.json',comparison)
    write_csv(args.output/'signature_comparison.csv',comparison['matches'])
    write_csv(args.output/'slowdown_summary.csv',comparison['summary'])
    save(args.output/'receipt.json',dict(cpu_only=True,control=control_receipt,encoder_overlap=overlap_receipt,
        selected_phase=args.phase,control_mode=args.control_mode,overlap_mode=args.overlap_mode,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    lines=['# Early-return serial control: CPU trace audit','',
        f"Control: `{args.control}` / `{args.control_mode}`; comparator: `{args.overlap}` / `{args.overlap_mode}`.",
        f"Control contract passed: **{checks['passed']}**. Exact signature/count comparison eligible: **{checks['matched_comparison_eligible']}**.",'',
        '| Check | Value |','|---|---|']
    lines += [f'| {k} | {v} |' for k,v in checks['checks'].items()]
    lines += ['',f"Actual AE×Vision: {checks['pair_overlap_total_ms']['AE_x_Vision']:.9f} ms; AE×VLM: {checks['pair_overlap_total_ms']['AE_x_VLM']:.9f} ms.",
        f"Prefill continues after API return: {checks['requests_with_prefill_after_return']}/{checks['requests']} measured requests.",
        f"Matched signatures: {len(comparison['matches'])}; unmatched control/overlap: {len(comparison['baseline_unmatched'])}/{len(comparison['candidate_unmatched'])}.",'',
        'The comparator/control ratio in slowdown_summary.csv is based on exact phase/module/kernel-name/geometry signatures. It is not an end-to-end speedup. Model output correctness, snapshot age, two-slot ownership and same-Graph capture owner must also pass the benchmark result checks.','',
        'Full GPU kernels are retained after the Host request ends. Vision means the complete prefill/vision Graph (including embedding/metadata), not pure Encoder arithmetic. No layer or iteration labels are inferred.']
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(checks),flush=True)


if __name__=='__main__':
    main()
