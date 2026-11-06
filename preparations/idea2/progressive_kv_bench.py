"""Bounded progressive-KV ablation; no production default is changed."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import shlex
import sys
import time
import traceback

import numpy as np


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = list((root/'preparations/idea2').glob('progressive_kv*.py'))
    paths += [root/'preparations/idea2'/n for n in ('layer_graphs.py','layer_scheduler.py','runtime_graph.py','graph_sampler.py','cewp_preparation.py','cewp_segmented.py','openpi_backend.py')]
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if not p.name.startswith('test_')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu-uuid', default='GPU-599477fb-0507-40dc-14df-d491f7a7b202')
    parser.add_argument('--checkpoint', default='shared/checkpoints/openpi/pi05_libero_pytorch_idea2')
    parser.add_argument('--bank', default='results/idea1_v3_1/input_bank.pkl')
    parser.add_argument('--leads', type=int, nargs='+', default=[0,1,3,6,18])
    parser.add_argument('--variants', nargs='+', choices=['no_lookahead','host_only','encoder'], default=['host_only','encoder'])
    parser.add_argument('--pairs', type=int, default=20)
    parser.add_argument('--length', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--continuous', type=int, default=0)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--profile-lead', type=int, default=3)
    parser.add_argument('--profile-variant', default='encoder', choices=['host_only','encoder','no_lookahead'])
    parser.add_argument('--verify-count', type=int, default=2)
    parser.add_argument('--force-old', action='store_true')
    parser.add_argument('--conditions', nargs='+', help='Explicit lead/variant pairs, e.g. 3/encoder 18/host_only')
    parser.add_argument('--arrival-period-ms', type=float, default=0)
    args = parser.parse_args()
    from .layer_bench import require_idle_gpu, require_same_processes, gpu_snapshot, statistics, within_tolerance
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=False)
    result = dict(status='starting', args=vars(args), source_before=source_hashes(),
                  command=shlex.join([sys.executable,'-m','preparations.idea2.progressive_kv_bench',*sys.argv[1:]]),
                  rows=[], groups=[], performance_eligible=not args.profile and args.pairs>=100,
                  semantics='Full pi05 D10/H10; per-round frozen new-KV frontier, old complete KV fallback. Not fresh-equivalent.',
                  tolerance=dict(max_abs=1e-5,mae=1e-6), bank_sha256=hashlib.sha256(Path(args.bank).read_bytes()).hexdigest(),
                  environment={k:os.environ.get(k) for k in ('CUDA_DEVICE_MAX_CONNECTIONS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})
    def save():
        (dest/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    result['gpu_before']=require_idle_gpu(args.gpu_uuid)
    save()
    engine = None
    try:
        import torch
        from .experiments import load_backend, native_raw
        from .progressive_kv import ProgressiveRuntime
        records=pickle.loads(Path(args.bank).read_bytes())
        backend=load_backend(args.checkpoint,model_config='pi05_libero',num_steps=10)
        raw=[native_raw(x) for x in records]
        prepared=[backend.prepare(x) for x in raw]
        noise=[backend.noise(1701+i) for i in range(len(records))]
        engine=ProgressiveRuntime(backend,prepared[0],noise[0],profile=args.profile)
        result.update(identity=backend.identity, graph_setup_ms=engine.graphs.setup_ms,
                      kv_addresses=engine.graphs.kv_addresses,
                      scratch_bytes=sum(x.numel()*x.element_size() for x in engine.graphs.shared_scratch))
        own={x['pid'] for x in gpu_snapshot(args.gpu_uuid)['processes']}
        assert len(own)==1
        result['own_gpu_processes']=sorted(own)
        conditions=[(lead,variant) for lead in args.leads for variant in args.variants]
        if args.conditions:
            conditions=[(int(value.split('/')[0]),value.split('/')[1]) for value in args.conditions]
            if any(l not in (0,1,3,6,18) or v not in ('encoder','host_only','no_lookahead') for l,v in conditions):
                raise ValueError('Invalid explicit condition')
        if args.reverse: conditions.reverse()
        def prepare(index):
            return backend.prepare(raw[index%len(raw)])
        def sequence(lead,variant,length,phase,pair,start_index=0,verify=False):
            # Identical bootstrap and sources/noise for every arm in this pair.
            engine.reset(prepared[start_index%len(prepared)])
            current=None
            group_rows=[]
            clock=time.perf_counter()
            next_source=start_index+1
            dropped=0
            start=time.perf_counter()
            for i in range(length):
                admitted=time.perf_counter()
                if args.arrival_period_ms:
                    period=args.arrival_period_ms/1000
                    if current is None:
                        available=max(next_source,int((admitted-clock)/period)+start_index+1)
                        delay=clock+(available-start_index-1)*period-admitted
                        if delay>0:time.sleep(delay)
                        dropped+=max(0,available-next_source)
                        next_source=available
                source=next_source
                prepared_begin=time.perf_counter()
                if current is None:current=prepare(source)
                prepare_ms=(time.perf_counter()-prepared_begin)*1000
                ni=source%len(noise)
                future=source+1
                def lookahead():
                    if args.arrival_period_ms:
                        arrived=int((time.perf_counter()-clock)/(args.arrival_period_ms/1000))+start_index+1
                        if arrived<future:return None
                    return prepare(future)
                want_next=i+1<length and variant!='no_lookahead'
                output,row,nxt=engine.request(current,noise[ni],lead=lead,force_old=args.force_old,
                    lookahead=lookahead if want_next else None,next_encoder=variant=='encoder',
                    verify=verify,collect_events=False)
                row.update(phase=phase,pair=pair,index=i,variant=variant,source_id=source,noise_seed=1701+ni,
                           input_prepare_ms=prepare_ms,admission_to_action_ms=(time.perf_counter()-admitted)*1000,
                           capture_to_action_ms=((time.perf_counter()-(clock+(source-start_index-1)*args.arrival_period_ms/1000))*1000
                                                 if args.arrival_period_ms else None),
                           input_sha256=hashlib.sha256(pickle.dumps(raw[ni],protocol=5)).hexdigest())
                if verify:
                    assert within_tolerance(row['numerical']['matched_schedule'],1e-5,1e-6),row
                group_rows.append(row)
                current=nxt
                next_source=source+1
            engine.drain()
            end=time.perf_counter()
            for row in group_rows:engine.finish_record(row)
            elapsed=(end-start)*1000
            group=dict(phase=phase,pair=pair,lead=lead,variant=variant,count=length,
                       all_work_ms=elapsed,all_work_hz=length*1000/elapsed,dropped=dropped,
                       tickets=engine.pool.ticket_diagnostics['counts'])
            if phase!='warmup':
                result['rows'].extend(group_rows)
                result['groups'].append(group)
            return group
        result['status']='warming';save()
        # Exercise both captured slot addresses and 0/full frontier policies.
        for lead,variant in conditions:
            if args.warmup:sequence(lead,variant,args.warmup,'warmup',-1)
            if args.verify_count:sequence(lead,variant,args.verify_count,'verification',-1,verify=True)
        require_same_processes(args.gpu_uuid,own)
        torch.cuda.reset_peak_memory_stats()
        result['status']='measuring';save()
        if args.profile:
            torch.cuda.cudart().cudaProfilerStart()
            sequence(args.profile_lead,args.profile_variant,args.length,'profile',0)
            torch.cuda.cudart().cudaProfilerStop()
        else:
            for pair in range(args.pairs):
                ordered=conditions if pair%2==0 else conditions[::-1]
                for lead,variant in ordered:
                    g=sequence(lead,variant,args.length,'paired',pair,start_index=pair%len(raw))
                    print('pair',pair,'lead',lead,variant,round(g['all_work_ms'],3),flush=True)
                save()
            if args.continuous:
                for lead,variant in conditions:
                    sequence(lead,variant,args.continuous,'continuous',0)
        result['gpu_after']=require_same_processes(args.gpu_uuid,own)
        result['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        result['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        result['source_after']=source_hashes()
        result['source_unchanged']=result['source_after']==result['source_before']
        if not result['source_unchanged']:raise RuntimeError('Runtime sources changed during experiment')
        result['summary']={}
        for lead,variant in conditions:
            selected=[x for x in result['rows'] if x['lead']==lead and x['variant']==variant and x['phase']!='verification']
            groups=[g for g in result['groups'] if g['lead']==lead and g['variant']==variant and g['phase']=='paired']
            key=f'lead{lead}/{variant}'
            result['summary'][key]=dict(samples=len(selected),trajectory_ms=statistics([g['all_work_ms'] for g in groups]))
            for metric in ('admission_to_action_ms','cpu_cpu_ms','vision_ms','vlm_ms','ae_ms','exposed_ae_after_vlm_ms',
                           'model_after_encoder_ms','ae_finish_minus_next_encoder_ms','ae_start_after_vlm_start_ms'):
                result['summary'][key][metric]=statistics([x.get(metric) for x in selected])
            result['summary'][key]['mean_new_layers_by_step']=np.mean([x['frontiers'] for x in selected],axis=0).tolist() if selected else None
        result['status']='passed'
    except BaseException:
        result.update(status='failed',error=traceback.format_exc())
        raise
    finally:
        if engine is not None:engine.close()
        save()
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':
    main()
