"""Periodic latest-available camera replay with one admitted lookahead input.

Unlike the legacy capacity harness, lookahead selects the LATEST arrived frame,
not source+1 from a growing historical FIFO. Unselected arrivals are explicitly
counted as drops; one newest unprocessed frame may remain buffered at the end.
"""
import argparse
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--bank',required=True)
    parser.add_argument('--checkpoint',default='shared/checkpoints/openpi/pi05_libero_pytorch_idea2')
    parser.add_argument('--gpu-uuid',default='GPU-599477fb-0507-40dc-14df-d491f7a7b202')
    parser.add_argument('--conditions',nargs='+',required=True)
    parser.add_argument('--continuous',type=int,default=100)
    parser.add_argument('--warmup',type=int,default=20)
    parser.add_argument('--arrival-period-ms',type=float,required=True)
    parser.add_argument('--reverse',action='store_true')
    # Accepted matrix fields; this driver has no paired-capacity trajectory.
    parser.add_argument('--pairs',type=int,default=0)
    parser.add_argument('--length',type=int,default=8)
    parser.add_argument('--verify-count',type=int,default=1)
    args=parser.parse_args()
    assert args.arrival_period_ms>0 and args.continuous>0 and args.pairs==0
    from preparations.idea2.layer_bench import require_idle_gpu,gpu_snapshot,require_same_processes,statistics,within_tolerance
    from preparations.idea2.progressive_kv_bench import source_hashes
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=False)
    result=dict(status='starting',args=vars(args),source_before=source_hashes(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        command=shlex.join([sys.executable,'-m','results.idea2.progressive_kv_20261025.slo_bench',*sys.argv[1:]]),
        performance_eligible=False,measurement_scope='Continuous outputs per camera condition; independent reverse run',
        queue_policy='Latest available at admission; one admitted prepared/encoded lookahead, never future input',
        gpu_before=require_idle_gpu(args.gpu_uuid),bank_sha256=hashlib.sha256(Path(args.bank).read_bytes()).hexdigest(),
        rows=[],groups=[],summary={})
    def save():(dest/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    save();engine=None
    try:
        import torch
        from preparations.idea2.experiments import load_backend,native_raw
        from preparations.idea2.progressive_kv import ProgressiveRuntime
        backend=load_backend(args.checkpoint,model_config='pi05_libero',num_steps=10)
        records=pickle.loads(Path(args.bank).read_bytes())
        if len({row['instruction'] for row in records})!=1:
            raise RuntimeError('Same-task bank required')
        raw=[native_raw(x) for x in records]
        prepared=[backend.prepare(x) for x in raw]
        masks=[[x.detach().cpu().numpy() for x in (*p.image_masks,p.token_masks)] for p in prepared]
        if any(not all(np.array_equal(a,b) for a,b in zip(masks[0],m,strict=True)) for m in masks):
            raise RuntimeError('Fixed prefix mask contract failed')
        result['same_task_and_mask_verified']=True
        noises=[backend.noise(1701+i) for i in range(len(records))]
        engine=ProgressiveRuntime(backend,prepared[0],noises[0])
        result.update(identity=backend.identity,graph_setup_ms=engine.graphs.setup_ms,kv_addresses=engine.graphs.kv_addresses)
        own={x['pid'] for x in gpu_snapshot(args.gpu_uuid)['processes']};assert len(own)==1
        conditions=[(int(c.split('/')[0]),c.split('/')[1]) for c in args.conditions]
        if args.reverse:conditions.reverse()
        for lead,variant in conditions:
            assert lead in (0,1,3,6,18) and variant in ('encoder','host_only','no_lookahead')
            engine.reset(prepared[-1])
            for i in range(args.warmup):
                _,row,_=engine.request(prepared[i%len(records)],noises[i%len(records)],lead=lead,collect_events=False)
                engine.finish_record(row)
            for i in range(args.verify_count):
                _,row,_=engine.request(prepared[(i+1)%len(records)],noises[(i+1)%len(records)],lead=lead,verify=True)
                if not within_tolerance(row['numerical']['matched_schedule'],1e-5,1e-6):raise RuntimeError('Serial schedule oracle failed')
            engine.reset(prepared[-1])
            torch.cuda.reset_peak_memory_stats()
            period=args.arrival_period_ms/1000
            clock=time.perf_counter()
            previous_source=-1
            last_admitted=-1
            current=None
            rows=[]
            def select_latest(wait):
                nonlocal last_admitted
                while True:
                    now=time.perf_counter()
                    source=int((now-clock)/period)
                    if source>last_admitted:
                        admission=now
                        p=backend.prepare(raw[source%len(raw)])
                        last_admitted=source
                        return dict(source=source,capture=clock+source*period,admitted=admission,prepared=p)
                    if not wait:return None
                    time.sleep(min(.002,max(.0001,clock+(last_admitted+1)*period-now)))
            start=clock
            for index in range(args.continuous):
                if current is None:current=select_latest(True)
                holder=[]
                def lookahead():
                    selected=select_latest(False)
                    if selected is None:return None
                    holder.append(selected)
                    return selected['prepared']
                source=current['source']
                output,row,nxt=engine.request(current['prepared'],noises[source%len(noises)],lead=lead,
                    lookahead=lookahead if variant!='no_lookahead' and index+1<args.continuous else None,
                    next_encoder=variant=='encoder',collect_events=False)
                decode_start=time.perf_counter()
                state=current['prepared'].state[0].detach().cpu().numpy()
                decoded=backend.policy._output_transform(dict(state=state,actions=output.copy()))['actions']
                if np.asarray(decoded).shape!=(10,7) or not np.isfinite(decoded).all():
                    raise RuntimeError('Invalid decoded action chunk')
                row['decode_host_ms']=(time.perf_counter()-decode_start)*1000
                end=time.perf_counter()
                assert current['capture']<=current['admitted']<=end
                row.update(phase='continuous',variant=variant,index=index,source_id=source,
                    old_source_id=previous_source,new_source_id=source,noise_seed=1701+source%len(noises),
                    captured_host=current['capture'],admitted_host=current['admitted'],output_host=end,
                    capture_to_action_ms=(end-current['capture'])*1000,
                    admission_to_action_ms=(end-current['admitted'])*1000,
                    current_kv_source_age_ms=(end-current['capture'])*1000,
                    old_source_is_bootstrap=previous_source<0,
                    old_kv_consumed=any(f<18 for f in row['frontiers']),
                    old_kv_source_age_ms=((end-(clock+previous_source*period))*1000
                                          if previous_source>=0 and any(f<18 for f in row['frontiers']) else None),
                    old_source_gap_frames=source-previous_source if previous_source>=0 else None,
                    dropped_before_admission=source-previous_source-1,
                    camera_age_frames_at_output=int((end-clock)/period)-source)
                previous_source=source
                rows.append(row)
                if holder:
                    assert nxt is holder[0]['prepared']
                    current=holder[0]
                else:
                    assert nxt is None
                    current=None
            engine.drain();finish=time.perf_counter()
            received=int((finish-clock)/period)+1
            buffered=int(received>last_admitted+1)
            dropped=received-len(rows)-buffered
            assert dropped>=sum(r['dropped_before_admission'] for r in rows)>=0
            for row in rows:engine.finish_record(row)
            key=f'{lead}/{variant}'
            group=dict(lead=lead,variant=variant,count=len(rows),all_work_ms=(finish-start)*1000,
                all_work_hz=len(rows)/(finish-start),received=received,admitted=len(rows),dropped=dropped,
                dropped_before_admission=sum(r['dropped_before_admission'] for r in rows),
                buffered_unprocessed=buffered,full_kv_builds=len(rows),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())
            assert received==len(rows)+dropped+buffered
            result['groups'].append(group);result['rows'].extend(rows)
            result['summary'][key]=dict(group)
            for metric in ('capture_to_action_ms','admission_to_action_ms','current_kv_source_age_ms','old_kv_source_age_ms',
                           'ae_ms','vlm_ms','exposed_ae_after_vlm_ms','ae_finish_minus_next_encoder_ms'):
                result['summary'][key][metric]=statistics([r.get(metric) for r in rows])
            print(key,'Hz',round(group['all_work_hz'],3),'P95',round(result['summary'][key]['capture_to_action_ms']['p95'],3),
                  'received/drop/buffer',received,dropped,buffered,flush=True)
            save()
        require_same_processes(args.gpu_uuid,own)
        result['source_after']=source_hashes();result['source_unchanged']=result['source_after']==result['source_before']
        if not result['source_unchanged']:raise RuntimeError('Runtime sources changed')
        result['status']='passed'
        result['performance_eligible']=args.continuous>=100
    except BaseException:
        result.update(status='failed',error=traceback.format_exc());raise
    finally:
        if engine:engine.close()
        save()


if __name__=='__main__':main()
