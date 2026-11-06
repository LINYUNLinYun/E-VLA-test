"""Read-only five-request camera-cycle costs with observed native work checks."""
import json
import math
from pathlib import Path
import statistics
from .common import sha,conditions
from .worker import validate_ae,validate_camera
from preparations.ease_insights_20261103.vlm.recovery import scheduled_nfe


def cost_records(root,manifest):
    services=[];excluded=[];configs={c['id']:c for c in conditions()};ids=list(configs)
    for path in sorted(Path(root).glob('*/runs/*/fixed_input_cost.json')):
        try:
            protocol,gate,cross,obj=[json.loads(p.read_text()) for p in
                (path.parent/'protocol.json',path.parent/'gate.json',path.parent/'cross_service.json',path)]
            if (any(v['manifest_hash']!=manifest['manifest_hash'] for v in (protocol,gate,obj))
                or not gate['passed'] or not cross['passed'] or not protocol['real_input_parity_passed']
                or not protocol['camera_numerical_gate_passed']):raise ValueError('Cost service lacks matching passed camera gates')
            if ([s['suite'] for s in obj['suites']]!=manifest['suites'] or obj['total_warmup_requests']!=160
                or obj['total_measured_requests']!=320):raise ValueError('Camera cost bank coverage differs')
            cycles={cid:[] for cid in ids};stages={cid:dict(full=[],fast=[]) for cid in ids}
            for suite in obj['suites']:
                if sorted(b['condition'] for b in suite['conditions'])!=sorted(ids):raise ValueError('Camera cost condition coverage differs')
                noise=[]
                for block in suite['conditions']:
                    cid=block['condition'];cfg=configs[cid];warm=block['warmup'];rows=block['measured']
                    if (block['config']!=cfg or len(warm)!=10 or len(rows)!=20
                        or [r['round'] for r in warm+rows]!=list(range(30))):raise ValueError('Camera cost warmup/measurement protocol differs')
                    noise.append([(r['seed'],r['model_noise_sha256']) for r in warm+rows])
                    for r in warm+rows:
                        index=r['round'];full=cid=='full_d10' or index%5==0
                        source=index*5 if cid=='full_d10' else (index//5)*25;age=index*5-source
                        timing=r['model_timing'];nfe=10 if full else 5
                        fake=dict(route=dict(ae_execution_audit=timing['ae_execution_audit'],ae_bypassed_layers=[],
                            camera_execution_audit=r['camera_execution_audit']),timing=timing)
                        validate_ae(fake,[],nfe);validate_camera(fake,cfg,full,source,age)
                        if (r['refresh']!=full or not math.isfinite(r['policy_wall_ms']) or r['policy_wall_ms']<=0
                            or scheduled_nfe(timing['nodes'])!=nfe):raise ValueError('Camera cost stage, native NFE or latency differs')
                        previous=source-5 if cid=='full_d10' else source-25
                        if r['history_steps']!=([source] if previous<0 else [previous,source]):raise ValueError('Cost own-Full source history differs')
                    for r in rows:stages[cid]['full' if r['refresh'] else 'fast'].append(r['policy_wall_ms'])
                    cycles[cid].extend(sum(r['policy_wall_ms'] for r in rows[i:i+5]) for i in range(0,20,5))
                if any(n!=noise[0] for n in noise):raise ValueError('Camera cost paired seeds/noise differ')
            means={cid:statistics.mean(v) for cid,v in cycles.items()}
            services.append(dict(path=str(path),sha256=sha(path),gpu_uuid=protocol['gpu_uuid'],cycle_mean_ms=means,
                full_over_policy_cycle={cid:means['full_d10']/v for cid,v in means.items()},
                stage_policy_mean_ms={cid:{s:statistics.mean(v) if v else None for s,v in group.items()} for cid,group in stages.items()}))
        except (ValueError,KeyError,OSError,TypeError) as error:excluded.append(dict(path=str(path),reason=str(error)))
    means={cid:statistics.mean(s['cycle_mean_ms'][cid] for s in services) for cid in ids} if services else {}
    return dict(valid_services=len(services),services=services,excluded_or_pending=excluded,mean_cycle_ms=means,
        full_over_policy_cycle={cid:means['full_d10']/v for cid,v in means.items()},
        pooling='Equal completed-service weight; four equally weighted old-bank suite inputs; whole five-request cycles',
        scope='Native policy cost including cache copy/dispatch and output transfer; actual AE18/NFE and camera batch checked; RPC/environment excluded; shared GPU timing is descriptive')
