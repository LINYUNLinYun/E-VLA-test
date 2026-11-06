"""Three-request cost records plus actual AE/NFE and exact Euler-node checks."""
import json
from pathlib import Path
import statistics
from preparations.ease_insights_20261103.vlm.analyze import cost_records as original_cost_records
from .worker import validate_ae
from .nodes import scheduled_nfe


def cost_records(root,manifest):
    raw=original_cost_records(Path(root),manifest);good=[];bad=list(raw['excluded_or_pending'])
    for service in raw['services']:
        try:
            data=json.loads(Path(service['path']).read_text())
            for suite in data['suites']:
                for block in suite['conditions']:
                    cfg=next(c for c in manifest['conditions'] if c['id']==block['condition'])
                    if block['config']!=cfg:raise ValueError('Cost configuration differs')
                    for r in block['warmup']+block['measured']:
                        full=cfg['id']=='full_d10' or r['round']%3==0;nfe=10 if full else cfg['D'];skip=[] if full else cfg['ae_skip']
                        timing=r['model_timing'];fake=dict(timing=timing,route=dict(ae_execution_audit=timing['ae_execution_audit'],ae_bypassed_layers=skip))
                        validate_ae(fake,skip,nfe)
                        if scheduled_nfe(timing['nodes'])!=nfe:raise ValueError('Cost Euler time nodes differ')
            good.append(service)
        except (ValueError,KeyError,OSError,TypeError) as error:bad.append(dict(path=service['path'],reason=str(error)))
    ids=[c['id'] for c in manifest['conditions']];means={c:statistics.mean(s['cycle_mean_ms'][c] for s in good) for c in ids} if good else {}
    return dict(valid_services=len(good),services=good,excluded_or_pending=bad,mean_cycle_ms=means,
        full_over_policy_cycle={c:means['full_d10']/v for c,v in means.items()},
        pooling='Equal completed-service weight and equally weighted four-suite old-bank inputs; full three-request cycles',
        scope='Actual original-adapter policy cost, ordered AE visits and exact time-node receipts; not closed-loop or exclusive-GPU acceleration')
