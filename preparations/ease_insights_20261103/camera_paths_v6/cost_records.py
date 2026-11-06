"""Validate actual fresh-vision/compact-prefix/AE dispatch for completed native cost probe calls."""
import json
from pathlib import Path
from preparations.ease_insights_20261103.vlm.analyze import cost_records as original_cost_records
from .worker import validate_ae,validate_camera
from .nodes import scheduled_nfe


def cost_records(root,manifest):
    raw=original_cost_records(Path(root),manifest);good=[];bad=list(raw['excluded_or_pending'])
    for service in raw['services']:
        try:
            path=Path(service['path']);data=json.loads(path.read_text())
            if json.loads((path.parent/'result.json').read_text()).get('complete') is not True:raise ValueError('Cost service incomplete')
            for suite in data['suites']:
                for block in suite['conditions']:
                    cfg=next(c for c in manifest['conditions'] if c['id']==block['condition'])
                    if cfg!=block['config']:raise ValueError('Cost configuration differs')
                    for r in block['warmup']+block['measured']:
                        full=cfg['id']=='full_d10' or r['round']%3==0;nfe=10 if full else 2;t=r['model_timing']
                        row=dict(timing=t,route=dict(ae_execution_audit=t['ae_execution_audit'],ae_bypassed_layers=[],camera_execution_audit=t['camera_execution_audit'],prefix_pattern='all' if full else cfg['prefix_pattern'],prefix_execution_audit=t['prefix_execution_audit']))
                        validate_ae(row,[],nfe);validate_camera(row,cfg,full)
                        if scheduled_nfe(t['nodes'])!=nfe:raise ValueError('Cost native nodes differ')
            good.append(service)
        except (OSError,ValueError,KeyError,TypeError) as error:bad.append(dict(path=service['path'],reason=str(error)))
    return dict(valid_services=len(good),services=good,excluded_or_pending=bad,
        scope='Actual native fixed-input policy cost including both fresh camera pipelines;not closed-loop speed or quality')
