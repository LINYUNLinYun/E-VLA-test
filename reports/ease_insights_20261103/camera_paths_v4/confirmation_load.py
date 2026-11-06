"""Complete-only loading of800 freshly executed within-family holdout rollouts."""
import copy
import importlib.util
from pathlib import Path
from preparations.ease_insights_20261103.camera_paths_v4 import common,worker
from preparations.ease_insights_20261103.vlm.analyze import load_episodes
from preparations.ease_insights_20261103.vlm.analyze_confirmation import audit_identities,unexpected_cases


def load_stage(root,stage):
    root=Path(root).resolve();m=common.read_manifest(root/'manifest.json',verify_sources=True)
    common.require(stage=='confirm' and common.stage_for(m)=='confirm','Wrong confirmation stage')
    common.require(len(list(root.glob('*/confirm_shard*of02_complete.json')))==8
        and len(list(root.glob('*/episodes/*/task*/init*/episode_receipt.json')))==800,'No partial confirmation outcome analysis')
    scenes=common.cases(m,'confirm');view=copy.deepcopy(m);view['discovery_inits']=m['confirmation_inits']
    records,missing,invalid,evidence=load_episodes(root,view);invalid.extend(audit_identities(root,m,records))
    tool=common.ROOT/'reports/ease_insights_20261103/common_case_audit_v1/audit.py'
    common.require(common.sha(tool)=='2fec7afe6e80cca6286e52a467d42be9e525f6125da76d0b63873985b4a6d113','Readonly validator changed')
    spec=importlib.util.spec_from_file_location('fresh_camera_confirmation_validator',tool);validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
    for cid,k in list(records):
        try:validator.case(common,worker,root,m,cid,records[cid,k]['case'])
        except (ValueError,RuntimeError,KeyError,OSError) as error:
            invalid.append(dict(condition=cid,case=records[cid,k]['case'],error=repr(error)));del records[cid,k]
    unexpected=unexpected_cases(root,m,scenes)
    return dict(root=root,manifest=m,records=records,scenes=scenes,missing=missing,invalid=invalid,unexpected=unexpected,
        episode_evidence=evidence,complete=len(records)==800 and not missing and not invalid and not unexpected)
