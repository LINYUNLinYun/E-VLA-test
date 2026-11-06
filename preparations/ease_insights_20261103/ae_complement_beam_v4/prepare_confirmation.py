"""Freeze both360-rollout streams together before releasing either request."""
import copy
import json
import time
from pathlib import Path
from .common import *
from .protocol import REPORT

PACKAGE=Path(__file__).resolve().parent
BUNDLE=REPORT/'confirmation_freeze.json'
ROOTS=(FAMILY_ROOT/'confirmation',FAMILY_ROOT)


def read_bundle(path=BUNDLE,verify_sources=True):
    value=json.loads(Path(path).read_text());payload=dict(value);declared=payload.pop('bundle_hash')
    require(digest(payload)==declared and value['revision']=='ease-cross-module-confirmation-bundle-v4','Confirmation bundle changed')
    require(value['total_new_rollouts']==720 and value['unique_physical_states_per_algorithm']==120
        and value['preregistered_hypotheses']==DIRECTIONS and len(value['streams'])==2,'Confirmation bundle design differs')
    for repeat,(item,root) in enumerate(zip(value['streams'],ROOTS)):
        path=root/'manifest.json';require(item['noise_repeat']==repeat and item['output_root']==str(root)
            and item['manifest_file_sha256']==sha(path),'Stream manifest byte identity differs')
        m=read_manifest(path,verify_sources=verify_sources)
        require(m['manifest_hash']==item['manifest_hash'] and m['noise_repeat']==repeat
            and m['preregistered_hypotheses']==DIRECTIONS,'Stream canonical identity differs')
    require(sha(value['protocol_evidence']['path'])==value['protocol_evidence']['sha256'],'Protocol changed')
    return value


def prepare():
    if BUNDLE.exists():return dict(reused=True,bundle=read_bundle(),bundle_file_sha256=sha(BUNDLE),GPU_started=False)
    require(not any((r/'manifest.json').exists() for r in ROOTS),'Partial freeze exists; inspect before continuing')
    base=read_manifest(FAMILY_ROOT/'calibration/manifest.json',verify_sources=True)
    accepted=REPORT/'calibration_acceptance.json';acceptance=json.loads(accepted.read_text())
    require(acceptance['status']=='pass' and acceptance['manifest_hash']==base['manifest_hash']
        and acceptance['all_twelve_controls_bitwise_identical'] and acceptance['joint_cost_qualification']['pass'],
        'Need twelve exact bridges and accepted joint cost qualification')
    # One inventory for BOTH planned noise streams, before either creates outcome files.
    from preparations.ease_insights_20261103.vlm_diversity_v2.common import read_manifest as read_old
    from preparations.ease_insights_20261103.vlm_diversity_v2.prepare import unused
    original=read_old(ROOT/'results/ease_insights_20261103/vlm_diversity_v2/discovery/manifest.json',verify_sources=True)
    prior=unused(original,[47,48,49])
    for row in base['official_init_availability']['cases']:
        require(sha(row['bddl_path'])==row['bddl_sha256'] and sha(row['init_file'])==row['init_file_sha256'],
            'Official physical asset changed')
        require(all(str(i) in row['init_sha256'] for i in (47,48,49)),'Official fresh initial state missing')
    local={str(p.relative_to(ROOT)):sha(p) for p in sorted(PACKAGE.glob('*.py'))}
    sources=dict(base['source_files']);sources.update(local)
    for p in sorted(REPORT.glob('*.py')):sources[str(p.relative_to(ROOT))]=sha(p)
    streams=[];manifests=[]
    for repeat,output in enumerate(ROOTS):
        m=copy.deepcopy(base)
        for key in ('manifest_hash','calibration_inits','total_calibration_episodes','calibration_case_scope','confirmation_initial_state_prior_use_audit'):
            m.pop(key,None)
        m.update(revision='ease-cross-module-confirmation-v4',phase='cross_module_confirm_v4',output_root=str(output),
            tasks=list(range(10)),case_table=[dict(suite=s,task=t) for s in SUITES for t in range(10)],
            confirmation_inits=[47,48,49],cases_per_condition=120,total_confirmation_episodes=360,noise_repeat=repeat,
            preregistered_hypotheses=copy.deepcopy(DIRECTIONS),calibration_evidence=dict(path=str(accepted),sha256=sha(accepted)),
            local_runtime_sources=local,analysis_sources=local,source_files=sources,
            confirmation_initial_state_prior_use_audit=copy.deepcopy(prior),
            heldout_preparation=dict(outcomes_inspected_before_freeze=False,official_initials=[47,48,49],
                two_streams_frozen_together=True,policy_noise_repeats=[0,1],physical_states_shared_across_noise=True,
                total_new_rollouts=720,unique_physical_states_per_algorithm=120,rollouts_per_algorithm=240,
                primary_clusters_per_suite=10,observations_per_task=6,controls_reused_for_confirmation=False),
            policy_noise_design=dict(rule='repeat0 original;repeat1 SHA256-derived condition-independent seed',
                explicit_model_noise=True,environment_seed=7,order_dependent=False),
            initial_pairing='Each stream pairs all3 algorithms on the same physical observation and explicit noise; streams share physical states but use different policy noise')
        m['manifest_hash']=digest(m);validate_output(output,m);manifests.append((output,m))
    # Validation and freeze are complete for both designs before the first is written.
    for output,m in manifests:
        atomic_json(output/'manifest.json',m);read_manifest(output/'manifest.json',verify_sources=True)
        streams.append(dict(noise_repeat=m['noise_repeat'],output_root=str(output),manifest_hash=m['manifest_hash'],
            manifest_file_sha256=sha(output/'manifest.json'),new_rollouts=360))
    value=dict(revision='ease-cross-module-confirmation-bundle-v4',time=time.time(),streams=streams,
        total_new_rollouts=720,unique_physical_states_per_algorithm=120,rollouts_per_algorithm=240,
        preregistered_hypotheses=copy.deepcopy(DIRECTIONS),protocol_evidence=base['protocol_evidence'],
        calibration_evidence=dict(path=str(accepted),sha256=sha(accepted)),
        confirmation_initial_state_prior_use_audit=prior,outcomes_inspected_before_freeze=False,
        primary='Ten whole-task clusters with6 paired observations each; exact signflip andHolm2',
        source_files_frozen=len(sources),GPU_started=False)
    value['bundle_hash']=digest(value);atomic_json(BUNDLE,value);read_bundle()
    return dict(reused=False,bundle=value,bundle_file_sha256=sha(BUNDLE),GPU_started=False)


if __name__=='__main__':print(json.dumps(prepare()))
