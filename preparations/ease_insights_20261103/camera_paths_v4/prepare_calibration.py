"""Freeze8 known controls after21+6 CPU contracts;no model or simulator launched here."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from .common import *
from .protocol import REPORT,freeze
PACKAGE=Path(__file__).resolve().parent
OUTPUT=FAMILY_ROOT/'calibration'


def official_assets():
    from preparations.ease_insights_20261103.baseline_recovery_v1.prepare import official_assets as original
    assets=original()
    from libero.libero import benchmark
    from preparations.idea1_v2_1.runtime_client import official_init_states
    import numpy as np
    selected={s:benchmark.get_benchmark_dict()[s]() for s in SUITES}
    for row in assets['cases']:
        vectors=official_init_states(selected[row['suite']],row['task'])
        row['init_sha256']={str(i):hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for i,v in enumerate(vectors)}
        if len({row['init_sha256'][str(i)] for i in (10,11,12,13,14)})!=5:raise ValueError('Confirmation vectors are not distinct within task')
    assets.update(calibration_init=45,discovery_inits=[45,46],reserved_confirmation_inits=[10,11,12,13,14],
        scope='OfficialCPUvectorsandassets;holdoutrelativewithinnewfamily,notgloballyunusedstates.No model/simulator/CUDA context.')
    return assets


def prepare():
    target=OUTPUT/'manifest.json'
    if target.exists():
        m=read_manifest(target,verify_sources=True);return dict(reused=True,manifest_hash=m['manifest_hash'],manifest_file_sha256=sha(target))
    from preparations.ease_insights_20261103.ae_complement_beam_v4.common import read_manifest as read_previous
    base=read_previous(ROOT/'results/ease_insights_20261103/ae_complement_beam_v4/confirmation/manifest.json',verify_sources=True)
    for alias in REFERENCES:reference(alias,verify_sources=True)
    pin=freeze();protocol=json.loads(Path(pin['path']).read_text())
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    proc=subprocess.run([str(ENV_PYTHON),'-B','-m','preparations.ease_insights_20261103.camera_paths_v4.prepare_calibration','--assets-only'],cwd=ROOT,env=env,text=True,capture_output=True,check=True)
    lines=[line for line in proc.stdout.splitlines() if line.startswith('{')];require(bool(lines),'CPU asset helper did not return JSON');assets=json.loads(lines[-1])
    for old,new in zip(base['official_init_availability']['cases'],assets['cases']):
        require(all(old[k]==new[k] for k in ('suite','task','prompt','bddl_path','bddl_sha256','init_file','init_file_sha256')),'Official physical assets changed')
        require(all(old['init_sha256'][k]==new['init_sha256'][k] for k in old['init_sha256']),'Known official state vectors changed')
    atomic_json(REPORT/'official_assets.json',assets)
    keys=('checkpoints','bank','environment','noise','quality_environment_reuse','technical_max_attempts','health_evidence','resource_execution_revision')
    m={k:copy.deepcopy(base[k]) for k in keys};local={str(p.relative_to(ROOT)):sha(p) for p in sorted(PACKAGE.glob('*.py'))};sources=dict(base['source_files']);sources.update(local)
    for p in sorted(REPORT.glob('*.py')):sources[str(p.relative_to(ROOT))]=sha(p)
    for p in [ROOT/'preparations/ease_insights_20261103/camera_paths_v3/arithmetic.py',ROOT/'preparations/ease_insights_20261103/camera_paths_v3/vision_views.py',ROOT/'preparations/ease_insights_20261103/camera_paths_v3/vision_dispatch.py']:
        sources[str(p.relative_to(ROOT))]=sha(p)
    m.update(revision='ease-fresh-camera-depth-calibration-v4',phase='fresh_vision_calibration_v4',output_root=str(OUTPUT),
        conditions=conditions(),planned_conditions=['full_d10','dense_d2'],suites=list(SUITES),tasks=[0],case_table=[dict(suite=s,task=0) for s in SUITES],
        calibration_inits=[45],cases_per_condition=4,total_calibration_episodes=8,noise_repeat=0,official_init_availability=assets,
        protocol_evidence=pin,design_evidence=pin,antecedent_evidence=protocol['antecedent_evidence'],
        local_runtime_sources=local,analysis_sources=local,source_files=sources,
        science=dict(module='fresh_camera_depth',H=50,R=5,VLM_layers=18,AE_layers=18,vision_Full=[27,27],vision_Fast_choices=[[27,18],[18,27]],
            Full_D=10,Fast_D=2,period=2,training=False,environment_seed=7,settling_actions=10,policy_action_limits=dict(LIMITS),both_images_fresh=True,vision_cache=False),
        preflight=dict(status='required_before_rollout',atol=1e-5,rtol=1e-5),
        reference_origins=copy.deepcopy(REFERENCES),calibration_case_scope='8knowncontrolrepeats,zeroindependentqualitycases',
        heldout_scope=protocol['state_exposure_scope'],fixed_input_cost=dict(conditions='All4fixeddefinitionsineverycompletedservice',warmup_requests=96,measured_requests=192))
    m['manifest_hash']=digest(m);validate_output(OUTPUT,m);atomic_json(target,m);read_manifest(target,verify_sources=True)
    return dict(reused=False,manifest_hash=m['manifest_hash'],manifest_file_sha256=sha(target),known_controls=8,source_files=len(sources),GPU_started=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--assets-only',action='store_true');a=p.parse_args();print(json.dumps(official_assets() if a.assets_only else prepare()))
