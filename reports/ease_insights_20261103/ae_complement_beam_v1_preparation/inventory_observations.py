"""CPU-only provenance inventory for a possible fixed-budget AE pair search."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path('/root/nfs/embodied/benchmark');OUT=Path(__file__).resolve().parent
BASE=ROOT/'policies/realtime-vla-flash/results/experiment9_standard_vlm_pruning_20261004/action_expert_layer_probe'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)


def main():
    path=BASE/'observations/manifest.json';manifest=json.loads(path.read_text());sources={};observations=[];traces={}
    require(manifest['sample_count']==110 and manifest['task_count']==40,'Archived observation coverage changed')
    for filename in manifest['observations']:
        p=Path(filename);digest=sha(p)
        with np.load(p,allow_pickle=False) as z:
            meta=json.loads(str(z['metadata_json'].item()));prompt=str(z['prompt'].item())
            require(meta['sample_id']==p.stem and prompt==meta['task_description'],'Observation identity/prompt changed')
            require(z['agent_image'].shape==z['wrist_image'].shape==(224,224,3)
                and z['agent_image'].dtype==z['wrist_image'].dtype==np.uint8,'Archived camera shape/dtype changed')
            require(z['robot_state'].shape==(8,) and z['robot_state'].dtype==np.float32 and np.isfinite(z['robot_state']).all(),'Archived state changed')
        require(meta['seed']==7 and meta['warmup_steps']==10 and meta['episode_idx']==0 and meta['eef_replay_abs_error']<=1e-5,'Replay metadata differs')
        trace=Path(meta['source_trace']);expected=meta['source_trace_sha256']
        if trace not in traces:
            require(sha(trace)==expected,'Source trajectory trace changed');traces[trace]=expected
        require(traces[trace]==expected,'Inconsistent source trace binding')
        sources[meta['sample_id']]=digest
        observations.append(dict(path=str(p),sha256=digest,sample_id=meta['sample_id'],suite=meta['suite'],task=meta['task_id'],
            phase=meta['semantic_phase'],trajectory_step=meta['trajectory_step'],instruction=prompt,source_trace=dict(path=str(trace),sha256=expected)))
    archives=[]
    for seed in (12345,23456,34567):
        paths=sorted((BASE/'probe_shards'/('seed'+str(seed))/'samples').glob('*.npz'));require(len(paths)==110,'Archived probe count differs')
        seen=set()
        for p in paths:
            with np.load(p,allow_pickle=False) as z:meta=json.loads(str(z['metadata_json'].item()))
            require(meta['sample_id'] not in seen and meta['observation_sha256']==sources[meta['sample_id']] and meta['noise_seed']==seed,'Probe observation hash or noise identity differs')
            seen.add(meta['sample_id']);archives.append(dict(path=str(p),sha256=sha(p),sample_id=meta['sample_id'],noise_seed=seed))
        require(seen==set(sources),'Probe observation set differs')
    suites=sorted(set(o['suite'] for o in observations));coverage={s:dict(tasks=len(set(o['task'] for o in observations if o['suite']==s)),
        phases=dict(Counter(o['phase'] for o in observations if o['suite']==s))) for s in suites}
    value=dict(status='pass',time=time.time(),observations=len(observations),tasks=40,archived_probe_records=len(archives),
        input_manifest=dict(path=str(path),sha256=sha(path)),coverage=coverage,observation_evidence=observations,probe_evidence=archives,
        physical_trace_sources_rehashed=len(traces),GPU_or_model_or_environment_started=False,
        current_camera_outcomes_read=False,historical_normalization_and_runtime_recreated=False,
        scope='Archived replayed224x224 rotated camera inputs and8D state; independently rehashed observation-to-probe and source-trace bindings. Existing phase labels are historical annotations, not an online phase detector. No quality or complementary-path evidence.',
        future_requirement='A new runtime must accept the archived preprocessed input contract explicitly, check current checkpoint/norm assets and numerical gates, and generate fresh D2 references. Do not treat old D10 single-layer scores as new multi-layer or R5 rollout evidence.',
        source=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)))
    (OUT/'observation_inventory.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:v for k,v in value.items() if k not in ('observation_evidence','probe_evidence')}))


if __name__=='__main__':main()
