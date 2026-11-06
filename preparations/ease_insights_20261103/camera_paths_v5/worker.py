"""Strict Full calibration receipts and case-local recovery."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
from preparations.idea1_v3_1.common import lock
from .noise import model_seed
from .common import atomic_json, read_manifest, sha
from .common import cases, planned_config, runtime_revision, validate_output, stage_for
from .rollout import SETTLING_ACTIONS
from .common import LIMITS
from preparations.ease_insights_20261103.vlm.recovery import run_with_retries, recover_case
from .nodes import scheduled_nfe


def validate_ae(row,skip,nfe):
    audit=row['route'].get('ae_execution_audit')
    expected=dict(actual_NFE=nfe,attempted_layers_per_step=list(range(1,19)),
        executed_layers_per_step=[l for l in range(1,19) if l not in skip],bypassed_layers_per_step=list(skip),
        actual_layer_calls=nfe*(18-len(skip)),attempted_layer_calls=nfe*18,bypassed_layer_calls=nfe*len(skip),
        original_helper_restored=True,scope='Observed eager helper dispatch, not FLOPs, latency or quality')
    if (audit!=expected or row['timing'].get('ae_execution_audit')!=expected
        or row['timing'].get('actual_NFE')!=nfe or row['route'].get('ae_bypassed_layers')!=list(skip)):
        raise ValueError('Actual AE layer visits, bypass or native NFE differ')


def episode_dir(output, condition, scene):
    return Path(output)/condition/"episodes"/scene["suite"]/f"task{scene['task']:02d}"/f"init{scene['init']:02d}"


def completed(output, condition, scene, manifest_hash):
    directory = episode_dir(output, condition, scene)
    path = directory / "episode_receipt.json"
    if not path.exists():
        return False
    receipt = json.loads(path.read_text())
    if receipt["manifest_hash"] != manifest_hash or receipt["condition"] != condition or receipt["case"] != scene:
        raise ValueError("Mixed episode identity")
    for name, expected in receipt["artifacts"].items():
        if sha(directory/name) != expected:
            raise ValueError("Episode artifact changed")
    return bool(receipt["valid"])


def initial_gate(output, manifest_hash, condition, scene, input_hash, noise_seed, *, assets=None):
    path = Path(output)/"initial_observations"/scene["suite"]/f"task{scene['task']:02d}"/f"init{scene['init']:02d}.json"
    payload = dict(manifest_hash=manifest_hash, case=scene, input_hash=input_hash, seed=noise_seed)
    if assets is not None:
        payload["physical_assets"] = assets
    with lock(str(path)+".lock"):
        if path.exists():
            ref = json.loads(path.read_text())
            if any(ref.get(k) != v for k, v in payload.items()):
                raise ValueError("Paired initial observation/noise mismatch")
        else:
            atomic_json(path, dict(payload, condition=condition))
    return dict(**payload, passed=True, reference=str(path))


def validate_prefix(row,config,full):
    from reports.ease_insights_20261103.prefix_kv_preparation_v1.layout_contract import plan
    pattern='all' if full else config['prefix_pattern'];audit=row['route']['prefix_execution_audit']
    layout=plan('identity' if pattern=='all' else pattern,audit['original_prefix']);nfe=10 if full else 2
    expected=dict(pattern=pattern,original_prefix=layout.original_prefix,packed_prefix=layout.packed_prefix,
        kept_prefix_indices=list(layout.keep_indices),visual_tokens_per_camera=[256,256] if pattern=='all' else [128,128],
        language_tokens_retained=layout.original_prefix-512,decoder_absolute_positions=list(layout.decoder_absolute_positions),
        decoder_storage_positions=list(layout.decoder_storage_positions),kv_shape=list(layout.kv_shape),attention_shape=list(layout.attention_shape),
        original_rope_table_reused=True,dense_VLM_prefill_already_completed=True,actual_helper_calls=18*nfe,
        actual_prefix_length=layout.packed_prefix,actual_total_key_tokens=layout.packed_prefix+51,
        original_buffers_restored=True,original_helper_restored=True,packed=pattern!='all',packing_latency_measured=pattern!='all',
        scope='Actual full AE18 helper visits and packed key shape with unchanged RoPE;no quality or speed claim')
    if pattern!='all':expected['kept_index_sha256']=hashlib.sha256(json.dumps(list(layout.keep_indices),separators=(',',':')).encode()).hexdigest()
    timing_keys={'packing_wall_ms','packing_cuda_ms','restore_wall_ms'}
    if (set(audit)!=set(expected)|timing_keys or any(audit.get(k)!=v for k,v in expected.items())
        or row['timing'].get('prefix_execution_audit')!=audit or row['route'].get('prefix_pattern')!=pattern):
        raise ValueError('Actual prefix storage,original indices/RoPE,helper count or restoration differs')
    for k in timing_keys:
        value=audit[k]
        if pattern=='all':
            if value!=(None if k=='packing_cuda_ms' else 0.):raise ValueError('Full native prefix must not report packing work')
        elif not isinstance(value,(int,float)) or not np.isfinite(value) or value<0:raise ValueError('Invalid real packing/restore timing')
    if any(row['timing'].get(k)!=audit[v] for k,v in [('prefix_pack_wall_ms','packing_wall_ms'),('prefix_pack_cuda_ms','packing_cuda_ms'),('prefix_restore_wall_ms','restore_wall_ms')]):
        raise ValueError('Packing latency fields differ')


def validate_camera(row,config,full):
    from preparations.ease_insights_20261103.camera_paths_v4.worker import validate_camera as native_vision
    native_vision(row,dict(id='dense_d2'),True);validate_prefix(row,config,full)


def finalize(output,condition,scene,manifest,directory=None):
    directory=Path(directory) if directory is not None else episode_dir(output,condition,scene)
    result=json.loads((directory/'result.json').read_text());done=json.loads((directory/'complete.json').read_text());identity=json.loads((directory/'identity.json').read_text())
    cfg=planned_config(manifest,condition)
    if identity.get('noise_repeat')!=0 or manifest['noise_repeat']!=0:raise ValueError('Fresh-vision stream must be original condition-independent noise0')
    if (not result['valid'] or not done['valid'] or any(x['manifest_hash']!=manifest['manifest_hash'] for x in (result,done,identity))
        or result['case']!=scene or identity['case']!=scene or result['condition']!=condition or identity['config']!=cfg or result['config']!=cfg):
        raise ValueError('Invalid or mixed fresh-vision episode')
    assets=verify_physical_identity(manifest,scene,identity);protocol=identity['protocol']
    if (protocol['manifest_hash']!=manifest['manifest_hash'] or protocol['condition_id']!=condition or protocol['config']!=cfg
        or not protocol['real_input_parity_passed'] or protocol.get('qk_reference_passed') is not True
        or protocol['checkpoint_sha256']!=manifest['checkpoints']['pi0']['sha256'] or protocol.get('policy_action_limits')!=LIMITS
        or protocol.get('settling_actions')!=SETTLING_ACTIONS or protocol.get('AE_numerical_gate_passed') is not True
        or protocol.get('camera_numerical_gate_passed') is not True or protocol.get('internal_actual_NFE_exported') is not True
        or protocol.get('fixed_ae_skip_layers')!=[] or protocol.get('fixed_vision_kept')!=cfg['vision_kept'] or protocol.get('fresh_cameras')!=[0,1] or protocol.get('fixed_prefix_pattern')!=cfg['prefix_pattern'] or protocol.get('prefix_attention_numerical_gate_passed') is not True):
        raise ValueError('Incorrect or ungated fresh-vision service')
    rows=[json.loads(s) for s in (directory/'infer.jsonl').read_text().splitlines()]
    if len(rows)!=result['predictions'] or not rows:raise ValueError('Incomplete inference records')
    validate_action_artifacts(directory,rows,result);step=0;history=[]
    for index,row in enumerate(rows):
        expected_seed=model_seed(0,scene['suite'],scene['task'],scene['init'],index)
        expected_noise=hashlib.sha256(np.random.default_rng(expected_seed).standard_normal((50,32),dtype=np.float32).tobytes()).hexdigest()
        full=condition=='full_d10' or index%3==0
        if full:history=[*history,step][-2:]
        route=row['route'];execution=route['execution_audit'];full_count=index+1 if condition=='full_d10' else index//3+1
        if (row['round']!=index or row['step']!=step or row['seed']!=expected_seed or row['model_noise_sha256']!=expected_noise
            or route['refresh']!=full or route['path']!=('full' if full else 'prefix_attention')
            or route['history_steps']!=history or route['compensation'] is not None or route['camera_cache_used'] is not False
            or route['skip_layers']!=[] or route['capture_layers']!=[] or route['selector_calls']!=0
            or row['counts']!=dict(full=full_count,draft=index+1-full_count,verify=0)
            or execution['qkv_layers']!=list(range(1,19)) or execution['hidden_update_layers']!=list(range(1,18))
            or execution['kv_only_layers']!=[] or execution['patched_layers']!=[] or execution['captured_layers']!=[]
            or execution['grad_enabled'] or execution['selector_calls']!=0):
            raise ValueError('Full/Fast cadence,VLM execution,noise or no-cache contract differs')
        nfe=10 if full else 2
        if scheduled_nfe(row['timing']['nodes'])!=nfe:raise ValueError('Native Euler node schedule differs')
        validate_ae(row,[],nfe);validate_camera(row,cfg,full)
        if type(row['executed_length']) is not int or not 0<row['executed_length']<=5:raise ValueError('Invalid executed length')
        step+=row['executed_length']
    from preparations.idea1_v2_1.runtime_client import input_digest
    first_input=directory/'input_0000.npz'
    if sha(first_input)!=rows[0]['input_file_sha256']:raise ValueError('Initial input artifact changed')
    with np.load(first_input,allow_pickle=False) as saved:
        actual_input=input_digest({k:saved[k] for k in saved.files if k not in ('instruction','seed')},str(saved['instruction'].item()))
        if actual_input!=rows[0]['input_hash'] or int(saved['seed'])!=rows[0]['seed'] or str(saved['instruction'].item())!=identity['prompt']:
            raise ValueError('Initial saved input,noise or instruction differs')
    if step!=result['actions'] or result['counts']!=rows[-1]['counts']:raise ValueError('Final executed accounting differs')
    gate=initial_gate(output,manifest['manifest_hash'],condition,scene,rows[0]['input_hash'],rows[0]['seed'],assets=assets)
    receipt=dict(valid=True,manifest_hash=manifest['manifest_hash'],condition=condition,case=scene,initial_gate=gate,
        policy_action_limit=LIMITS[scene['suite']],settling_actions=SETTLING_ACTIONS,success=result['success'],executed_steps=step,predictions=len(rows),counts=result['counts'],
        technical_runtime_revision=runtime_revision(manifest),
        ae_audit=dict(scope='exported schedule nodes',scheduled_NFE_counts=dict(Counter(scheduled_nfe(r['timing']['nodes']) for r in rows)),actual_NFE_exported=True),
        prefix_audit=dict(actual_key_token_counts=dict(Counter(r['route']['prefix_execution_audit']['actual_total_key_tokens'] for r in rows)),original_absolute_RoPE_retained=True,packing_cost_in_policy_wall=True),
        vision_audit=dict(image_layer_evaluation_counts=dict(Counter(r['route']['camera_execution_audit']['image_layer_evaluations'] for r in rows)),both_images_fresh=True,cached_views_used=False),
        wall_s=result['wall_s'],model_request_ms_total=sum(r['timing']['policy_wall_ms'] for r in rows),rpc_ms_total=sum(r['rpc_ms'] for r in rows),
        gpu_uuid=protocol['gpu_uuid'],worker_pid=os.getpid(),artifacts={name:sha(directory/name) for name in ('identity.json','result.json','complete.json','infer.jsonl','telemetry.jsonl','arrays.npz','input_0000.npz')})
    atomic_json(directory/'episode_receipt.json',receipt);return receipt


def validate_action_artifacts(directory, rows, result):
    actions = result["actions"]
    action_limit = LIMITS[result["case"]["suite"]]
    if (type(actions) is not int or not 0 < actions <= action_limit or
        result.get("policy_action_limit") != action_limit or result.get("settling_actions") != SETTLING_ACTIONS or
        result.get("termination") != ("success" if result["success"] else "timeout") or
        (not result["success"] and actions != action_limit)):
        raise ValueError("Episode termination or uniform policy action budget differs")
    telemetry = [json.loads(line) for line in (directory/"telemetry.jsonl").read_text().splitlines()]
    if len(telemetry) != actions+1 or telemetry[0]["step"] != 0 or telemetry[0]["action"] is not None:
        raise ValueError("Incomplete policy action telemetry")
    step = 0
    with np.load(directory/"arrays.npz", allow_pickle=False) as saved:
        for index, row in enumerate(rows):
            decoded, raw = saved[f"actions_{index}"], saved[f"raw_{index}"]
            if (decoded.shape != (50,7) or raw.shape != (50,32) or
                not np.isfinite(decoded).all() or not np.isfinite(raw).all() or row.get("execute_length") != 5):
                raise ValueError("Raw/decoded output shape or execution prefix differs")
            count = row["executed_length"]
            if type(count) is not int or not 0 < count <= 5 or (index < len(rows)-1 and count != 5):
                raise ValueError("Invalid executed prefix length")
            for command in decoded[:count]:
                step += 1
                observed = telemetry[step]
                if observed["step"] != step or not np.array_equal(np.asarray(observed["action"],np.float32),command):
                    raise ValueError("Executed action differs from acknowledged model prefix")
    if step != actions:
        raise ValueError("Action artifact count differs")


def physical_assets(identity):
    return {k: identity[k] for k in ("init_vector_sha256", "bddl_sha256", "prompt", "env_seed", "settling", "policy_action_limit", "control_hz")}


def verify_physical_identity(manifest, scene, identity):
    expected = next(r for r in manifest["official_init_availability"]["cases"]
                    if r["suite"] == scene["suite"] and r["task"] == scene["task"])
    if (identity["case"] != scene or identity["manifest_hash"] != manifest["manifest_hash"] or
        identity["init_vector_sha256"] != expected["init_sha256"][str(scene["init"])] or
        identity["bddl_sha256"] != expected["bddl_sha256"] or identity["prompt"] != expected["prompt"] or
        identity["env_seed"] != 7 or identity["settling"] != SETTLING_ACTIONS or
        identity["policy_action_limit"] != LIMITS[scene["suite"]] or identity["control_hz"] != 20.):
        raise ValueError("Official state, BDDL, prompt, seed, settling or action limit changed")
    return physical_assets(identity)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True); parser.add_argument("--condition", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--stage", default=None, choices=("calibration", "discovery", "confirm"))
    parser.add_argument("--shard-index", type=int, default=0); parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args(); output = Path(args.output).resolve()
    manifest = read_manifest(output/"manifest.json", verify_sources=True)
    validate_output(output, manifest)
    cfg = planned_config(manifest, args.condition)
    from .common import reference
    control_origins = [reference(alias,verify_sources=True) for alias in ('full_d10','dense_d2')] if stage_for(manifest)=='discovery' else []
    scenes = cases(manifest, args.stage, args.shard_index, args.shards)
    from .rollout import episode
    from preparations.idea1_v3_1.rollout import Environments
    from preparations.idea1_v2_1.runtime_client import input_digest, rpc
    class SeededEnvironments(Environments):
        def acquire(self, suite, task):
            random.seed(7); np.random.seed(7)
            return super().acquire(suite, task)
    pool = SeededEnvironments(reuse=False)
    def gated_rpc(port, request):
        response = rpc(port, request)
        if request.get("op") == "health":
            p = response["protocol"]
            if (p["manifest_hash"] != manifest["manifest_hash"] or p["condition_id"] != cfg["id"] or
                p.get("config") != cfg or not p["real_input_parity_passed"] or
                p.get("qk_reference_passed") is not True or p.get("policy_action_limits") != LIMITS):
                raise ValueError("Incorrect or ungated within-suite service")
        return response
    try:
        for scene in scenes:
            directory = episode_dir(output, cfg["id"], scene)
            with lock(str(directory)+".lock"):
                if completed(output, cfg["id"], scene, manifest["manifest_hash"]):
                    continue
                recovered = recover_case(output, cfg["id"], scene, manifest, finalize, already_locked=True)
                if recovered is not None:
                    print(json.dumps(recovered), flush=True); continue
                def gate(identity, request):
                    assets = verify_physical_identity(manifest, scene, identity)
                    for old_root,old_manifest,old_common,old_worker,old_cid in control_origins:
                        old=json.loads((old_worker.episode_dir(old_root,old_cid,scene)/'episode_receipt.json').read_text())['initial_gate']
                        if old['input_hash']!=input_digest(request['obs'],request['instruction']) or old['seed']!=request['seed'] or old['physical_assets']!=assets:
                            raise ValueError('Preserved control initial input/noise/physical identity differs before prediction')
                    initial_gate(output, manifest["manifest_hash"], cfg["id"], scene,
                        input_digest(request["obs"], request["instruction"]), request["seed"], assets=assets)
                result = run_with_retries(directory, manifest["technical_max_attempts"],
                    lambda: episode(directory, args.port, scene, cfg, pool, manifest["manifest_hash"],
                                    max_policy_actions=LIMITS[scene["suite"]],
                                    settling_actions=manifest["science"]["settling_actions"],
                                    noise_repeat=manifest['noise_repeat'],
                                    rpc_call=gated_rpc, initial_input_gate=gate),
                    lambda: finalize(output, cfg["id"], scene, manifest), pool.close)
                print(json.dumps(result), flush=True)
    finally:
        pool.close()


if __name__ == "__main__":
    main()
