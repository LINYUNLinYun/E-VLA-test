"""New formal native CALVIN100 registration; never edits/relabels clipped pilot."""
import argparse
import json
from pathlib import Path
import time
from preparations.idea1_v3_2.transfer_protocol import digest,sha,load_manifest
from .native_envs import CALVIN_ENV,CALVIN_MODELS,DEBUG_D
from .policy_native import native_checkpoint_header

ROOT=Path(__file__).resolve().parents[2]
CANONICAL=ROOT/"results/idea1_v3_2_e4_supplement/native_preflight/calvin_formal1000/sequences.json"
ASSETS=ROOT/"shared/checkpoints/e4_supplement/RLinf/RLinf-Pi0-CALVIN-ABC-D-SFT"


def freeze(deadline_unix):
    if deadline_unix<=time.time(): raise ValueError("Fresh sufficient formal deadline required")
    old=load_manifest(ROOT/"results/idea1_v3_2_e4_supplement/policy_audit/native_pilot_v3/manifest.json")
    sequences=json.loads(CANONICAL.read_text())
    if len(sequences["sequences"])!=1000: raise ValueError("Canonical1000 source required before fixed100prefix")
    for index,row in enumerate(sequences["sequences"][:100]):
        if row["index"]!=index or len(row["tasks"])!=5: raise ValueError("Fixed100prefix changed")
    sources=dict(old["source_files"])
    paths=[Path(__file__),Path(__file__).with_name("evaluation_v2.py"),Path(__file__).with_name("native_exact.py"),
        Path(__file__).with_name("native_envs.py"),Path(__file__).with_name("evaluation.py"),
        ROOT/"preparations/idea1_v3_2/transfer_client.py",ROOT/"preparations/idea1_v3_2/transfer_policy.py",
        ROOT/"preparations/idea1_v3_2/transfer_prompts.py",ROOT/"preparations/idea1_v3_1/server.py",
        CALVIN_ENV/"calvin_env/robot/robot.py",CALVIN_ENV/"calvin_env/envs/play_table_env.py",
        CALVIN_ENV/"calvin_env/envs/tasks.py"]
    for filename,expected in sequences["contract"]["source_sha256"].items():
        if sha(ROOT/filename)!=expected: raise ValueError("Canonical native sequence source changed: "+filename)
        sources[filename]=expected
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in paths})
    config_files=[DEBUG_D/".hydra/merged_config.yaml",CALVIN_ENV/"conf/scene/calvin_scene_D_eval.yaml",
        CALVIN_ENV/"conf/cameras/cameras/static.yaml",CALVIN_ENV/"conf/cameras/cameras/gripper.yaml",
        CALVIN_MODELS/"conf/callbacks/rollout/tasks/new_playtable_tasks.yaml",
        CALVIN_MODELS/"conf/annotations/new_playtable_validation.yaml"]
    checkpoint=ASSETS/"model.safetensors";identity=checkpoint.with_name(checkpoint.name+".download_identity.json")
    metadata=json.loads(identity.read_text())
    if sha(checkpoint)!=metadata["sha256"]: raise ValueError("Native checkpoint hash differs pinned release")
    stats=ASSETS/"InternRobotics/InternData-Calvin_ABC/norm_stats.json"
    sign=ROOT/"results/idea1_v3_2_e4_supplement/native_preflight/calvin_captured_formalD0/native_input_gripper_sign.npz"
    evidence=ROOT/"results/idea1_v3_2_e4_supplement/asset_audit/calvin_training_episode0_evidence_v1.json"
    receipts=[CANONICAL,identity,stats,sign,evidence,*config_files]
    manifest=dict(schema=2,kind="formal_native_CALVIN_ABC_to_D_fixed100",family="calvin_abc_d_native",
        checkpoint_metadata=metadata,checkpoint_header=native_checkpoint_header(checkpoint),normalizer_sha256=sha(stats),
        modes=["full_d10","v14_d10"],H=5,R=5,D=10,sequence_count=100,conditions=200,
        sequences=sequences["sequences"][:100],canonical1000_sha256=sha(CANONICAL),
        shard_size=10,shards=[list(range(start,start+10)) for start in range(0,100,10)],
        per_launch_new_sequence_limit=10,sharding_is_not_target_cancellation=True,
        deadline_unix=deadline_unix,gripper_state="actual observed robot_obs[-1] stored +/-1, never current predicted action",
        action_contract="ExactNativeCalvinEnv: native decoded first6 passed through unchanged, onlygripper>0:+1 else:-1",
        episode_contract=sequences["contract"],
        success_protocol="one actual oracle check aftereveryexecutedstep; stopchunk immediatelyat success; stopchain at firstfailedsubtask",
        logging="flush each native step/request/subtask; completed sequenceattempt immutable; technicalfailure resumable not policyzero",
        previous_pilot_quality_reusable=False,previous_pilot_reason="Clipped first6 protocol; retain pilot3 independently, rerun exactnativeprefix0..99",
        ABCD_to_D_status="No verified ABCD-trained native checkpoint supplied; do not relabel ABC-trained weights",
        source_files=sources,receipt_files={str(p):sha(p) for p in receipts},created_unix=time.time(),
        launch_gate="CPUregistration only; separately grant idle physicalGPU, strictallkeyload+realinputFullbitwise/strict1e-5gate andV14shell4 required beforequality")
    manifest["manifest_hash"]=digest(manifest)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",required=True);p.add_argument("--deadline-unix",type=float,required=True)
    a=p.parse_args();path=Path(a.output)
    if path.exists(): raise FileExistsError(path)
    result=freeze(a.deadline_unix);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(path),manifest_hash=result["manifest_hash"],sequences=100,conditions=200)))


if __name__=="__main__":main()
