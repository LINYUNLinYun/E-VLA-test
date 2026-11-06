"""Read-only CPU diagnosis of preserved E4 inputs/actions; immutable new receipt."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
from scipy.spatial.transform import Rotation
from .policy_native import native_libero_state, native_libero_axisangle, decode_actions, normalize

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "results/idea1_v3_2/e4_transfer_20261022T051417Z"
SOURCE = ROOT / "results/idea1_v3_2/e4_cpu_adapters/source_reference.json"
STATS = ROOT / "shared/checkpoints/realtime-vla-flash/converted/base/assets/physical-intelligence/libero/norm_stats.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit():
    source = json.loads(SOURCE.read_text())
    stats = json.loads(STATS.read_text())["norm_stats"]
    q = np.asarray(source["actual_eef_world_quaternion_xyzw"])
    native = native_libero_state(source["actual_eef_world_xyz"], q, source["actual_gripper_qpos"])
    previous = np.asarray(source["reference_state8"])
    native_r, previous_r = Rotation.from_rotvec(native[3:6]), Rotation.from_rotvec(previous[3:6])
    am, ast = (np.asarray(stats["actions"][k])[:7] for k in ("mean", "std"))
    sm, sst = (np.asarray(stats["state"][k])[:8] for k in ("mean", "std"))
    rows, hashes = [], {str(p): sha(p) for p in (SOURCE, STATS, Path(__file__), Path(__file__).with_name("policy_native.py"))}
    for initial in sorted(OLD.glob("episodes/*/*/full_d10/attempt*/initial_raw.npz")):
        infer = initial.with_name("infer.jsonl")
        if not infer.exists() or not infer.stat().st_size:
            continue
        first = json.loads(infer.open().readline())
        data = np.load(initial, allow_pickle=False)
        state = data["state8"]
        response = first["response"]
        actions = first.get("actions", response.get("actions"))
        if actions is None:
            raise ValueError("Preserved inference missing decoded actions: " + str(infer))
        replay = decode_actions(response["raw_actions"], am, ast, state, extra_delta=True)
        error = float(np.max(np.abs(replay - np.asarray(actions))))
        clipped = []
        telemetry = initial.with_name("telemetry.jsonl")
        if telemetry.exists():
            for line in telemetry.open():
                command = json.loads(line).get("executed_command") or {}
                executed_raw = command.get("raw7", command.get("raw"))
                if executed_raw is not None:
                    clipped.append(np.abs(executed_raw) > 1)
        rows.append(dict(path=str(initial),state8=state.tolist(),state_z_max_abs=float(np.max(np.abs(normalize(state,sm,sst)))),
            zero_model_output_counterfactual=decode_actions(np.zeros((1,32)),am,ast,state,extra_delta=True)[0].tolist(),
            independent_decode_max_abs=error,decode_passed=error <= 1e-5,
            executed_command_clip_fraction_by_axis=np.mean(clipped,axis=0).tolist() if clipped else None,
            executed_steps=len(clipped)))
        for path in (initial,infer,telemetry):
            if path.exists():
                hashes[str(path)] = sha(path)
    return dict(status="complete",cpu_only=True,gpu_used=False,model_constructed=False,source_sha256=hashes,
        source_branch=dict(original_quaternion_xyzw=q.tolist(),old_state8=previous.tolist(),native_state8=native.tolist(),
            same_SO3_max_matrix_abs=float(np.max(np.abs(native_r.as_matrix()-previous_r.as_matrix()))),
            state_rotation_difference=(native[3:6]-previous[3:6]).tolist(),
            decoded_extraDelta_shift_first6=(native[:6]-previous[:6]).tolist(),
            native_q_negq_state_difference=(native_libero_axisangle(q)-native_libero_axisangle(-q)).tolist()),
        initial_inputs=rows,all_independent_decode_passed=all(r["decode_passed"] for r in rows),
        action_contract=dict(legacy_LIBERO="Keep released extraDelta/AbsoluteActions; original first6 state is added after unnormalization",
            CALVIN="Official RLinf evaluator uses robot_obs[:7]: XYZ+EulerXYZ+opening width, despite norm seventh quantiles near +/-1; preserve discrepancy. Own stats, H5, no extraDelta; relative controller Euler addition",
            PushT="Native target checkpoint state2/action2 H50/R50/D10, not LIBERO7-to2 projection",
            timing_seconds=dict(source_LIBERO_R5_20Hz=.25,old_CALVIN_R5_30Hz=5/30,old_Google_R5_3Hz=5/3,old_PushT_R5_10Hz=.5)),
        limitations=["Branch discrepancy is a proven representation bug, not a measured causal success-rate attribution.",
            "Zero model output is a counterfactual decoder diagnostic, not observed policy prediction.",
            "Command clipping is not measured EEF XYZ; TCP orientation calibration remains separate.",
            "Target-trained Full/V14 are distinct positive controls, not matched controls for LIBERO-trained drafts.",
            "CPU decoder/state parity does not establish model-policy parity or cross-environment performance."])


def freeze_native_pilot():
    """Separate service-only pilot registration, never an old E4/formal denominator."""
    from preparations.idea1_v3_2.transfer_protocol import digest,sha as stream_sha
    from .policy_native import native_checkpoint_header
    directory=Path(__file__).parent
    sources=list(directory.glob("policy_*.py"))+[directory/"test_policy_native.py"]
    sources += [ROOT/"policies/realtime-vla-flash/src/openpi/models_pytorch/pi0_pytorch.py",
        ROOT/"policies/realtime-vla-flash/src/openpi/models_pytorch/gemma_pytorch.py",
        ROOT/"policies/realtime-vla-flash/src/openpi/policies/policy.py",
        ROOT/"policies/realtime-vla-flash/src/openpi/transforms.py",
        ROOT/"policies/RLinf/rlinf/models/embodiment/openpi/policies/calvin_policy.py",
        ROOT/"policies/RLinf/rlinf/models/embodiment/openpi/dataconfig/calvin_dataconfig.py",
        ROOT/"policies/RLinf/rlinf/models/embodiment/openpi/dataconfig/__init__.py"]
    sources+=list((ROOT/"policies/realtime-vla-flash/src/openpi/models_pytorch/transformers_replace").rglob("*.py"))
    checkpoint=ROOT/"shared/checkpoints/e4_supplement/RLinf/RLinf-Pi0-CALVIN-ABC-D-SFT/model.safetensors"
    norm=checkpoint.parent/"InternRobotics/InternData-Calvin_ABC/norm_stats.json"
    identity=checkpoint.with_name(checkpoint.name+".download_identity.json")
    checkpoint_identity=json.loads(identity.read_text())
    actual=stream_sha(checkpoint)
    if actual!=checkpoint_identity["sha256"]: raise ValueError("Pinned native checkpoint hash mismatch")
    capture=ROOT/"results/idea1_v3_2_e4_supplement/native_preflight/calvin_captured_formalD0"
    receipts=[identity,norm,capture/"native_input_gripper_sign.npz",capture/"gripper_sign_capture_receipt.json",
        ROOT/"results/idea1_v3_2_e4_supplement/asset_audit/calvin_training_episode0_evidence_v1.json",
        ROOT/"results/idea1_v3_2_e4_supplement/policy_audit/native_calvin_gpu_v1.json"]
    manifest=dict(schema=1,kind="bounded_native_CALVIN_pilot_not_formal100",family="calvin_abc_d_native",
        modes=["full_d10","v14_d10"],H=5,R=5,D=10,official_sequence_indices=[0,1,2],
        denominator="pilot only: first3 canonical sequences per model; not formal100 or old E4 denominator",
        gripper_primary="actual observed robot_obs[-1] stored action sign, never current predicted action",
        gripper_diagnostic="official standalone width path preserved in v1 implementation-only gate",
        input_preflight="exact captured instruction and same realD images; canonical Full strict1e-5 wrapper parity",
        checkpoint_metadata=checkpoint_identity,checkpoint_header=native_checkpoint_header(checkpoint),
        normalizer_sha256=sha(norm),source_files={str(p.relative_to(ROOT)):sha(p) for p in sources},
        receipt_files={str(p):sha(p) for p in receipts},created_unix=time.time())
    manifest["manifest_hash"]=digest(manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--native-pilot-input")
    parser.add_argument("--instruction",default="open the drawer")
    parser.add_argument("--launch",action="store_true")
    parser.add_argument("--freeze-native-pilot",action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if args.freeze_native_pilot:
        result=freeze_native_pilot()
        with output.open("x") as stream: json.dump(result,stream,indent=2,allow_nan=False)
        print(json.dumps(dict(output=str(output),manifest_hash=result["manifest_hash"])));return
    if args.native_pilot_input:
        if not args.launch: raise RuntimeError("Explicit --launch required")
        from .policy_native import NativeCalvinPolicy
        import signal
        def expired(signum,frame): raise TimeoutError("NATIVE_PILOT_1200S_DEADLINE")
        signal.signal(signal.SIGALRM,expired)
        signal.alarm(1200)
        started=time.time()
        result=dict(status="started",pid=os.getpid(),started_unix=started,deadline_unix=started+1200,
            source_sha256={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name("policy_native.py"))},
            input_path=args.native_pilot_input,input_sha256=sha(args.native_pilot_input))
        with output.open("x") as stream: json.dump(result,stream,indent=2)
        try:
            data=np.load(args.native_pilot_input,allow_pickle=False)
            raw={key:data[key] for key in ("base_rgb","wrist_rgb","state7")}
            checkpoint=ROOT/"shared/checkpoints/e4_supplement/RLinf/RLinf-Pi0-CALVIN-ABC-D-SFT/model.safetensors"
            stats=ROOT/"shared/checkpoints/e4_supplement/RLinf/RLinf-Pi0-CALVIN-ABC-D-SFT/InternRobotics/InternData-Calvin_ABC/norm_stats.json"
            policy=NativeCalvinPolicy(checkpoint,stats,launch=True)
            result.update(constructed_unix=time.time(),strict_load_passed=True,header_audit=policy.header_audit,
                gpu_preflight=policy.gpu_preflight,transformers_overlay={p:sha(p) for p in policy.transformers_overlay},
                state7=raw["state7"].tolist(),normalizer_sha256=sha(stats))
            result["gate"]=policy.preflight(raw,args.instruction,7)
            result["full"]=policy.predict(raw,args.instruction,7,mode="full_d10")
            result["v14"]=policy.predict(raw,args.instruction,7,mode="v14_d10")
            result.update(status="complete",completed_unix=time.time(),gpu_resources_released="process exit")
        except Exception:
            result.update(status="failed",error=traceback.format_exc(),completed_unix=time.time())
            raise
        finally:
            from preparations.idea1_v3_1.common import atomic_json
            atomic_json(output,result)
        print(json.dumps(dict(output=str(output),status=result["status"],gate=result["gate"])))
        return
    result = audit()
    with output.open("x") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(output),inputs=len(result["initial_inputs"]),decode_passed=result["all_independent_decode_passed"],branch=result["source_branch"])))


if __name__ == "__main__":
    main()
