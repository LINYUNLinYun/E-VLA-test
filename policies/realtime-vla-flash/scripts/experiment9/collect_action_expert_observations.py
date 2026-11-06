"""Replay Experiment 9 Full18 init0 traces; export exact semantic-phase observations."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/spec"))
import spec_client_libero as client
from libero.libero import benchmark

RESULTS = ROOT / "results/experiment9_standard_vlm_pruning_20261004"
PHASES = ("initial", "contact", "goal_alignment")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suites", nargs="+", default=["libero_object", "libero_spatial", "libero_goal", "libero_10"])
    parser.add_argument("--task-limit", type=int, default=10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved, absent, replay = [], [], []
    for suite_name in args.suites:
        suite = benchmark.get_benchmark_dict()[suite_name]()
        path = sorted(RESULTS.glob(f"exp9_pilot_Full18_{suite_name}_i00_10_a*"), reverse=True)[0]
        episodes = json.loads((path / "episode_log.json").read_text())
        for task_id in range(min(10, args.task_limit)):
            row = next(r for r in episodes if r["task_id"] == task_id and r["episode_idx"] == 0)
            telemetry = [json.loads(s) for s in (path / row["telemetry_path"]).read_text().splitlines() if s]
            trace_path = path / row["trace_path"]
            actions = [r["executed_action"] for r in (json.loads(s) for s in trace_path.read_text().splitlines() if s) if r.get("executed_action") is not None]
            if len(telemetry) != len(actions) + 1:
                raise RuntimeError(f"Source alignment invalid: {trace_path}")
            selected = {}
            for phase in PHASES:
                match = next((r for r in telemetry if r.get("semantic_phase") == phase), None)
                if match is None:
                    absent.append({"suite": suite_name, "task_id": task_id, "episode_idx": 0, "phase": phase, "reason": "not_observed_in_prespecified_baseline_trace"})
                else:
                    selected[int(match["step"])] = phase
            task = suite.get_task(task_id)
            env, prompt = client._get_libero_env(task, client.LIBERO_ENV_RESOLUTION, 7)
            max_error, done = 0.0, False
            try:
                env.reset()
                obs = env.set_init_state(suite.get_task_init_states(task_id)[0])
                for _ in range(10):
                    obs, _, _, _ = env.step(client.LIBERO_DUMMY_ACTION)
                for step in range(len(actions) + 1):
                    error = float(np.max(np.abs(np.asarray(obs["robot0_eef_pos"]) - np.asarray(telemetry[step]["eef_pos"]))))
                    max_error = max(max_error, error)
                    if error > 1e-5:
                        raise RuntimeError(f"Replay EEF mismatch {suite_name}/{task_id}/{step}: {error}")
                    if step in selected:
                        phase = selected[step]
                        sample_id = f"{suite_name}_task{task_id:02d}_ep000_step{step:04d}_{phase}"
                        target = args.output_dir / f"{sample_id}.npz"
                        base = client.image_tools.convert_to_uint8(client.image_tools.resize_with_pad(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),224,224))
                        wrist = client.image_tools.convert_to_uint8(client.image_tools.resize_with_pad(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),224,224))
                        state = np.concatenate((obs["robot0_eef_pos"],client._quat2axisangle(obs["robot0_eef_quat"]),obs["robot0_gripper_qpos"])).astype(np.float32)
                        metadata = {"sample_id":sample_id,"suite":suite_name,"task_id":task_id,"episode_idx":0,"semantic_phase":phase,"trajectory_step":step,"task_description":str(prompt),"episode_success":bool(row["success"]),"source_trace":str(trace_path),"source_trace_sha256":hashlib.sha256(trace_path.read_bytes()).hexdigest(),"source_telemetry":str(path / row["telemetry_path"]),"eef_replay_abs_error":error,"seed":7,"warmup_steps":10}
                        if target.exists():
                            with np.load(target,allow_pickle=False) as old:
                                if json.loads(str(old['metadata_json'].item())) != metadata or not np.array_equal(old['robot_state'],state) or not np.array_equal(old['agent_image'],base) or not np.array_equal(old['wrist_image'],wrist):
                                    raise RuntimeError(f"Existing observation differs: {target}")
                        else:
                            np.savez_compressed(target,agent_image=base,wrist_image=wrist,robot_state=state,prompt=np.asarray(str(prompt)),metadata_json=np.asarray(json.dumps(metadata,sort_keys=True)))
                        saved.append(str(target.resolve()))
                    if step < len(actions):
                        obs, _, done, _ = env.step(actions[step])
                if bool(done) != bool(row["success"]):
                    raise RuntimeError(f"Replay success mismatch: {trace_path}")
            finally:
                env.close()
            replay.append({"suite":suite_name,"task_id":task_id,"episode_idx":0,"max_eef_abs_error":max_error,"success_matched":True})
            print(f"Replayed {suite_name}/{task_id}: phases={list(selected.values())}, EEF max error={max_error}",flush=True)
    manifest={"sample_count":len(saved),"task_count":len(replay),"prespecified_phases":PHASES,"observations":saved,"missing_phases":absent,"replay_audit":replay,"protocol":"Real observations replayed from Experiment9 Full18 official init0, exact N+1 telemetry step alignment; no synthetic/random observation inputs."}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"Saved {len(saved)} observations; {len(absent)} prespecified task/phases were not observed.",flush=True)


if __name__ == "__main__":
    main()
