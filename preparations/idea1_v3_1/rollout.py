"""Lean native LIBERO evaluation: camera inputs always rendered, optional I/O."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import uuid

import numpy as np

from .common import LIMITS, SUITES, atomic_json, lock, read_manifest, seed
from .server import rpc


class Environments:
    def __init__(self,reuse=False):
        self.reuse = reuse
        self.entries = {}

    def acquire(self,suite,task):
        from libero.libero import benchmark,get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        selected = benchmark.get_benchmark_dict()[suite]()
        definition = selected.get_task(task)
        key = (suite,task)
        created = key not in self.entries
        if created:
            # Keep at most one offscreen environment per worker. Consecutive
            # official inits reuse it; changing tasks releases the old renderer.
            self.close()
            bddl = Path(get_libero_path("bddl_files"))/definition.problem_folder/definition.bddl_file
            env = OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256)
            self.entries[key] = (env,definition,bddl)
        return (*self.entries[key],selected,created)

    def release(self,suite,task):
        if not self.reuse:
            self.entries.pop((suite,task))[0].close()

    def close(self):
        for env,_,_ in self.entries.values():
            env.close()
        self.entries.clear()


def run_episode(directory,port,condition,suite,task,init,pool,manifest_hash,video=False,save_observations=False):
    from preparations.idea1_v2_1.runtime_client import official_init_states,input_digest,telemetry_helpers
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    begin = time.perf_counter()
    timings = dict(environment_acquire_ms=0.,reset_ms=0.,rpc_ms=0.,environment_step_ms=0.,io_ms=0.)
    tick = time.perf_counter()
    env,definition,bddl,selected,created = pool.acquire(suite,task)
    timings["environment_acquire_ms"] = (time.perf_counter()-tick)*1000
    writer = None
    try:
        states = official_init_states(selected,task)
        vector = np.asarray(states[init])
        init_hash = hashlib.sha256(vector.tobytes()).hexdigest()
        random.seed(7)
        np.random.seed(7)
        tick = time.perf_counter()
        env.seed(7)
        env.reset()
        obs = env.set_init_state(vector.copy())
        for _ in range(10):
            obs,_,_,_ = env.step([0.]*6+[-1.])
        timings["reset_ms"] = (time.perf_counter()-tick)*1000
        if float(env.env.control_freq) != 20.:
            raise ValueError("Native control frequency must remain 20Hz")
        controller = env.env.robots[0].controller
        scales = {name:np.asarray(getattr(controller,name)).tolist() for name in ("input_min","input_max","output_min","output_max")}
        if not (np.allclose(scales["input_min"],-1.) and np.allclose(scales["input_max"],1.)
                and np.allclose(scales["output_max"],[.05,.05,.05,.5,.5,.5])
                and np.allclose(scales["output_min"],[-.05,-.05,-.05,-.5,-.5,-.5])):
            raise ValueError("Original native OSC controller scale changed")
        # No per-episode server reset is needed: each native policy is stateless.
        # Avoid resetting a shared server in between another client's requests.
        protocol = rpc(port,dict(op="health"))["protocol"]
        if protocol["manifest_hash"] != manifest_hash or protocol["condition_id"] != condition["id"]:
            raise ValueError("RPC condition does not match frozen rollout")
        if not protocol.get("real_input_parity_passed"):
            raise ValueError("Run real-input parity on the service before quality/throughput rollout")
        horizon = protocol["H"]
        execute = condition["execute_length"]
        execute = execute[suite] if isinstance(execute,dict) else execute
        if horizon != condition["horizon"] or not 0 < execute <= horizon:
            raise ValueError("Native H/R mismatch")
        identity = dict(manifest_hash=manifest_hash,condition=condition,suite=suite,task=task,init=init,
                        init_vector_sha256=init_hash,bddl_sha256=hashlib.sha256(bddl.read_bytes()).hexdigest(),
                        env_seed=7,environment_reused=not created,control_hz=20,controller=scales,protocol=protocol)
        atomic_json(directory/"identity.json",identity)
        replay = telemetry_helpers()
        primary,target = replay._goal_object_names(env.env)
        initial_positions = {name:replay._object_state(env.env,name).get("pos") for name in primary}
        if video:
            import imageio.v2 as imageio
            writer = imageio.get_writer(str(directory/"video.mp4"),fps=20,codec="libx264",pixelformat="yuv420p",macro_block_size=2)
        telemetry,infers = [],[]
        def capture(step,action):
            tick = time.perf_counter()
            row = replay.snapshot(env,obs,step=step,when="initial" if step==0 else "post_action",action=action,
                                  primary_names=primary,target_names=target,initial_positions=initial_positions)
            # Preserve unknown contact/grasp truth instead of treating API
            # failures in the reused helper as negative observations.
            for name in primary:
                try:
                    gripper,obj = env.env.robots[0].gripper,env.env.get_object(name)
                    row["robot_contacts"][name] = bool(env.env.check_contact(gripper,obj))
                    row["robot_grasps"][name] = bool(env.env._check_grasp(gripper=gripper,object_geoms=obj))
                except Exception:
                    row["robot_contacts"][name] = row["robot_grasps"][name] = None
            row.update(executed_action=action,sim_time_s=float(env.env.sim.data.time),host_time_s=time.time())
            telemetry.append(row)
            if writer:
                writer.append_data(np.ascontiguousarray(obs["agentview_image"][::-1,::-1]))
            timings["io_ms"] += (time.perf_counter()-tick)*1000
        capture(0,None)
        steps,done = 0,False
        while steps < LIMITS[suite] and not done:
            index = len(infers)
            model_obs = {key:np.asarray(value) for key,value in obs.items() if isinstance(value,np.ndarray)}
            noise_seed = seed(suite,task,init,index)
            if save_observations:
                tick = time.perf_counter()
                np.savez_compressed(directory/f"observation_{index:04d}.npz",**model_obs,instruction=np.asarray(definition.language),seed=np.asarray(noise_seed))
                timings["io_ms"] += (time.perf_counter()-tick)*1000
            tick = time.perf_counter()
            response = rpc(port,dict(op="predict",obs=model_obs,instruction=definition.language,suite=suite,seed=noise_seed))
            roundtrip = (time.perf_counter()-tick)*1000
            timings["rpc_ms"] += roundtrip
            actions = np.asarray(response.pop("actions"),np.float32)
            if actions.shape != (horizon,7) or not np.isfinite(actions).all():
                raise ValueError("Invalid native action output")
            if response["input_hash"] != input_digest(model_obs,definition.language):
                raise ValueError("Model input differs from captured policy observation")
            record = dict(response,request=index,step=steps,executed_length=0,client_roundtrip_ms=roundtrip,
                          actions=actions.tolist(),seed=noise_seed)
            infers.append(record)
            for action in actions[:min(execute,LIMITS[suite]-steps)]:
                command = action.copy()
                if condition["model"]=="db-cogact":
                    command[-1] = -1. if command[-1] < 0 else 1.
                tick = time.perf_counter()
                obs,_,done,_ = env.step(command.tolist())
                timings["environment_step_ms"] += (time.perf_counter()-tick)*1000
                steps += 1
                record["executed_length"] += 1
                capture(steps,command.tolist())
                if done:
                    break
        tick = time.perf_counter()
        replay.assign_semantic_phases(telemetry)
        for state in telemetry:
            available = any(value is not None for value in state["robot_contacts"].values())
            grasp_available = any(value is not None for value in state["robot_grasps"].values())
            state.update(contact_truth_available=available,grasp_truth_available=grasp_available,
                         phase_interpretation="Evidence-backed heuristic, not a causal failure diagnosis")
            if state["semantic_phase"]=="grasp_or_manipulate" and not grasp_available and not any(value for value in state["robot_contacts"].values() if value is not None):
                state["semantic_phase"] = "unknown"
                state["phase_unknown_reason"] = "No grasp truth or positive contact evidence"
        for filename,rows in (("telemetry.jsonl",telemetry),("infer.jsonl",infers)):
            (directory/filename).write_text("".join(json.dumps(row)+"\n" for row in rows))
        if writer:
            writer.close()
            writer = None
            import subprocess
            subprocess.run(["ffmpeg","-v","error","-i",str(directory/"video.mp4"),"-f","null","-"],check=True,
                           stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        timings["io_ms"] += (time.perf_counter()-tick)*1000
        phase_order = ["initial","approach","contact","grasp_or_manipulate","transport","goal_alignment","release_or_final_action","success"]
        known_phases = [row["semantic_phase"] for row in telemetry if row["semantic_phase"] in phase_order]
        result = dict(valid=True,success=bool(done),suite=suite,task=task,init=init,condition=condition["id"],
                      model=condition["model"],manifest_hash=manifest_hash,init_vector_sha256=init_hash,
                      actions=steps,predictions=len(infers),wall_s=time.perf_counter()-begin,timing_ms=timings,
                      video=video,save_observations=save_observations,environment_reused=not created,
                      furthest_phase=max(known_phases,key=phase_order.index) if known_phases else "unknown",
                      final_phase=telemetry[-1]["semantic_phase"],
                      phase_caveat="Farthest evidence-backed heuristic phase is not the cause of failure",
                      camera_rendering="policy input cameras remain enabled in every environment step")
        atomic_json(directory/"result.json",result)
        atomic_json(directory/"complete.json",dict(valid=True,manifest_hash=manifest_hash,video_decoded=video))
        return result
    finally:
        if writer:
            writer.close()
        pool.release(suite,task)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--condition",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--port",required=True,type=int)
    parser.add_argument("--stage",choices=("screen","confirm","profile"),default="screen")
    parser.add_argument("--variant",choices=("full_io","lean_io","reuse_env","two_workers"))
    parser.add_argument("--shard-index",type=int,default=0)
    parser.add_argument("--shards",type=int,default=1)
    parser.add_argument("--launch",action="store_true")
    parser.add_argument("--vision-parity")
    parser.add_argument("--environment-reuse",choices=("on","off"))
    args = parser.parse_args()
    manifest = read_manifest(args.manifest,verify_sources=args.launch)
    condition = next(c for c in manifest["conditions"] if c["id"]==args.condition)
    variant = next((v for v in manifest["e2"]["variants"] if v["id"]==args.variant),None)
    if args.stage=="profile":
        if variant is None:
            raise ValueError("Profile requires frozen --variant")
        scenes = manifest["e2"]["episodes"]
    else:
        scenes = [dict(suite=suite,task=task,init=init) for suite in manifest["suites"] for task in manifest["tasks"]
                  for init in manifest["screen_inits" if args.stage=="screen" else "confirm_inits"]]
    # Assign whole tasks to a worker so consecutive init0/init1 can reuse the
    # same environment even in the two-worker profiling arm.
    task_keys = list(dict.fromkeys((scene["suite"],scene["task"]) for scene in scenes))
    owned = {key for index,key in enumerate(task_keys) if index%args.shards==args.shard_index}
    scenes = [scene for scene in scenes if (scene["suite"],scene["task"]) in owned]
    if not args.launch:
        print(json.dumps(dict(dry_run=True,condition=condition["id"],stage=args.stage,episodes=len(scenes),variant=variant)))
        return
    if condition["experiment"]=="E3" and condition["skip"]:
        parity = json.loads(Path(args.vision_parity).read_text()) if args.vision_parity else {}
        if not parity.get("passed") or parity.get("model")!=condition["model"] or parity.get("manifest_hash")!=manifest["manifest_hash"]:
            raise ValueError("Masked vision rollout requires matching real Full parity artifact")
    from .reuse_gate import quality_reuse
    pool = Environments(reuse=quality_reuse(manifest,args.environment_reuse,variant,args.output,condition["model"]))
    try:
        for scene in scenes:
            directory = Path(args.output)/condition["id"]/(args.variant or "episodes")/scene["suite"]/f"task{scene['task']:02d}"/f"init{scene['init']:02d}"
            with lock(str(directory)+".lock"):
                if (directory/"complete.json").exists():
                    previous = json.loads((directory/"complete.json").read_text())
                    if previous.get("manifest_hash") != manifest["manifest_hash"]:
                        raise ValueError("Completed result belongs to another manifest")
                    continue
                video = variant["video"]=="all" if variant else (scene["task"]==0 and scene["init"]==0)
                for attempt in range(manifest["technical_max_attempts"]):
                    if directory.exists():
                        previous = directory.with_name(directory.name+f".invalid.{uuid.uuid4().hex}")
                        directory.rename(previous)
                    try:
                        result = run_episode(directory,args.port,condition,**scene,pool=pool,manifest_hash=manifest["manifest_hash"],
                                             video=video,save_observations=bool(variant and variant["save_observations"]))
                        print(json.dumps(result),flush=True)
                        break
                    except Exception as exc:
                        directory.mkdir(parents=True,exist_ok=True)
                        atomic_json(directory/"technical_error.json",dict(valid=False,error=repr(exc),attempt=attempt+1))
                        pool.close()
                        if attempt+1==manifest["technical_max_attempts"]:
                            raise
    finally:
        pool.close()


if __name__ == "__main__":
    main()
