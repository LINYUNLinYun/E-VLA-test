"""Experimental instance-local RGB render suppression; never a quality launcher.

Keep Observable.update clocks and noncamera sensors native. Cache RGB only on
intermediate action frames; the final action frame samples at ORIGINAL native
sensor times. A final-state force_update would change baseline pixel inputs.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import random
import sys
import types

import numpy as np

from .common import ROOT,PYTHONS,LIMITS,atomic_json,read_manifest
from .same_process_parity import sha

REVISION = "experimental_sparse_rgb_native_clock_v1"
PHYSICS_FIELDS = ("qpos","qvel","qacc","qacc_warmstart","act","ctrl","qfrc_applied","xfrc_applied","mocap_pos","mocap_quat","userdata")


def known_rgb_code():
    from robosuite.environments.robot_env import RobotEnv
    return next(c for c in RobotEnv._create_camera_sensors.__code__.co_consts
        if isinstance(c,types.CodeType) and c.co_name=="camera_rgb")


def revision():
    site = ROOT/"policies/libero-eval-v21/.venv/lib/python3.11/site-packages/robosuite"
    files = [Path(__file__),Path(__file__).with_name("rollout.py"),
        site/"environments/base.py",site/"environments/robot_env.py",site/"utils/observables.py",
        ROOT/"LIBERO/libero/libero/envs/env_wrapper.py",ROOT/"LIBERO/libero/libero/envs/bddl_base_domain.py",
        ROOT/"policies/realtime-vla-flash/scripts/experiment6/trajectory_replay.py"]
    return dict(name=REVISION,source_sha256={str(p):sha(p) for p in files},
        execution_is_unmodified_frozen_rollout=False,nn_modified=False,
        physics_hash_scope=dict(sim_state="MuJoCo get_state.flatten + simulation time",available_data_fields=list(PHYSICS_FIELDS),
            additional="All noncamera ndarray observations, reward/done and Python/NumPy RNG state"),
        mutation="Profile child wraps frozen rollout.Environments/run_episode/rpc for proxy/instrumentation; only simulation mutation is instance camera Observable._sensor substitution after unchanged reset/10 warm steps, and never for video. Observable clocks, physics and noncamera update methods remain native.",
        freshness="Original camera sampling on final action frame; never force a final-state image that differs from native cadence.")


def physics_hash(env,obs,reward,done,camera_names):
    from preparations.idea1_v2_1.runtime_client import value_digest
    sim = env.env.sim
    arrays = {key:np.asarray(getattr(sim.data,key)).copy() for key in PHYSICS_FIELDS if hasattr(sim.data,key)}
    assert "qpos" in arrays and "qvel" in arrays
    state = np.asarray(sim.get_state().flatten())
    assert np.isfinite(state).all() and all(np.isfinite(value).all() for value in arrays.values()) and np.isfinite(float(reward))
    noncamera = {k:np.asarray(v) for k,v in obs.items() if isinstance(v,np.ndarray) and k not in camera_names and k!="image-state"}
    return dict(physics=value_digest(dict(state=state,time=float(sim.data.time),arrays=arrays)),
        noncamera=value_digest(noncamera),reward=float(reward),done=bool(done),
        rng=hashlib.sha256(pickle.dumps((random.getstate(),np.random.get_state()),protocol=5)).hexdigest())


class SparseCamera:
    """Reversible sensor hook; construction/imports never create an environment."""
    def __init__(self,inner):
        self.inner,self.active,self.records = inner,False,[]
        self.fresh,self.skipped,self.original = {},{},{}

    def install(self):
        # Public set_sensor validates by rendering immediately. Direct instance
        # substitution avoids that extra render and is explicitly revisioned.
        from robosuite.utils.observables import NO_CORRUPTION,NO_FILTER,NO_DELAY
        assert self.inner.renderer=="mujoco" and not self.inner.viewer_get_obs
        assert not any(self.inner.camera_depths) and all(s is None for s in self.inner.camera_segmentations)
        expected = {name+"_image" for name in self.inner.camera_names}
        cameras = {name:o for name,o in self.inner._observables.items() if o.modality=="image"}
        assert set(cameras)==expected and cameras
        for name,observable in cameras.items():
            sensor = observable._sensor
            assert sensor.__name__=="camera_rgb" and sensor.__code__ is known_rgb_code()
            assert observable._corrupter is NO_CORRUPTION and observable._filter is NO_FILTER and observable._delayer is NO_DELAY
            assert observable.is_enabled() and observable.is_active()
            assert observable._sampling_timestep==1./self.inner.control_freq
            cached = np.asarray(observable.obs).copy()
            assert cached.dtype==np.uint8 and cached.ndim==3 and cached.shape[-1]==3
            self.original[name] = (observable,sensor)
            self.fresh[name],self.skipped[name] = 0,0
            def replacement(cache,_name=name,_sensor=sensor,_cached=cached):
                if self.active:
                    self.skipped[_name] += 1
                    return _cached
                value = _sensor(cache)
                _cached[...] = value
                self.fresh[_name] += 1
                return value
            replacement.__modality__ = "image"
            observable._sensor = replacement

    def restore(self):
        for observable,sensor in self.original.values():
            observable._sensor = sensor
        self.original.clear()
        self.active = False


class StepProxy:
    def __init__(self,env,sparse=False,video=False):
        self.wrapped,self.sparse,self.video = env,bool(sparse and not video),video
        self.camera = SparseCamera(env.env)
        self.started,self.remaining,self.warm_steps = False,0,0
        self.quality_steps,self.rows = 0,[]
        self.fresh_at_last = None
        self.camera_names = {name for name,o in env.env._observables.items() if o.modality=="image"}

    def __getattr__(self,name):
        return getattr(self.wrapped,name)

    def begin_chunk(self,length):
        assert length>0 and self.warm_steps==10 and self.remaining==0
        if self.sparse and not self.started:
            self.camera.install()
        if self.sparse and self.started:
            assert all(self.camera.fresh[k]>v for k,v in self.fresh_at_last.items()),"No original fresh RGB sample on final frame"
        self.started,self.remaining = True,int(length)
        self.fresh_at_last = dict(self.camera.fresh)

    def step(self,action):
        if self.started:
            assert self.remaining>0,"No camera chunk declaration for action"
            self.camera.active = self.sparse and self.remaining>1
        result = self.wrapped.step(action)
        obs,reward,done,_ = result
        row = physics_hash(self.wrapped,obs,reward,done,self.camera_names)
        row.update(index=len(self.rows),phase="quality" if self.started else "warmup",action=np.asarray(action).tolist())
        self.rows.append(row)
        if self.started:
            self.remaining -= 1
            self.quality_steps += 1
            if self.sparse and self.remaining==0:
                assert all(self.camera.fresh[k]>v for k,v in self.fresh_at_last.items()),"Boundary camera did not sample"
        else:
            self.warm_steps += 1
        return result

    def restore(self):
        self.camera.restore()


def worker(args):
    """Opt-in profile-only child; frozen NN/server and manifest stay untouched."""
    from . import rollout
    manifest = read_manifest(args.manifest,verify_sources=True)
    receipt = json.loads((Path(args.root)/"probe.json").read_text())
    assert receipt["coordinator_pid"]==os.getppid() and receipt["manifest_hash"]==manifest["manifest_hash"]
    assert receipt["execution_revision"]==revision()
    condition = next(c for c in manifest["conditions"] if c["id"]==args.model+"_full")
    assert not condition["skip"] and not condition["reference"]
    original_pool,original_episode,original_rpc = rollout.Environments,rollout.run_episode,rollout.rpc
    context = {}
    class ProbePool(original_pool):
        def acquire(self,suite,task):
            native,definition,bddl,selected,created = super().acquire(suite,task)
            proxy = StepProxy(native,sparse=args.arm=="sparse",video=context["video"])
            context["proxy"] = proxy
            return proxy,definition,bddl,selected,created
    def episode(*values,**keywords):
        bound = inspect.signature(original_episode).bind(*values,**keywords)
        bound.apply_defaults()
        directory = Path(bound.arguments["directory"])
        context.update(directory=directory,video=bound.arguments["video"],requests=0,suite=bound.arguments["suite"])
        try:
            result = original_episode(*values,**keywords)
            proxy = context["proxy"]
            assert len(proxy.rows)==10+result["actions"]
            with (directory/"probe_physics.jsonl").open("x") as stream:
                stream.write("".join(json.dumps(row)+"\n" for row in proxy.rows))
            execution = dict(revision=revision(),arm=args.arm,video_original_all_frame_bypass=bool(context["video"]),
                warm_steps=proxy.warm_steps,physics_steps=len(proxy.rows),fresh_rgb_samples=proxy.camera.fresh,
                skipped_rgb_samples=proxy.camera.skipped,requests=context["requests"],worker_pid=os.getpid())
            atomic_json(directory/"probe_execution.json",execution)
            for name in ("identity.json","result.json"):
                row = json.loads((directory/name).read_text())
                row["active_environment_execution_revision"] = execution
                if name=="result.json":
                    row["camera_rendering"] = "Probe: native Observable sampling cadence; unused intermediate RGB cached only in sparse non-video arm"
                    result.update(row)
                atomic_json(directory/name,row)
            return result
        except BaseException as exc:
            directory.mkdir(parents=True,exist_ok=True)
            atomic_json(directory/"probe_failure.json",dict(error=repr(exc),revision=revision()))
            raise
        finally:
            if context.get("proxy") is not None:
                context["proxy"].restore()
    def rpc(port,request):
        if request["op"]=="predict":
            proxy = context["proxy"]
            R = condition["execute_length"]
            R = R[context["suite"]] if isinstance(R,dict) else R
            proxy.begin_chunk(min(R,LIMITS[context["suite"]]-proxy.quality_steps))
            directory = context["directory"]/"probe_inputs"
            directory.mkdir(exist_ok=True)
            np.savez_compressed(directory/f"request{context['requests']:04d}.npz",**request["obs"],
                instruction=np.asarray(request["instruction"]),seed=np.asarray(request["seed"]))
            context["requests"] += 1
        return original_rpc(port,request)
    rollout.Environments,rollout.run_episode,rollout.rpc = ProbePool,episode,rpc
    sys.argv = ["profile-probe","--manifest",args.manifest,"--condition",condition["id"],"--port",str(args.port),
        "--output",str(Path(args.root)/args.arm),"--stage","profile","--variant","two_workers",
        "--shard-index",str(args.shard_index),"--shards","2","--launch"]
    try:
        rollout.main()
    finally:
        rollout.Environments,rollout.run_episode,rollout.rpc = original_pool,original_episode,original_rpc
