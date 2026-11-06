"""CPU fake-sim tests using the ACTUAL pinned Observable timer implementation."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import random
import sys
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from .common import ROOT
from . import sparse_camera as wrapper
from . import sparse_camera_probe as probe

OBSERVABLE = ROOT/"policies/libero-eval-v21/.venv/lib/python3.11/site-packages/robosuite/utils/observables.py"
spec = importlib.util.spec_from_file_location("cpu_pinned_observable",OBSERVABLE)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)  # numpy only; no robosuite environment/package import


def camera_sensor(sim):
    def camera_rgb(cache):
        return sim.render()
    camera_rgb.__modality__ = "image"
    return camera_rgb


class FakeSim:
    def __init__(self):
        self.renders = 0
        self.data = SimpleNamespace(time=0.,qpos=np.zeros(3),qvel=np.zeros(3),qacc=np.zeros(3),ctrl=np.zeros(3))
    def render(self):
        self.renders += 1
        return np.full((2,2,3),round(self.data.qpos[0]*10000)%256,np.uint8)
    def get_state(self):
        values = np.r_[self.data.time,self.data.qpos,self.data.qvel]
        return SimpleNamespace(flatten=lambda:values.copy())


class FakeEnv:
    def __init__(self,depth=False,stochastic=False):
        sim = FakeSim()
        camera = native.Observable("agentview_image",camera_sensor(sim),sampling_rate=20)
        def eef(cache):
            return sim.data.qpos.copy()
        eef.__modality__ = "proprio"
        position = native.Observable("robot0_eef_pos",eef,sampling_rate=20)
        if stochastic:
            camera._corrupter = lambda x:x
        self.env = SimpleNamespace(sim=sim,renderer="mujoco",viewer_get_obs=False,control_freq=20,
            camera_depths=[depth],camera_segmentations=[None],camera_names=["agentview"],
            _observables={camera.name:camera,position.name:position})
    def step(self,action):
        for _ in range(25):
            self.env.sim.data.qvel[:] = action[:3]
            self.env.sim.data.ctrl[:] = action[:3]
            self.env.sim.data.qpos += np.asarray(action[:3])*.002
            self.env.sim.data.time += .002
            cache = getattr(self,"cache",{})
            self.cache = cache
            for observable in self.env._observables.values():
                observable.update(.002,cache)
        return {k:o.obs.copy() for k,o in self.env._observables.items()},0.,False,{}


class SparseCameraTests(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules,{"robosuite":ModuleType("robosuite"),
            "robosuite.utils":ModuleType("robosuite.utils"),"robosuite.utils.observables":native})
        self.modules.start()
        self.code = patch.object(wrapper,"known_rgb_code",return_value=camera_sensor(FakeSim()).__code__)
        self.code.start()
    def tearDown(self):
        self.code.stop()
        self.modules.stop()

    def execute(self,sparse,R=5,video=False):
        random.seed(7)
        np.random.seed(7)
        env = FakeEnv()
        proxy = wrapper.StepProxy(env,sparse=sparse,video=video)
        for _ in range(10):
            obs,*_ = proxy.step([0.]*7)
        original_sensor = env.env._observables["agentview_image"]._sensor
        inputs = [copy.deepcopy(obs)]
        before = env.env.sim.renders
        for _ in range(3):
            proxy.begin_chunk(R)
            for _ in range(R):
                obs,*_ = proxy.step([.1,.2,.3,0,0,0,-1])
            inputs.append(copy.deepcopy(obs))
        stats = dict(renders=env.env.sim.renders,warm_renders=before,skipped=dict(proxy.camera.skipped),fresh=dict(proxy.camera.fresh))
        proxy.restore()
        self.assertIs(env.env._observables["agentview_image"]._sensor,original_sensor)
        return inputs,proxy.rows,stats

    def test_exact_boundary_inputs_physics_rng_noncamera_and_native_clocks(self):
        for R in (5,10,12,15,16):
            baseline,base_rows,base_stats = self.execute(False,R)
            sparse,sparse_rows,sparse_stats = self.execute(True,R)
            self.assertEqual(base_rows,sparse_rows)
            self.assertEqual(base_stats["warm_renders"],sparse_stats["warm_renders"])
            self.assertLess(sparse_stats["renders"],base_stats["renders"])
            self.assertGreater(sum(sparse_stats["skipped"].values()),0)
            for a,b in zip(baseline,sparse):
                self.assertEqual(a.keys(),b.keys())
                self.assertTrue(all(np.array_equal(a[k],b[k]) for k in a))

    def test_video_bypass_is_original_every_frame_and_no_hooks(self):
        a,ar,astats = self.execute(False,5,True)
        b,br,bstats = self.execute(True,5,True)
        self.assertEqual(ar,br)
        self.assertEqual(astats,bstats)
        self.assertEqual(bstats["skipped"],{})
        self.assertTrue(all(np.array_equal(x["agentview_image"],y["agentview_image"]) for x,y in zip(a,b)))

    def test_depth_stochastic_unknown_sensor_and_stale_boundary_fail_closed(self):
        for env in (FakeEnv(depth=True),FakeEnv(stochastic=True)):
            proxy = wrapper.StepProxy(env,True)
            for _ in range(10):
                proxy.step([0.]*7)
            with self.assertRaises(AssertionError):
                proxy.begin_chunk(5)
            proxy.restore()
        env = FakeEnv()
        unknown = lambda cache:np.zeros((2,2,3),np.uint8)
        unknown.__modality__ = "image"
        env.env._observables["agentview_image"]._sensor = unknown
        proxy = wrapper.StepProxy(env,True)
        for _ in range(10):
            proxy.step([0.]*7)
        with self.assertRaises(AssertionError):
            proxy.begin_chunk(5)
        proxy.restore()
        env = FakeEnv()
        proxy = wrapper.StepProxy(env,True)
        for _ in range(10):
            proxy.step([0.]*7)
        proxy.begin_chunk(2)
        proxy.step([0.]*7)
        env.env._observables["agentview_image"]._enabled = False
        with self.assertRaises(AssertionError):
            proxy.step([0.]*7)
        proxy.restore()

    def test_forcing_final_state_is_not_native_sampling_time(self):
        env = FakeEnv()
        for _ in range(10):
            env.step([0.]*7)
        obs,*_ = env.step([.1,0,0,0,0,0,0])
        before = obs["agentview_image"].copy()
        env.env._observables["agentview_image"].update(.002,env.cache,force=True)
        after = env.env._observables["agentview_image"].obs
        self.assertFalse(np.array_equal(before,after),"Fresh final-state force changes baseline pixels")

    def test_pair_gate_negative_outcomes_valid_drift_or_hash_changes_fail(self):
        row = dict(success=False,init_vector_sha256="official",infer=[dict(input_hash="pixels",noise_hash="noise",seed=1,
            actions=[[0.]*7],executed_length=1)],physics=[dict(physics="actual",rng="state")],telemetry=[dict(step=0,host_time_s=1.)])
        other = copy.deepcopy(row)
        other["telemetry"][0]["host_time_s"] = 2.
        self.assertTrue(all(probe.compare_pair(row,other).values()))
        other["infer"][0]["actions"][0][0] = 1e-8
        self.assertFalse(probe.compare_pair(row,other)["all_requests_exact"])
        other = copy.deepcopy(row)
        other["physics"][0]["rng"] = "changed"
        self.assertFalse(probe.compare_pair(row,other)["all_physics_noncamera_rng_reward_done_exact"])

    def test_dryrun_no_files_queries_models_or_environments(self):
        with patch("sys.argv",["probe","--manifest","results/idea1_v3_1/manifest_v2.json","--model","pi05","--port","9899"]), \
             patch.object(probe,"run",side_effect=AssertionError("launch")), \
             patch.object(probe.subprocess,"Popen",side_effect=AssertionError("process")), \
             patch.object(probe.subprocess,"check_output",side_effect=AssertionError("GPU query")), \
             patch("sys.stdout",io.StringIO()) as output:
            probe.main()
        result = json.loads(output.getvalue())
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["episodes"],16)
        self.assertEqual(result["quality_episodes"],0)
        self.assertFalse(result["execution_revision"]["execution_is_unmodified_frozen_rollout"])
        self.assertEqual(json.loads(json.dumps(wrapper.revision())),wrapper.revision())


if __name__=="__main__":
    unittest.main()
