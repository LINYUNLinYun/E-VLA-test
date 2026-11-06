"""Synthetic CPU registration/routing tests, never model or authorization proof."""
from collections import Counter
from contextlib import contextmanager
import time
import unittest
from unittest.mock import patch
import numpy as np
from preparations.idea1_v3_2.transfer_protocol import digest
from preparations.idea1_v3_2.test_transfer_policy import Backend
from .policy_B_server import BRegisteredPolicy,validate_binding
from .transfer_registration import MODES,GOOGLE_TASKS,SCOPE


def documents():
    cases=[dict(case_id="c%d"%i,environment="calvin",sequence=dict(index=i,tasks=["task"]*5),
        max_steps_per_subtask=360,native_hz=30) for i in range(100)]
    cases += [dict(case_id=task+str(i),environment="simpler",task=task,seed=i,max_steps=80,native_hz=3) for task in GOOGLE_TASKS for i in range(50)]
    cases += [dict(case_id="p%d"%i,environment="pusht",seed=i,stress_projection=True,max_steps=300,native_hz=10) for i in range(100)]
    registration=dict(scope=SCOPE,family="libero_transfer",modes=list(MODES),H=50,R=5,
        denoising={m:None if m.startswith("draft_") else 10 for m in MODES},cases=cases,condition_count=1800,
        policy_reference=dict(manifest_hash="test_model_reference_not_scientific_permission"),source_files={},input_files={})
    registration["registration_hash"]=digest(registration)
    manifest=dict(scope=SCOPE,registration_hash=registration["registration_hash"],
        model_reference_manifest_hash=registration["policy_reference"]["manifest_hash"])
    manifest["manifest_hash"]=digest(manifest)
    authorization=dict(scope=SCOPE,allow_quality=True,registration_hash=registration["registration_hash"],
        service_manifest_hash=manifest["manifest_hash"],model_reference_manifest_hash=registration["policy_reference"]["manifest_hash"],
        gpu=1,gpu_uuid="GPU-test-no-allocation",modes=list(MODES),environments=["calvin","simpler","pusht"],deadline_unix=time.time()+1000)
    return registration,authorization,manifest


class BContract(unittest.TestCase):
    def setUp(self):
        self.registration,self.authorization,self.manifest=documents();self.backend=Backend()
        self.policy=BRegisteredPolicy(self.registration,self.authorization,self.manifest,backend=self.backend)
        self.raw=dict(base_rgb=np.zeros((2,2,3),np.uint8),wrist_rgb=np.zeros((2,2,3),np.uint8),state8=np.zeros(8,np.float32))
    @contextmanager
    def observed(self):
        self.backend.observed=Counter();yield self.backend.observed
    def request(self,**kwargs):
        request=dict(registration_hash=self.registration["registration_hash"],case_id="c0",environment="calvin",
            mode="full_d10",raw=self.raw,instruction="actual original task",seed=7)
        request.update(kwargs)
        with patch("preparations.idea1_v3_2.transfer_policy.route_observations",self.observed):return self.policy.predict(**request)

    def test_plus_or_reference_identity_is_not_B_permission(self):
        with self.assertRaises(PermissionError):validate_binding(self.registration,dict(self.authorization,scope="Plus"),self.manifest)
        with self.assertRaises(PermissionError):validate_binding(self.registration,self.authorization,dict(self.manifest,scope="Plus"))
        with self.assertRaises(PermissionError):validate_binding(self.registration,dict(self.authorization,allow_quality=False),self.manifest)
        with self.assertRaises(PermissionError):validate_binding(self.registration,dict(self.authorization,deadline_unix=0),self.manifest)

    def test_backend_never_constructed_without_launch(self):
        with self.assertRaises(RuntimeError):BRegisteredPolicy(self.registration,self.authorization,self.manifest)

    def test_wrong_registration_case_environment_rejected_before_route(self):
        for fields in (dict(registration_hash="Plus"),dict(case_id="unknown"),dict(environment="libero_plus")):
            with self.assertRaises(PermissionError):self.request(**fields)
        self.assertEqual(self.backend.prepares,[])

    def test_real_source_gate_required_for_quality(self):
        with self.assertRaisesRegex(RuntimeError,"LEGACY_GATE_REQUIRED"):self.request()
        self.assertEqual(self.backend.prepares,[])

    def test_explicit_attribution_and_separate_counter_namespace(self):
        self.policy.inner.legacy_gate_passed=True  # toy route only, not GPU proof
        result=self.request()
        namespace=SCOPE+":"+self.registration["registration_hash"]
        self.assertEqual(result["registration_hash"],self.registration["registration_hash"])
        self.assertEqual(result["counter_namespace"],namespace)
        self.assertEqual(set(result["counters"]),{namespace})
        self.assertEqual(result["attribution"]["case_id"],"c0")
        self.assertEqual(result["attribution"]["service_manifest_hash"],self.manifest["manifest_hash"])
        self.assertNotEqual(result["attribution"]["service_manifest_hash"],result["attribution"]["model_reference_manifest_hash"])
        self.assertEqual(self.backend.prepares[0][1],"actual original task")
        self.assertEqual(len(result["actions"]),50)


if __name__=="__main__":unittest.main()
