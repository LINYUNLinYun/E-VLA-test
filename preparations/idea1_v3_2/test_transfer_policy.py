"""CPU routing/input/transport tests; actual model parity is a separate GPU gate."""
from collections import Counter
from contextlib import contextmanager
import json
import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from .transfer_policy import DRAFT_SUITES,MODES,RawTritonBackend,TransferPolicy,V14,validate_raw
from .transfer_prompts import instruction_tokens


class Tensor:
    def detach(self): return self
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return np.zeros((50,32),np.float32)


class Backend:
    protocol = {}
    startup_events = []
    def __init__(self):
        self.observed = Counter()
        self.prepares = []
    def prepare(self,raw,instruction,skip,encode):
        self.prepares.append((raw,instruction,skip,encode))
        self.observed["vision_encoder_calls"] += 1
        self.observed["vlm_prefix_calls"] += int(encode)
        return {},{}
    def learned(self,context,suite):
        self.observed["learned_head_calls"] += 1
        self.head = suite
        return np.zeros((50,7),np.float32),Tensor(),{}
    def sample(self,context,seed,steps):
        self.observed["ae_layer_calls"] += 18*steps
        return np.zeros((50,7),np.float32),Tensor(),dict(actual_NFE=steps,nodes=list(range(11)))


class TransferRouting(unittest.TestCase):
    def setUp(self):
        self.backend = Backend()
        self.policy = TransferPolicy({},"frozen",backend=self.backend)
        self.raw = dict(base_rgb=np.arange(12,dtype=np.uint8).reshape(2,2,3),
                        wrist_rgb=np.zeros((2,2,3),np.uint8),state8=np.zeros(8,np.float32))
    @contextmanager
    def observed(self):
        self.backend.observed = Counter()
        yield self.backend.observed
    def predict(self,mode):
        with patch("preparations.idea1_v3_2.transfer_policy.route_observations",self.observed):
            return self.policy.predict(mode,self.raw,"a",7,category="preflight")

    def test_quality_requires_real_gpu_gate(self):
        with self.assertRaisesRegex(RuntimeError,"LEGACY_GATE_REQUIRED"):
            self.policy.predict("full_d10",self.raw,"a",7)

    def test_images_unchanged_no_native_libero_flip(self):
        self.predict("full_d10")
        np.testing.assert_array_equal(self.backend.prepares[-1][0]["base_rgb"],self.raw["base_rgb"])

    def test_all_six_routes_and_accounting(self):
        for mode in MODES:
            result = self.predict(mode)
            self.assertEqual(result["execute_length"],5)
            self.assertEqual(self.backend.prepares[-1][2],V14 if mode=="v14_d10" else ())
            self.assertEqual(self.backend.prepares[-1][3],mode not in DRAFT_SUITES)
            self.assertEqual(result["observed_calls"]["ae_euler_updates"],0 if mode in DRAFT_SUITES else 10)
            if mode in DRAFT_SUITES:
                self.assertEqual(self.backend.head,DRAFT_SUITES[mode])
                self.assertFalse(result["noise_used_in_inference"])
        self.assertEqual(self.policy.counters["quality"],{})
        self.assertEqual(len(self.policy.counters["preflight"]),6)

    def test_output_python38_pickle_safe_no_numpy_objects(self):
        result = self.predict("draft_goal")
        encoded = pickle.dumps(result,protocol=4)
        self.assertNotIn(b"numpy",encoded)
        self.assertIsInstance(result["actions"],list)
        json.dumps(pickle.loads(encoded),allow_nan=False)

    def test_invalid_state_and_image_dtype(self):
        with self.assertRaises(ValueError):
            validate_raw(dict(self.raw,state8=[float("nan")]*8))
        with self.assertRaises(ValueError):
            validate_raw(dict(self.raw,base_rgb=self.raw["base_rgb"].astype(np.float32)))

    def test_dynamic_injection_preserves_old_prompt_metadata(self):
        backend = RawTritonBackend.__new__(RawTritonBackend)
        _,ids = instruction_tokens("a")
        old = torch.ones((1,2048),dtype=torch.bfloat16)
        storage = SimpleNamespace(checkpoint={"language_embedding_weight":torch.zeros((max(ids)+1,2048),dtype=torch.bfloat16)},
                                  embeddings={("libero_spatial","old"):old})
        backend.adapter = SimpleNamespace(backend=storage,prompt_suites={"old":"libero_spatial"})
        backend.dynamic_prompt_rows = {}
        backend.ensure_prompt("a")
        self.assertIs(storage.embeddings[("libero_spatial","old")],old)
        self.assertEqual(backend.adapter.prompt_suites["old"],"libero_spatial")
        new = storage.embeddings[(backend.adapter.prompt_suites["a"],"a")]
        self.assertEqual(tuple(new.shape),(len(ids),2048))
        self.assertEqual(new.device.type,"cpu")

    def test_failed_route_retains_observed_accounting(self):
        def failed(context,suite):
            raise RuntimeError("bounded test failure")
        self.backend.learned = failed
        with self.assertRaisesRegex(RuntimeError,"bounded test failure"):
            self.predict("draft_goal")
        row = self.policy.counters["preflight"]["draft_goal"]
        self.assertEqual(row["vision_encoder_calls"],1)
        self.assertEqual(row["failed_requests"],1)


if __name__ == "__main__":
    unittest.main()
