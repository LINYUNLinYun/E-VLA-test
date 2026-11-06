"""CPU checks of mirror guards/transport gate; not actual model parity."""
import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from .policy_plus_mirror import audit_cpu_backend,cross_gate


class MirrorTests(unittest.TestCase):
    def test_cpu_constructor_audit_rejects_runtime(self):
        backend=SimpleNamespace(adapter=SimpleNamespace(backend=SimpleNamespace(runtime=object())))
        with self.assertRaises(RuntimeError):audit_cpu_backend(backend)

    def test_cpu_constructor_no_cuda_allocation(self):
        import torch
        backend=SimpleNamespace(torch=torch,adapter=SimpleNamespace(backend=SimpleNamespace(runtime=None,
            checkpoint={'x':torch.zeros(2)},embeddings={'x':torch.zeros(1)})))
        with patch.object(torch.cuda,'is_initialized',return_value=False):result=audit_cpu_backend(backend)
        self.assertEqual(result['constructor_tensors_checked'],2)
        self.assertTrue(result['all_constructor_tensors_cpu'])

    def test_exactly_twelve_preflight_transport_calls(self):
        calls=[]
        def fake(port,payload,timeout):
            calls.append((port,payload))
            return dict(raw_actions=np.zeros((50,32)),actions=np.zeros((50,7)),input_hash='same',noise_hash='same',observed_calls={})
        record=dict(obs={},instruction='task original',seed=7)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bank.pkl';path.write_bytes(pickle.dumps([record]))
            with patch('preparations.idea1_v2_1.runtime_models.native_observation',return_value=([np.zeros((2,2,3),np.uint8)]*2,np.zeros(8))),patch('preparations.idea1_v3_2_e4_supplement.policy_plus_mirror.rpc',fake):
                result=cross_gate(19143,19146,path)
        self.assertTrue(result['passed']);self.assertEqual(len(calls),12)
        self.assertTrue(all(payload['category']=='preflight' for _,payload in calls))
        self.assertEqual({port for port,_ in calls},{19143,19146})


if __name__=='__main__':unittest.main()
