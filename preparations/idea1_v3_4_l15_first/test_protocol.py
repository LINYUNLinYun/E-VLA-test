import unittest
from unittest.mock import Mock,patch
import numpy as np
import torch
from preparations.idea1_v3_4_followup.backend import Backend,OriginalBackend
from preparations.idea1_v3_4_followup.policy import Policy
from .common import CONFIG
from .controller import available_gpus


class Protocol(unittest.TestCase):
    def test_full_hybrid_v14_cycle_keeps_independent_full_snapshot(self):
        backend=Backend.__new__(Backend);backend.protocol={}
        backend.prepare=Mock(side_effect=lambda obs,prompt,mask:({'input':obs},{}))
        backend.sample=Mock(side_effect=lambda c,seed,n:(np.zeros((50,7),np.float32),torch.zeros(50,32),dict(nodes=[1-j/n for j in range(n+1)])))
        policy=Policy(backend);policy.configure(CONFIG)
        snapshots=[]
        def hybrid(obs,prompt,full,cutoff):
            self.assertEqual(cutoff,15);snapshots.append(full['input'])
            return {'input':obs},dict(old_kv_layers=4,current_kv_layers=14)
        paths=[];ages=[]
        with patch.object(OriginalBackend,'prepare_hybrid',side_effect=hybrid) as called:
            for i in range(7):
                r=policy.predict(i,'prompt','libero_spatial',i)
                paths.append(r['route']['path']);ages.append(r['full_cache_age'])
                if i%3==2:
                    self.assertFalse(r['route']['old_kv_used']);self.assertEqual(r['timing']['old_kv_layers'],0)
                policy.acknowledge(r['actions'][:5],np.tile([0,0,0,1.],(5,1)))
            self.assertEqual(called.call_count,2)
        self.assertEqual(paths,['full','hybrid','v14','full','hybrid','v14','full'])
        self.assertEqual(snapshots,[0,3])
        self.assertEqual(ages[1:3],[dict(rounds=1,env_steps=5),dict(rounds=2,env_steps=10)])
        self.assertEqual([c.args[2] for c in backend.sample.call_args_list],[10,2,2,10,2,2,10])

    def test_gpu_with_another_compute_owner_is_not_available(self):
        with patch('subprocess.check_output',side_effect=['0, UUID0, 1\n1, UUID1, 200\n2, UUID2, 19000\n3, UUID3, 200\n','UUID0, 123\n']) as query:
            devices,free=available_gpus()
        self.assertEqual(free,[1,3])
        for call in query.call_args_list:self.assertEqual(call.args[0][2],'0,1,2,3')


if __name__=='__main__':unittest.main()
