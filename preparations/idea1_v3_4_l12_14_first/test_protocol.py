import unittest
import tempfile,json
from pathlib import Path
from unittest.mock import Mock,patch
import numpy as np
import torch
from preparations.idea1_v3_4_followup.backend import Backend,OriginalBackend
from preparations.idea1_v3_4_followup.policy import Policy
from .common import CONFIGS
from .controller import available_gpus,Scheduler,StartupOOM


class Protocol(unittest.TestCase):
    def test_each_cutoff_cycle_and_reset_keep_independent_full_snapshot(self):
        backend=Backend.__new__(Backend);backend.protocol={}
        backend.prepare=Mock(side_effect=lambda obs,prompt,mask:({'input':obs},{}))
        backend.sample=Mock(side_effect=lambda c,seed,n:(np.zeros((50,7),np.float32),torch.zeros(50,32),dict(nodes=[1-j/n for j in range(n+1)])))
        policy=Policy(backend)
        for cfg in CONFIGS:
            with self.subTest(cutoff=cfg['cutoff']):
                policy.configure(cfg);snapshots=[];paths=[];ages=[]
                def hybrid(obs,prompt,full,cutoff):
                    self.assertEqual(cutoff,cfg['cutoff']);snapshots.append(full['input'])
                    return {'input':obs},dict(old_kv_layers=19-cutoff,current_kv_layers=cutoff-1)
                with patch.object(OriginalBackend,'prepare_hybrid',side_effect=hybrid) as called:
                    for i in range(7):
                        r=policy.predict(i,'prompt','libero_spatial',i)
                        paths.append(r['route']['path']);ages.append(r['full_cache_age'])
                        if i%3==1:self.assertEqual(r['timing']['old_kv_layers'],19-cfg['cutoff'])
                        if i%3==2:
                            self.assertFalse(r['route']['old_kv_used']);self.assertEqual(r['timing']['old_kv_layers'],0)
                        policy.acknowledge(r['actions'][:5],np.tile([0,0,0,1.],(5,1)))
                    self.assertEqual(called.call_count,2)
                self.assertEqual(paths,['full','hybrid','v14','full','hybrid','v14','full'])
                self.assertEqual(snapshots,[0,3])
                self.assertIsNone(ages[0])
                self.assertEqual(ages[1:3],[dict(rounds=1,env_steps=5),dict(rounds=2,env_steps=10)])
                self.assertEqual([c.args[2] for c in backend.sample.call_args_list[-7:]],[10,2,2,10,2,2,10])

    def test_gpu_with_another_compute_owner_is_not_available(self):
        with patch('subprocess.check_output',side_effect=['0, UUID0, 1\n1, UUID1, 200\n2, UUID2, 19000\n3, UUID3, 200\n','UUID0, 123\n']):
            devices,free=available_gpus()
        self.assertEqual(free,[1,3])

    def test_short_idle_gap_does_not_launch_model(self):
        devices={1:dict(uuid='UUID1',memory_mb=1)};clock=[0.]
        with tempfile.TemporaryDirectory() as tmp,patch('preparations.idea1_v3_4_l12_14_first.controller.OUT',Path(tmp)),patch('time.monotonic',side_effect=lambda:clock[0]),patch('preparations.idea1_v3_4_l12_14_first.controller.available_gpus') as query,patch('subprocess.Popen') as launch:
            scheduler=Scheduler()
            for t,free in [(0,[1]),(30,[]),(60,[1]),(90,[1])]:
                clock[0]=t;query.return_value=(devices,free);scheduler.discover()
            launch.assert_not_called()
            self.assertEqual(scheduler.idle_since,{1:60})

    def test_startup_oom_is_archived_before_retry(self):
        with tempfile.TemporaryDirectory() as tmp,patch('preparations.idea1_v3_4_l12_14_first.controller.OUT',Path(tmp)):
            scheduler=Scheduler();wd=Path(tmp)/'workers/gpu1';wd.mkdir(parents=True)
            (wd/'service.log').write_text('torch.OutOfMemoryError: CUDA out of memory')
            worker=dict(gpu=1,wd=wd,service=Mock(),service_log=Mock(),ready=False)
            worker['service'].poll.return_value=1;scheduler.workers=[worker]
            with self.assertRaises(StartupOOM):scheduler.health(worker)
            scheduler.discard_startup_oom(worker)
            self.assertEqual(scheduler.workers,[]);self.assertEqual(scheduler.startup_failures[1],1)
            self.assertTrue((Path(tmp)/'startup_failures/gpu1_attempt1/startup_failure.json').exists())


if __name__=='__main__':unittest.main()
