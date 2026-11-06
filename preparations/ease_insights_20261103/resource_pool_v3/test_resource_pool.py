"""CPU-only handoff contracts. No driver, scientific data or GPU processes."""
import contextlib
import copy
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from preparations.ease_insights_20261103.resource_pool_v3 import guardian as g, keeper as k
from preparations.ease_insights_20261103.reservoir_v1.test_reservoir import MemoryCUDA, UUID


class StagedMemoryContracts(unittest.TestCase):
    def setUp(self):
        self.guard=patch.object(ctypes,'CDLL',side_effect=AssertionError('RealCUDA forbidden'))
        self.guard.start();self.addCleanup(self.guard.stop)

    def test_six_to_sixteen_to_eight_retains_same_context(self):
        fake=MemoryCUDA();allocation=k.StagedReservoir(UUID,loader=lambda name:fake)
        allocation.acquire();context=allocation.context.value
        self.assertEqual(sum(fake.live.values()),6*1024**3)
        for target in (16384,8192,16384):
            allocation.resize(target)
            self.assertEqual(sum(fake.live.values()),target*1024**2)
            self.assertEqual(allocation.context.value,context)
        self.assertEqual(fake.calls.count('cuCtxCreate_v2'),1)
        allocation.close();self.assertEqual(fake.live,{})

    def test_failed_staging_cleans_only_new_anchor(self):
        fake=MemoryCUDA();fake.fail_above=5*1024**3
        allocation=k.StagedReservoir(UUID,loader=lambda name:fake)
        with self.assertRaises(RuntimeError):allocation.acquire()
        self.assertEqual(sum(fake.live.values()),64*1024**2)
        allocation.close();self.assertEqual(fake.live,{})

    def test_real_CPU_child_protocol_reports_six_then_sixteen(self):
        real_popen=subprocess.Popen
        code=("import argparse\n"
            "from preparations.ease_insights_20261103.resource_pool_v3 import keeper as k\n"
            "from preparations.ease_insights_20261103.reservoir_v1.test_reservoir import MemoryCUDA\n"
            "p=argparse.ArgumentParser()\n"
            "for n in ('gpu-uuid','control-fd','ready-fd','nonce','parent-pid','parent-start-ticks'):\n"
            " p.add_argument('--'+n,required=True)\n"
            "raise SystemExit(k.prior.run(k.prior.base.Config(**vars(p.parse_args())),"
            "allocation_factory=lambda u:k.StagedReservoir(u,loader=lambda n:MemoryCUDA())))\n")
        def child(argv,**kwargs):
            return real_popen([sys.executable,'-B','-u','-c',code,*argv[5:]],**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            client=g.Client(UUID,Path(directory),g.old.bc.legacy.Runtime())
            with patch.object(g.subprocess,'Popen',side_effect=child):
                try:
                    initial=client.start();pid=client.child.pid
                    self.assertEqual(initial['allocated_bytes'],6*1024**3)
                    self.assertEqual(client.resize(16384)['allocated_bytes'],16*1024**3)
                    self.assertEqual(client.resize(8192)['allocated_bytes'],8*1024**3)
                    self.assertTrue(client.live());self.assertEqual(client.child.pid,pid)
                finally:client.close()
            self.assertEqual(client.child.returncode,0)


class OrderingContracts(unittest.TestCase):
    GPU=next(iter(g.old.HOST_KEEPERS))

    def test_old_release_happens_only_after_six_GiB_ready_and_ownership_probe(self):
        events=[];original=g.old.HOST_KEEPERS[self.GPU];old_host,new_host=8001,8002
        ready={'allocated_bytes':6*1024**3}
        class Client:
            def start(self):events.append('new6ready');return ready
            def live(self):return True
            def resize(self,target):events.append('grow16');return {'allocated_bytes':16*1024**3}
        queries=[[(original,450),(old_host,16770),(new_host,6530)],
                 [(original,450),(new_host,6530)],[(original,450),(new_host,16770)]]
        def observe(uuid):events.append('observe');return queries.pop(0)
        def old_idle(spec,observer):events.append('old_idle');return {'owned':True}
        with patch.object(g,'old_idle',side_effect=old_idle):
            result=g.transfer(Client(),{'old_host_pid':old_host},None,self.GPU,
                lambda *a:events.append('release_old'),lambda:events.append('wait_old_exit'),
                lambda:events.append('take_borrower_lock'),observe=observe)
        self.assertEqual(result['new_host_pid'],new_host)
        self.assertEqual(events,['old_idle','new6ready','observe','old_idle','release_old',
            'wait_old_exit','take_borrower_lock','observe','grow16','observe'])

    def test_failed_or_incomplete_staging_never_releases_old_holder(self):
        for broken in ('allocation_fails','wrong_size','foreign_context'):
            events=[];original=g.old.HOST_KEEPERS[self.GPU]
            class Client:
                def start(self):
                    if broken=='allocation_fails':raise RuntimeError('stagingOOM')
                    return {'allocated_bytes':(5 if broken=='wrong_size' else 6)*1024**3}
                def live(self):return True
            rows=[(original,450),(8001,16770),(8002,6530),(9000,20)]
            with patch.object(g,'old_idle',return_value={}):
                with self.assertRaises(RuntimeError):
                    g.transfer(Client(),{'old_host_pid':8001},None,self.GPU,
                        lambda *a:events.append('released'),lambda:None,lambda:None,
                        observe=lambda uuid:rows)
            self.assertEqual(events,[])

    def test_staging_probe_rejects_wrong_old_or_new_memory(self):
        original=g.old.HOST_KEEPERS[self.GPU]
        for rows in ([(original,450),(8001,8192),(8002,6530)],
                     [(original,450),(8001,16770),(8002,5000)]):
            with self.assertRaises(RuntimeError):g.staging_binding(self.GPU,rows,8001)

    def test_previous_owner_must_have_no_scientific_children(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);ack={'pid':2,'proc_start_ticks':20,'allocated_bytes':16*1024**3}
            (p/'status.json').write_text(json.dumps({'phase':'idle_holding_16GiB','time':g.time.time()}))
            (p/'keeper_latest_ack.json').write_text(json.dumps(ack))
            old={'pid':1,'start_ticks':'10','argv':['python','old']}
            spec={'guardian':old,'directory':str(p),'latest_ack':ack}
            class Observer:
                def identity(self,pid):
                    if pid==1:return dict(old,state='S',uid=os.getuid(),children=['2','3'])
                    return {'pid':2,'start_ticks':'20','ppid':1,'state':'S'}
            with self.assertRaisesRegex(RuntimeError,'science'):
                g.old_idle(spec,Observer())


if __name__=='__main__':unittest.main()

