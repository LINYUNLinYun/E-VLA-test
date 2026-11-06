"""CPU checks for scientific case identity, resumability, and paired input gates."""
import json
from pathlib import Path
import tempfile
import unittest
from .runner import cases, commands
from types import SimpleNamespace
from .worker import completed, episode_dir, initial_gate
from preparations.idea1_v3_1.common import atomic_json
from .runner import sha


class Protocol(unittest.TestCase):
    def test_validation_splits_are_independent_and_pilot_is_retained(self):
        manifest=dict(suites=["s","o","g","l"],tasks=list(range(10)),screen_inits=[20],confirm_inits=list(range(20,25)))
        pilot=cases(manifest,"pilot"); confirm=cases(manifest,"confirm")
        self.assertEqual(len(pilot),40); self.assertEqual(len(confirm),200)
        self.assertTrue(all(c in confirm for c in pilot))
        left=cases(manifest,"confirm",0,2); right=cases(manifest,"confirm",1,2)
        self.assertEqual(len(left)+len(right),200)
        self.assertFalse(set((c['suite'],c['task']) for c in left)&set((c['suite'],c['task']) for c in right))
        self.assertFalse(set(c['init'] for c in confirm)&set(range(5)))

    def test_nested_worker_partition_preserves_scheduler_assignment(self):
        args=SimpleNamespace(output="/tmp/fastpath",condition="pi0_full",port=9231,workers=2,stage="pilot",shard_index=1,shards=3)
        _,clients=commands(args,{})
        self.assertEqual([int(c[c.index("--shard-index")+1]) for c in clients],[1,4])
        self.assertEqual([int(c[c.index("--shards")+1]) for c in clients],[6,6])

    def test_initial_gate_rejects_unpaired_conditions(self):
        with tempfile.TemporaryDirectory() as root:
            scene=dict(suite="g",task=0,init=20)
            initial_gate(root,"manifest","A",scene,"input",33)
            initial_gate(root,"manifest","B",scene,"input",33)
            with self.assertRaisesRegex(ValueError,"mismatch"):
                initial_gate(root,"manifest","B",scene,"changed",33)
            with self.assertRaisesRegex(ValueError,"mismatch"):
                initial_gate(root,"manifest","B",scene,"input",34)

    def test_valid_failure_is_resumed_and_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            scene=dict(suite="g",task=0,init=20); d=episode_dir(root,"A",scene); d.mkdir(parents=True)
            row=dict(scene,valid=True,success=False,manifest_hash="m",condition="A")
            atomic_json(d/"result.json",row); atomic_json(d/"complete.json",dict(valid=True,manifest_hash="m"))
            self.assertFalse(completed(root,"A",scene,"m"))
            atomic_json(d/"episode_receipt.json",dict(valid=True,manifest_hash="m",condition="A",case=scene,
                artifacts={name:sha(d/name) for name in ("result.json","complete.json")}))
            self.assertTrue(completed(root,"A",scene,"m"))
            with self.assertRaisesRegex(ValueError,"another"):
                completed(root,"A",scene,"other")
            atomic_json(d/"result.json",dict(row,success=True))
            with self.assertRaisesRegex(ValueError,"changed"):
                completed(root,"A",scene,"m")


if __name__=="__main__": unittest.main()
