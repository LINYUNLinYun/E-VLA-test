import argparse
import json
from pathlib import Path
import tempfile
import unittest

from .common import atomic_json, read_manifest
from .queue import completed_benchmark, completed_quality, execute, plan


def arguments(root, **updates):
    values = dict(operation="run",manifest="results/idea1_v3_1/manifest_v2.json",output=root,
        condition=["pi0_full","pi05_full"],model=None,experiment=None,stage="screen",variants=None,
        workers=1,port_start=9300,gpu="2",vision_parity="parity.json",environment_reuse="off",
        selection=None,bank=None,launch=False)
    return argparse.Namespace(**dict(values,**updates))


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.manifest = read_manifest("results/idea1_v3_1/manifest_v2.json")
        self.condition = self.manifest["conditions"][0]

    def test_unique_ports_dry_commands_and_forwarding(self):
        with tempfile.TemporaryDirectory() as root:
            jobs = plan(arguments(root),self.manifest)
        self.assertEqual([j["port"] for j in jobs],[9300,9301])
        self.assertNotIn("--launch",jobs[0]["command"])
        self.assertIn("--environment-reuse",jobs[0]["command"])

    def test_four_profile_arms_in_order(self):
        with tempfile.TemporaryDirectory() as root:
            jobs = plan(arguments(root,condition=["pi0_full"],stage="profile",variants=["all"]),self.manifest)
        self.assertEqual([j["variant"] for j in jobs],["full_io","lean_io","reuse_env","two_workers"])
        self.assertEqual(len({j["port"] for j in jobs}),4)

    def test_exact_keys_not_count_and_invalid_attempt_excluded(self):
        with tempfile.TemporaryDirectory() as root:
            for s in self.manifest["suites"]:
                for t in self.manifest["tasks"]:
                    p = Path(root)/self.condition["id"]/"episodes"/s/f"task{t:02d}"/"init00"
                    common = dict(manifest_hash=self.manifest["manifest_hash"],suite=s,task=t,init=0)
                    atomic_json(p/"complete.json",dict(common,valid=True))
                    atomic_json(p/"result.json",dict(common,valid=True,condition=self.condition["id"]))
                    atomic_json(p/"identity.json",dict(common,condition=self.condition))
            self.assertTrue(completed_quality(root,self.condition,self.manifest,"screen"))
            self.assertFalse(completed_quality(root,self.condition,self.manifest,"confirm"))
            p = Path(root)/self.condition["id"]/"episodes"/self.manifest["suites"][0]/"task00"/"init00"
            p.rename(p.with_name("init00.invalid.old"))
            self.assertFalse(completed_quality(root,self.condition,self.manifest,"screen"))

    def test_invalid_benchmark_never_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)/self.condition["id"]/"benchmark.json"
            atomic_json(p,dict(warmup=20,samples=99))
            with self.assertRaisesRegex(ValueError,"refusing overwrite"):
                completed_benchmark(root,self.condition,self.manifest)
            self.assertEqual(json.loads(p.read_text())["samples"],99)

    def test_complete_benchmark_skipped(self):
        with tempfile.TemporaryDirectory() as root:
            mh = self.manifest["manifest_hash"]
            p = Path(root)/self.condition["id"]/"benchmark.json"
            atomic_json(p,dict(manifest_hash=mh,condition=self.condition,warmup=20,samples=100,
                protocol=dict(manifest_hash=mh,condition_id=self.condition["id"]),
                rows=[dict(index=i,input_hash=str(i)) for i in range(100)]))
            self.assertTrue(completed_benchmark(root,self.condition,self.manifest))

    def test_first_nonzero_stops_and_receipts_pid_exit(self):
        class Child:
            pid = 12345
            def wait(self):
                return 7
        calls = []
        def launch(command,**kwargs):
            calls.append(command)
            return Child()
        with tempfile.TemporaryDirectory() as root:
            jobs = plan(arguments(root),self.manifest)
            receipt = dict(jobs=jobs,status="planned")
            p = Path(root)/"batch.json"
            self.assertEqual(execute(jobs,receipt,p,popen=launch),7)
            recorded = json.loads(p.read_text())
        self.assertEqual(len(calls),1)
        self.assertEqual(recorded["jobs"][0]["pid"],12345)
        self.assertEqual(recorded["jobs"][0]["exit_code"],7)
        self.assertEqual(recorded["jobs"][1]["status"],"pending")


if __name__ == "__main__":
    unittest.main()
