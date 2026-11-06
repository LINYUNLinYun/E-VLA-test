import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .final_audit import audit, finite, paired, scenes
from .manifest import build
from .reuse_gate import quality_reuse
from .controller import condition_commands


class AuditTests(unittest.TestCase):
    def test_empty_is_never_passed(self):
        with tempfile.TemporaryDirectory() as directory:
            result = audit(build(),directory,False)
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["quality_counts"]),91)
        self.assertGreaterEqual(len(result["missing_requirements"]),182)

    def test_exact_screen_and_confirmation_keys(self):
        manifest = build()
        self.assertEqual(len(scenes(manifest,[0])),40)
        self.assertEqual(len(scenes(manifest,range(5))),200)
        self.assertFalse(finite([[float("nan")]],(1,1)))

    def test_pair_rejects_noise_or_outcome_change(self):
        row = dict(init_vector_sha256="same",success=True,infer=[dict(seed=1,input_hash="a",noise_hash="b",actions=[[0]*7])])
        self.assertTrue(paired(row,row))
        self.assertFalse(paired(row,dict(row,success=False)))
        changed = dict(row,infer=[dict(row["infer"][0],noise_hash="different")])
        self.assertFalse(paired(row,changed))

    def test_quality_default_off_and_e2_overrides(self):
        self.assertFalse(quality_reuse({}))
        self.assertFalse(quality_reuse({"quality_environment_reuse":False}))
        self.assertTrue(quality_reuse({},"off",dict(environment_reuse=True)))
        self.assertFalse(quality_reuse({},"on",dict(environment_reuse=False)))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,"eight-scene"):
                quality_reuse(build(),"on",output=directory,model="pi0")

    def test_controller_forwarding(self):
        args = argparse.Namespace(stage="screen",workers=1,manifest="manifest",port=9131,output="unused",
            operation="run",variant=None,vision_parity=None,environment_reuse="off")
        _,clients = condition_commands(args,build(),build()["conditions"][0])
        index = clients[0].index("--environment-reuse")
        self.assertEqual(clients[0][index+1],"off")

    def test_on_requires_measured_adoption(self):
        evidence = dict(condition="pi0_full",variant="reuse_env",paired_valid_episodes=8,eligible_for_adoption=True)
        with patch("preparations.idea1_v3_1.reuse_gate.summarize",return_value={"evaluation_optimizations":[evidence]}):
            self.assertTrue(quality_reuse({},"on",output="unused",model="pi0"))


if __name__ == "__main__":
    unittest.main()
