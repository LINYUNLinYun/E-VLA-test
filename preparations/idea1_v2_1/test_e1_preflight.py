"""CPU-only safety, inventory and experimental-denominator tests."""
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import unittest

SPEC = importlib.util.spec_from_file_location("e1_preflight", Path(__file__).with_name("e1_preflight.py"))
e1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e1)


class PreparationTests(unittest.TestCase):
    def test_dependencies_respect_server_client_optional_boundary(self):
        self.assertNotIn("flash_attn", e1.MODELS["db-cogact"]["modules"])
        self.assertIn("flash_attn", e1.MODELS["db-cogact"]["optional_modules"])
        self.assertNotIn("tyro", e1.MODELS["xiaomi-robotic0"]["modules"])
        self.assertIn("tyro", e1.MODELS["xiaomi-robotic0"]["client_modules"])

    def test_denominators_and_priority(self):
        for model, total in [("xiaomi-robotic0", 2600), ("db-cogact", 2600), ("cogact", 2600), ("isaac-gr00t", 1000)]:
            plan = e1.task_plan(model)
            rows = plan["tasks"]
            self.assertEqual(len(rows), total)
            self.assertEqual(plan["planned_environment_total"], total + 12)
            keys = {(r["condition"], r["suite"], r["task"], r["init"]) for r in rows}
            self.assertEqual(len(keys), total)
            self.assertEqual([r["init"] for r in rows], sorted(r["init"] for r in rows))
            for condition in plan["conditions"]:
                self.assertEqual(sum(r["condition"] == condition["id"] for r in rows), 200)
            self.assertTrue(all(not r["dispatchable"] for r in rows))
            self.assertTrue(all(c["mask"] is None for c in plan["conditions"][1:]))
            self.assertEqual({r["init"] for r in plan["calibration"]}, {5})
            self.assertEqual(len(plan["calibration"]), 8)

    def test_xiaomi_semantics_preserve_shell(self):
        result = e1.validate_mask("xiaomi-robotic0", 36, [4, 20, 21, 36])
        self.assertEqual(result["kv_shell_layers"], [4, 20, 21, 36])
        self.assertEqual(result["fully_eliminated_layers"], [])
        self.assertFalse(result["operator_execution_verified"])
        db = e1.validate_mask("db-cogact", 28, [3, 17, 28])
        self.assertEqual(db["kv_shell_layers"], [3, 17, 28])
        self.assertEqual(db["fully_eliminated_layers"], [])
        for bad in ([1], [2], [3], [37], [4, 4], [True]):
            with self.assertRaises(ValueError):
                e1.validate_mask("xiaomi-robotic0", 36, bad)

    def test_readout_and_budget(self):
        with self.assertRaises(ValueError):
            e1.validate_mask("db-cogact", 28, [28], [28])
        self.assertEqual(e1.prediction_budgets(36)["max_probe_logical_predictions"], 2960)
        self.assertEqual(e1.prediction_budgets(28)["max_probe_logical_predictions"], 2320)

    def test_unknown_native_R_blocks(self):
        result = e1.validate_protocol({"effective_depth": 36, "H": 30, "D": 5})
        self.assertFalse(result["schema_complete"])
        self.assertFalse(result["execution_authorized"])
        self.assertEqual(sum("R_by_suite" in issue for issue in result["issues"]), 4)

    def test_candidates_cannot_be_empty_placeholders(self):
        with self.assertRaises(ValueError):
            e1.validate_candidates("xiaomi-robotic0", 36, [])
        with self.assertRaises(ValueError):
            e1.validate_candidates("db-cogact", 28, [{"id": f"c{i}", "mask": None} for i in range(12)])

    def test_outputs_never_overwrite(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "manifest.json"
            e1.write_new(path, {"first": True})
            with self.assertRaises(FileExistsError):
                e1.write_new(path, {"first": False})
            self.assertEqual(json.loads(path.read_text()), {"first": True})

    def test_metadata_does_not_execute_python(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)
            (path / "modeling.py").write_text("raise RuntimeError('must not execute')\n")
            (path / "config.json").write_text('{"num_hidden_layers":28}')
            result = e1.metadata_inventory(path)
            self.assertEqual(len(result), 2)
            self.assertTrue(all("sha256" in r for r in result))

    def test_environment_probe_never_imports_package_or_pth(self):
        with tempfile.TemporaryDirectory() as name:
            repo = Path(name) / "toy-model"
            env = repo / ".venv"
            binary = env / "bin"
            binary.mkdir(parents=True)
            (binary / "python").symlink_to(sys.executable)
            (env / "pyvenv.cfg").write_text("prompt = toy-model\n")
            site = env / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
            site.mkdir(parents=True)
            (site / "torch.py").write_text("raise RuntimeError('package must never import')\n")
            (site / "unsafe.pth").write_text("import os; raise RuntimeError('pth must never execute')\n")
            result = e1.environment(repo, ["torch", "missing_package"])
            self.assertEqual(result["probe_returncode"], 0)
            self.assertEqual(result["probe"]["module_discoverable"], {"torch": True, "missing_package": False})
            self.assertTrue(result["prompt_matches"])


if __name__ == "__main__":
    unittest.main()
