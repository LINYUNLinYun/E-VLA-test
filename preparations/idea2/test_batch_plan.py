"""Planning must stay executable on a machine with no CUDA libraries or GPUs."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import batch_plan


class BatchPlanTests(unittest.TestCase):
    def args(self, *extra):
        return batch_plan.parser().parse_args(["--output", "/tmp/idea2-batch-plan-test", *extra])

    def test_default_matrix_and_no_run(self):
        plan = batch_plan.build_plan(self.args())
        self.assertEqual(plan["status"], "prepared_not_run")
        self.assertEqual(len(plan["jobs"]), 36)
        self.assertFalse(plan["gpu_assigned"])
        self.assertEqual({x["stage"] for x in plan["jobs"]}, {"smoke", "performance", "profile"})
        self.assertEqual(len({x["output"] for x in plan["jobs"]}), 36)
        self.assertTrue(all(x["status"] == "not_started" for x in plan["jobs"]))
        self.assertFalse(plan["diagnostic_batch_drift_requested"])
        self.assertTrue(all(x["numerical_policy"] == "strict_b1" for x in plan["jobs"]))
        self.assertTrue(all("--diagnostic-batch-drift" not in x["argv"] for x in plan["jobs"]))

    def test_diagnostic_opt_in_never_weakens_b1_or_changes_tolerance(self):
        plan = batch_plan.build_plan(self.args("--diagnostic-batch-drift"))
        self.assertTrue(plan["diagnostic_batch_drift_requested"])
        self.assertEqual(plan["correctness_gate"]["max_abs"], 1e-5)
        self.assertEqual(plan["correctness_gate"]["mae"], 1e-6)
        for row in plan["jobs"]:
            diagnostic = row["batch_size"] > 1
            self.assertEqual(row["diagnostic_batch_drift"], diagnostic)
            self.assertEqual("--diagnostic-batch-drift" in row["argv"], diagnostic)
            self.assertEqual(row["numerical_policy"],
                             "same_batch_diagnostic" if diagnostic else "strict_b1")
        self.assertIn("remains failed", batch_plan.render_commands(plan))

    def test_diagnostic_b1_only_plan_refused(self):
        with self.assertRaisesRegex(ValueError, "B>1"):
            batch_plan.build_plan(self.args("--diagnostic-batch-drift", "--batch-sizes", "1"))

    def test_all_generated_commands_match_benchmark_cli(self):
        from .batch_bench import parser, validate_args

        for flags in ((), ("--diagnostic-batch-drift", "--green-sm-mins", "16", "24")):
            plan = batch_plan.build_plan(self.args(*flags))
            for row in plan["jobs"]:
                argv = row["argv"]
                arguments = parser().parse_args(argv[argv.index("-m") + 2:])
                validate_args(arguments)
                self.assertEqual(arguments.diagnostic_batch_drift, row["diagnostic_batch_drift"])

    def test_profile_keeps_nodes_and_live_ranges_without_precapture(self):
        rows = batch_plan.build_plan(self.args())["jobs"]
        for row in rows:
            if row["stage"] == "profile":
                self.assertIn("--cuda-graph-trace=node:host-only:nvtx-live", row["argv"])
                self.assertIn("--profile", row["argv"])
            else:
                self.assertNotIn("--profile", row["argv"])

    def test_paired_performance_has_100_pairs(self):
        rows = batch_plan.build_plan(self.args("--no-profiles"))["jobs"]
        for row in rows:
            if row["stage"] == "performance":
                self.assertEqual(row["argv"][row["argv"].index("--pairs") + 1], "100")

    def test_invalid_performance_scale_refused(self):
        with self.assertRaisesRegex(ValueError, "100 pairs"):
            batch_plan.build_plan(self.args("--pairs", "8"))

    def test_plan_only_creates_documents_without_subprocesses(self):
        with tempfile.TemporaryDirectory() as directory, patch("subprocess.run") as run:
            output = str(Path(directory) / "plan")
            batch_plan.main(["--output", output, "--batch-sizes", "2", "--no-profiles"])
            self.assertTrue((Path(output) / "commands.md").is_file())
            self.assertTrue((Path(output) / "plan.json").is_file())
            run.assert_not_called()
            with self.assertRaises(FileExistsError):
                batch_plan.main(["--output", output])


if __name__ == "__main__":
    unittest.main()
