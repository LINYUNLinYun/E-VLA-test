"""CPU-only regression for replay deduplication and rate denominators."""
import json
from pathlib import Path
import tempfile
import unittest

from .analyze import analyze


class ReplayStatisticsTests(unittest.TestCase):
    def test_unique_request_source_pairs_and_per_action_gripper_denominator(self):
        rows = []
        for interval in (1,2,4):
            for request in range(8):
                source = request//interval*interval
                age = request-source
                horizons = {}
                for horizon,count in (("first5",1),("full50",2)):
                    horizons[horizon] = {component:dict(rmse=float(age),max_abs=float(age))
                                         for component in ("position","rotation","gripper")}
                    horizons[horizon]["gripper_sign_mismatches"] = count if age else 0
                rows.append(dict(interval=interval,request=request,prefix_source=source,by_horizon=horizons,
                                 serial_async_error=dict(max_abs=0.)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"micro/libero_spatial/e2.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(rows=rows)))
            result = analyze(directory)
        derived = {(row["request_age"],row["horizon"]):row for row in result["e2_age"]}
        self.assertEqual([derived[(age,"first5")]["unique_request_source_pairs"] for age in range(4)],[8,4,2,2])
        first,full = derived[(1,"first5")],derived[(1,"full50")]
        self.assertEqual(first["position_coordinate_denominator"],4*5*3)
        self.assertEqual(full["position_coordinate_denominator"],4*50*3)
        self.assertEqual(first["gripper_sign_mismatches"],4)
        self.assertEqual(first["gripper_action_denominator"],20)
        self.assertAlmostEqual(first["gripper_sign_mismatch_rate"],.2)
        self.assertEqual(full["gripper_action_denominator"],200)
        self.assertAlmostEqual(full["gripper_sign_mismatch_rate"],.04)
        self.assertEqual(first["pooled_position_rmse"],1.)
        self.assertTrue(result["status"]["partial"])


if __name__=="__main__":
    unittest.main()
