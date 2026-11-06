import json
from pathlib import Path
import tempfile
import unittest

from .common import atomic_json, read_manifest, seed
from .manifest import build, masks, validate_mask
from .controller import promotion
from .collect_bank import choose


class ProtocolTests(unittest.TestCase):
    def test_vision_twenty_unique_masks_per_model(self):
        manifest = build()
        for model in ("pi0","pi05"):
            conditions = [c for c in manifest["conditions"] if c["model"]==model]
            self.assertEqual(len(conditions),21)
            self.assertEqual(sum(not c["skip"] for c in conditions),1)
            for keep in (24,21,18,15):
                group = [c for c in conditions if len(c["keep"])==keep]
                self.assertEqual(len(group),5)
                self.assertEqual(len({tuple(c["skip"]) for c in group}),5)
                for condition in group:
                    self.assertEqual(sorted(condition["keep"]+condition["skip"]),list(range(1,28)))
                    self.assertEqual(condition["denoising_steps"],10)

    def test_protected_vlm_indices_and_native_sampling(self):
        manifest = build()
        for c in manifest["conditions"]:
            if c["model"]=="xiaomi-robotic0":
                self.assertFalse(set(c["skip"]) & {1,2,3})
                self.assertEqual(c["denoising_steps"],2)
                self.assertEqual(c["horizon"],10)
            if c["model"]=="db-cogact":
                self.assertFalse(set(c["skip"]) & {1,2})
                self.assertEqual(c["denoising_steps"],10)

    def test_frozen_manifest_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"manifest.json"
            atomic_json(path,build())
            self.assertEqual(read_manifest(path)["relative_success_floor"],.7)
            data = json.loads(path.read_text())
            data["relative_success_floor"] = .2
            atomic_json(path,data)
            with self.assertRaises(ValueError):
                read_manifest(path)

    def test_masks_deterministic_and_strict_one_based(self):
        self.assertEqual(masks(27,12),masks(27,12))
        for invalid in ([0],[28],[1,1],[True]):
            with self.assertRaises(ValueError):
                validate_mask(invalid,27)

    def test_noise_pairing_independent_of_mask_worker(self):
        self.assertEqual(seed("libero_goal",5,1,7),seed("libero_goal",5,1,7))
        self.assertNotEqual(seed("libero_goal",5,1,7),seed("libero_goal",5,1,8))

    def test_incomplete_screen_never_promoted(self):
        with tempfile.TemporaryDirectory() as directory:
            result = promotion(build(),directory)
            self.assertFalse(result["selected_conditions"])
            self.assertEqual(len(result["decisions"]),4)

    def test_promotion_uses_relative_near_floor_and_best_per_budget(self):
        manifest = build()
        candidates = [c for c in manifest["conditions"] if c["model"]=="xiaomi-robotic0" and len(c["skip"])==12 and not c["reference"]]
        scores = {"xiaomi-robotic0_full":36,**{c["id"]:score for c,score in zip(candidates,[22,21,18,0,5],strict=True)}}
        with tempfile.TemporaryDirectory() as directory:
            for cid,score in scores.items():
                for index in range(40):
                    suite = manifest["suites"][index//10]
                    path = Path(directory)/cid/"episodes"/suite/f"task{index%10:02d}"/"init00"
                    atomic_json(path/"result.json",dict(valid=True,success=index<score,suite=suite,task=index%10,init=0,manifest_hash=manifest["manifest_hash"]))
                    atomic_json(path/"complete.json",dict(valid=True))
            result = promotion(manifest,directory)
            selected = result["selected_conditions"]
            self.assertIn("xiaomi-robotic0_full",selected)
            self.assertIn(candidates[0]["id"],selected)  #22/36 qualifies for the frozen .60 confirmation gate.
            self.assertNotIn(candidates[1]["id"],selected)

    def test_real_bank_selection_spans_all_scenes_and_trajectories(self):
        groups = {scene:[dict(input_hash=f"{scene}/{index}",request=index) for index in range(100)]
                  for scene in ((0,0),(0,1),(5,0),(5,1))}
        selected = choose(groups,25,3101)
        self.assertEqual(len({row["input_hash"] for row in selected}),25)
        for scene in groups:
            rows = [row for row in selected if row["input_hash"].startswith(str(scene)+"/")]
            self.assertIn(len(rows),(6,7))
            self.assertLess(min(row["request"] for row in rows),20)
            self.assertGreater(max(row["request"] for row in rows),70)


if __name__ == "__main__":
    unittest.main()
