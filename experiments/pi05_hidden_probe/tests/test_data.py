"""CPU tests for the experiment's data integrity guarantees."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from experiments.pi05_hidden_probe.data import (
    ManifestError, choose_donors, load_observation, read_manifest,
    split_manifest, split_summary, validate_manifest, write_jsonl,
)
from experiments.pi05_hidden_probe.build_probe_manifest import main as build_manifest

try:
    import numpy as np
except ImportError:
    np = None


def row(sample_id: str, *, label: str = "left", **changes) -> dict:
    value = {
        "sample_id": sample_id,
        "episode_id": f"episode_{sample_id}",
        "scene_id": f"scene_{sample_id}",
        "counterfactual_group_id": f"pair_{sample_id}",
        "original_task_instruction": "pick up the cup",
        "task_type": "spatial_relation",
        "label": label,
        "label_source": "human_verified",
        "camera_paths": {"agent/base_0_rgb": "agent.npy", "wrist/left_wrist_0_rgb": "wrist.npy"},
        "state": [0.0] * 8,
        "coordinate_frame": "policy_input_agent_view",
        "image_convention": "policy_input",
        "is_synthetic": False,
        "question": "Is the target to the left or right of the gripper?",
        "prompt_id": "spatial_v1",
        "labels": ["left", "right"],
    }
    value.update(changes)
    return value


class ManifestTests(unittest.TestCase):
    def test_transitive_group_closure_and_pair_isolation(self):
        rows = [
            row("a", episode_id="shared_episode"),
            row("b", episode_id="shared_episode", scene_id="shared_scene"),
            row("c", scene_id="shared_scene", counterfactual_group_id="pair_cd"),
            row("d", counterfactual_group_id="pair_cd"),
            row("e"), row("f"), row("g"),
        ]
        result = split_manifest(rows, seed=31)
        self.assertEqual(len({r["split"] for r in result[:4]}), 1)
        self.assertEqual(len({r["group_id"] for r in result[:4]}), 1)
        self.assertEqual(split_summary(result)["independent_groups"], 4)
        self.assertEqual({r["split"] for r in result}, {"calibration", "test"})
        reverse = split_manifest(reversed(rows), seed=31)
        self.assertEqual({r["sample_id"]: r["split"] for r in result},
                         {r["sample_id"]: r["split"] for r in reverse})
        self.assertNotIn("split", rows[0])
        subset = validate_manifest(result[:1])
        self.assertEqual(subset[0]["group_id"], result[0]["group_id"])

    def test_single_component_is_not_randomly_split(self):
        with self.assertRaisesRegex(ManifestError, "one connected"):
            split_manifest([row("a", scene_id="same"), row("b", scene_id="same")])

    def test_existing_transitive_split_leakage_is_rejected(self):
        rows = [row("a", episode_id="ab", split="test"),
                row("b", episode_id="ab", scene_id="bc", split="test"),
                row("c", scene_id="bc", split="calibration")]
        with self.assertRaisesRegex(ManifestError, "split leakage"):
            validate_manifest(rows)

    def test_existing_valid_splits_are_preserved(self):
        rows = [row("a", split="test"), row("b", split="calibration")]
        result = split_manifest(rows, seed=99)
        self.assertEqual([r["split"] for r in result], ["test", "calibration"])
        with self.assertRaisesRegex(ManifestError, "partially assigned"):
            validate_manifest([row("a", split="test"), row("b")])

    def test_requires_measured_state_and_coordinate_contract(self):
        for missing in ("state", "coordinate_frame", "image_convention"):
            candidate = row("a")
            del candidate[missing]
            with self.subTest(missing=missing), self.assertRaises(ManifestError):
                validate_manifest([candidate])
        candidate = row("a", state=[float("nan")])
        with self.assertRaisesRegex(ManifestError, "finite"):
            validate_manifest([candidate])

    def test_aliases_and_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "rows.jsonl"
            write_jsonl(manifest, [row("a")])
            result = read_manifest(manifest)[0]
            self.assertEqual(list(result["camera_paths"]), ["observation/image", "observation/wrist_image"])
            self.assertEqual(result["camera_paths"]["observation/image"], str(Path(directory) / "agent.npy"))
            duplicate = row("b")
            duplicate["camera_paths"]["observation/image"] = "also.npy"
            with self.assertRaisesRegex(ManifestError, "duplicate aliases"):
                validate_manifest([duplicate])

    def test_donors_match_question_answer_layout_and_split(self):
        rows = []
        for split in ("calibration", "test"):
            for prompt in ("one", "two"):
                for label in ("left", "right"):
                    rows.append(row(f"{split}_{prompt}_{label}", label=label, split=split,
                                    prompt_id=prompt, question=f"Question {prompt}"))
        donors = choose_donors(rows, seed=5)
        index = {r["sample_id"]: r for r in rows}
        self.assertEqual(donors, choose_donors(reversed(rows), seed=5))
        for sample, donor in donors.items():
            a, b = index[sample], index[donor]
            for key in ("split", "question", "prompt_id", "labels", "task_type"):
                self.assertEqual(a[key], b[key])
            self.assertNotEqual(a["label"], b["label"])

    def test_donors_do_not_fallback_across_split_or_layout(self):
        with self.assertRaisesRegex(ManifestError, "no different-label donor"):
            choose_donors([row("a", split="test"), row("b", label="right", split="calibration")])
        rows = [row("a", split="test"), row("b", label="right", split="test",
                image_masks={"agent": True, "wrist": False})]
        with self.assertRaisesRegex(ManifestError, "both physical cameras active"):
            choose_donors(rows)

    def test_spatial_donors_require_same_coordinate_frame(self):
        rows = [row("a", split="test"), row("b", split="test", label="right",
                coordinate_frame="robot_base")]
        with self.assertRaisesRegex(ManifestError, "no different-label donor"):
            choose_donors(rows)

    def test_cli_writes_group_summary_and_check_only_does_not_write(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.jsonl"
            output = Path(directory) / "processed" / "manifest.jsonl"
            write_jsonl(source, [row("a"), row("b"), row("c")])
            self.assertEqual(build_manifest(["--input", str(source), "--output", str(output)]), 0)
            self.assertEqual(len(read_manifest(output)), 3)
            summary = json.loads(output.with_name("split_manifest.json").read_text())
            self.assertEqual(summary["independent_groups"], 3)
            untouched = Path(directory) / "untouched.jsonl"
            self.assertEqual(build_manifest(["--input", str(source), "--output", str(untouched), "--check-only"]), 0)
            self.assertFalse(untouched.exists())


@unittest.skipIf(np is None, "numpy is required for observation array tests")
class ObservationTests(unittest.TestCase):
    def test_npz_uses_only_original_instruction_and_preserves_orientation(self):
        with tempfile.TemporaryDirectory() as directory:
            pixels = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
            observation_path = Path(directory) / "observation.npz"
            np.savez(observation_path, **{
                "observation/image": pixels, "observation/wrist_image": pixels + 1,
                "observation/state": np.arange(8, dtype=np.float32),
                "prompt": np.array({"unsafe": "do not read pickle"}, dtype=object),
            })
            sample = row("a", observation_path=str(observation_path))
            del sample["camera_paths"], sample["state"]
            loaded = load_observation(sample)
            self.assertEqual(loaded["prompt"], "pick up the cup")
            self.assertEqual(set(loaded), {"observation/image", "observation/wrist_image", "observation/state", "prompt"})
            np.testing.assert_array_equal(loaded["observation/image"], pixels)
            np.testing.assert_array_equal(loaded["observation/state"], np.arange(8))

    def test_npz_rejects_object_arrays_and_missing_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            sample = row("a", observation_path=str(path))
            del sample["camera_paths"], sample["state"]
            values = {"observation/image": np.zeros((3, 3, 3), dtype=np.uint8),
                      "observation/wrist_image": np.zeros((3, 3, 3), dtype=np.uint8)}
            np.savez(path, **values)
            with self.assertRaisesRegex(ManifestError, "missing required arrays"):
                load_observation(sample)
            values["observation/state"] = np.array([{}], dtype=object)
            np.savez(path, **values)
            with self.assertRaisesRegex(ManifestError, "allow_pickle=False"):
                load_observation(sample)

    def test_npy_state_and_images_loaded_relative_to_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pixels = np.zeros((3, 4, 3), dtype=np.uint8)
            pixels[0] = 250
            np.save(root / "agent.npy", pixels)
            np.save(root / "wrist.npy", pixels)
            np.save(root / "state.npy", np.arange(8, dtype=np.float32))
            sample = row("a", state_path="state.npy")
            del sample["state"]
            write_jsonl(root / "input.jsonl", [sample])
            loaded = load_observation(read_manifest(root / "input.jsonl")[0])
            np.testing.assert_array_equal(loaded["observation/image"], pixels)
            np.testing.assert_array_equal(loaded["observation/state"], np.arange(8))
            np.save(root / "agent.npy", pixels.astype(np.float32))
            with self.assertRaisesRegex(ManifestError, "uint8 HWC"):
                load_observation(read_manifest(root / "input.jsonl")[0])


if __name__ == "__main__":
    unittest.main()
