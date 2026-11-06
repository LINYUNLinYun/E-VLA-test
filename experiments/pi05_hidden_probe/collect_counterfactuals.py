"""Render explicitly synthetic LIBERO pairs/quads with simulator-backed labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .common import write_json
from .data import split_manifest, split_summary, write_jsonl


def digest_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.jsonl").exists():
        raise FileExistsError("Existing dataset must not be overwritten")
    (out / "observations").mkdir(exist_ok=True)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from preparations.idea1_v2_1.runtime_client import official_init_states
    from preparations.idea1_v2_1.runtime_models import native_observation
    from robosuite.utils.camera_utils import get_camera_transform_matrix
    from robosuite import macros
    if macros.IMAGE_CONVENTION != "opengl":
        raise ValueError("Collector expects native LIBERO OpenGL observations before policy rotation")
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task = suite.get_task(8)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    states = official_init_states(suite, 8)
    if args.scenes > len(states):
        raise ValueError("Requested more independent initial states than available")
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    names = {"bowl": "akita_black_bowl_1", "plate": "plate_1"}
    colors = {"red": [0.9, 0.025, 0.025, 1.0], "blue": [0.025, 0.10, 0.9, 1.0]}
    rows, receipts, previews = [], [], []
    try:
        env.seed(args.seed)
        env.reset()
        for scene in range(args.scenes):
            scene_id = f"goal8_init{scene:02d}"
            obs = env.set_init_state(np.asarray(states[scene]).copy())
            for _ in range(10):
                obs, _, _, _ = env.step([0.] * 6 + [-1.])
            sim = env.env.sim
            # Refresh kinematics once before defining the no-physics baseline;
            # the last env.step observation may precede this final forward.
            sim.forward()
            obs = env.env._get_observations(force_update=True)
            objects = {key: env.env.get_object(name) for key, name in names.items()}
            geoms = {key: [i for i in range(sim.model.ngeom)
                           if (sim.model.geom_id2name(i) or "").startswith(name + "_")]
                     for key, name in names.items()}
            if any(not ids for ids in geoms.values()):
                raise ValueError("No owned geometry found for target objects")
            rng = np.random.default_rng(args.seed + scene)
            # Keep both objects in front of the cabinet and inside the frame.
            x = float(rng.uniform(0.18, 0.20))
            side_y = float(rng.uniform(0.08, 0.095))
            original_qpos = {key: sim.data.get_joint_qpos(obj.joints[0]).copy() for key, obj in objects.items()}
            robot_state_hash = digest_array(np.concatenate([obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]]))
            variants = {}
            for layout, color in [(0, "red"), (0, "blue"), (1, "red")]:
                for index, (key, obj) in enumerate(objects.items()):
                    position = original_qpos[key].copy()
                    position[0] = x
                    position[1] = side_y if (index + layout) % 2 == 0 else -side_y
                    sim.data.set_joint_qpos(obj.joints[0], position)
                sim.model.geom_matid[geoms["bowl"]] = -1
                sim.model.geom_rgba[geoms["bowl"]] = np.asarray(colors[color])
                sim.forward()
                current = env.env._get_observations(force_update=True)
                images, state = native_observation(current)
                policy_images = [np.asarray(image) for image in images]
                # The segmentation pass is transformed identically to the policy RGB.
                seg = sim.render(camera_name="agentview", height=256, width=256, segmentation=True)[::-1, ::-1]
                if seg.shape != (256, 256, 2):
                    raise ValueError(f"Unexpected segmentation layout {seg.shape}")
                matrix = get_camera_transform_matrix(sim, "agentview", 256, 256)
                gripper = matrix @ np.r_[current["robot0_eef_pos"], 1.]
                gripper_x = 255. - float(gripper[0] / gripper[2])
                metadata = {}
                for key in names:
                    mask = (seg[..., 0] == 5) & np.isin(seg[..., 1], geoms[key])
                    yy, xx = np.where(mask)
                    if len(xx) < 600:
                        raise ValueError(f"Insufficient visible target pixels: {scene_id}/{key}/{layout}: {len(xx)}")
                    center_x = float(xx.mean())
                    if abs(center_x - gripper_x) < 25:
                        raise ValueError("Ambiguous near-overlap spatial label")
                    metadata[key] = {"name": names[key], "visible_pixels": len(xx),
                                     "image_centroid_xy": [center_x, float(yy.mean())],
                                     "bbox_xyxy": [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())],
                                     "gripper_x": gripper_x, "side": "left" if center_x < gripper_x else "right",
                                     "world_qpos": sim.data.get_joint_qpos(objects[key].joints[0]).tolist(),
                                     "median_visible_rgb": np.median(policy_images[0][mask], axis=0).tolist()}
                if metadata["bowl"]["side"] == metadata["plate"]["side"]:
                    raise ValueError("2x2 objects must be on opposite sides of gripper")
                rgb = metadata["bowl"]["median_visible_rgb"]
                expected_channel = 0 if color == "red" else 2
                if rgb[expected_channel] < max(rgb[k] for k in range(3) if k != expected_channel) + 30:
                    raise ValueError("Rendered object color does not agree with assigned material label")
                actual_state_hash = digest_array(np.concatenate([current["robot0_eef_pos"], current["robot0_eef_quat"], current["robot0_gripper_qpos"]]))
                if actual_state_hash != robot_state_hash:
                    raise ValueError("Robot state changed during object-only counterfactual rendering")
                stem = f"{scene_id}_layout{layout}_{color}"
                path = out / "observations" / f"{stem}.npz"
                np.savez_compressed(path, **{"observation/image": policy_images[0], "observation/wrist_image": policy_images[1], "observation/state": state})
                variants[(layout, color)] = {"path": str(path), "metadata": metadata}
                receipts.append({"scene_id": scene_id, "layout": layout, "color": color,
                                 "metadata": metadata, "robot_state_sha256": robot_state_hash,
                                 "sim_qpos_sha256": digest_array(sim.data.qpos), "image_sha256": digest_array(policy_images[0])})
                if scene < 4:
                    canvas = Image.fromarray(policy_images[0]).resize((256, 256))
                    draw = ImageDraw.Draw(canvas)
                    draw.line([(gripper_x, 0), (gripper_x, 255)], fill="yellow", width=1)
                    for key, meta in metadata.items():
                        draw.rectangle(meta["bbox_xyxy"], outline="lime", width=1)
                        draw.text((meta["bbox_xyxy"][0], meta["bbox_xyxy"][1]), key + ":" + meta["side"], fill="white")
                    draw.text((2, 240), stem, fill="white")
                    previews.append(canvas)

            def record(task_type, group, variant, object_key, instruction, question, label, labels):
                data = variants[variant]
                rows.append({"sample_id": f"{scene_id}_{task_type}_{len(rows):04d}",
                    "episode_id": scene_id, "scene_id": scene_id, "counterfactual_group_id": scene_id + "_" + group,
                    "observation_path": data["path"], "image_convention": "policy_input", "is_synthetic": True,
                    "original_task_instruction": instruction, "task_type": task_type, "label": label, "labels": labels,
                    "label_source": "mujoco_material_and_visible_geometry_segmentation",
                    "coordinate_frame": "preprocessed_agent_view", "question": question, "prompt_id": task_type + "_v1",
                    "object_text": object_key, "target_object_metadata": data["metadata"][object_key],
                    "origin": {"suite": "libero_goal", "task": 8, "init": scene, "original_benchmark_instruction": task.language,
                               "intervention": "rendered object material/pose and diagnostic task instruction; no closed-loop trajectory claim"}})

            for color in colors:
                record("visual_attribute", "color", (0, color), "bowl", "pick up the bowl",
                       "answer en What color is the bowl?\n", color, ["red", "blue"])
            for layout in [0, 1]:
                record("spatial_relation", "spatial", (layout, "red"), "bowl", "pick up the bowl",
                       "answer en Is the bowl to the left or right of the gripper?\n",
                       variants[(layout, "red")]["metadata"]["bowl"]["side"], ["left", "right"])
                for key in names:
                    record("instruction_conditioned", "instruction_quad", (layout, "red"), key, f"pick up the {key}",
                           "answer en Is the task target to the left or right of the gripper?\n",
                           variants[(layout, "red")]["metadata"][key]["side"], ["left", "right"])
            print(f"rendered {scene_id}: {len(rows)} labeled observations", flush=True)
    finally:
        env.close()
    rows = split_manifest(rows, calibration_fraction=0.3, seed=args.seed)
    write_jsonl(out / "manifest.jsonl", rows)
    write_json(out / "split_manifest.json", split_summary(rows))
    write_json(out / "render_receipts.json", {"bddl": str(bddl), "bddl_sha256": hashlib.sha256(bddl.read_bytes()).hexdigest(),
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": args.seed, "scenes": args.scenes, "samples": len(rows), "receipts": receipts,
        "data_scope": "synthetic counterfactuals derived from 40 official LIBERO initial states; labels from simulator, not a language model"})
    sheet = Image.new("RGB", (256 * 3, 256 * ((len(previews) + 2) // 3)))
    for i, preview in enumerate(previews):
        sheet.paste(preview, ((i % 3) * 256, (i // 3) * 256))
    sheet.save(out / "label_preview.png")
    print(out / "manifest.jsonl", flush=True)


if __name__ == "__main__":
    main()
