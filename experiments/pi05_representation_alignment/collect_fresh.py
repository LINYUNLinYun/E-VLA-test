"""Render color pairs from official initial states unused in the old experiment."""
import hashlib
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from experiments.pi05_hidden_probe.common import write_json
from experiments.pi05_hidden_probe.data import read_manifest, validate_manifest, write_jsonl
from .common import acquire_gpu, plan
from .prepare import OUT, ROOT


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main():
    config = plan()
    spec = config["fresh_test"]
    out = OUT / "fresh_data"
    out.mkdir(exist_ok=True)
    if (out / "complete.json").exists():
        print("existing_fresh_dataset_preserved", flush=True)
        return
    if (out / "manifest.jsonl").exists():
        raise FileExistsError("Preserve the unfinished manifest for inspection")
    forbidden = {tuple(r["origin"][k] for k in ("suite", "task", "init"))
                 for r in read_manifest(spec["forbidden_origins_manifest"])}
    initial_states = list(range(spec["init_offset"], spec["init_offset"] + spec["scenes"]))
    if any((spec["suite"], spec["task"], i) in forbidden for i in initial_states):
        raise ValueError("Fresh test overlaps a previously used official initial state")
    gpu, lease = acquire_gpu(out / "status.json")
    repo = ROOT / "policies/realtime-vla-flash"
    os.environ.update(MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID=str(gpu["index"]),
                      LIBERO_CONFIG_PATH=str(repo / ".libero"))
    sys.path.insert(0, str(repo / "third_party/libero"))
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from preparations.idea1_v2_1.runtime_client import official_init_states
    from preparations.idea1_v2_1.runtime_models import native_observation
    from robosuite import macros
    if macros.IMAGE_CONVENTION != "opengl":
        raise ValueError("Expected raw OpenGL images before policy rotation")
    suite = benchmark.get_benchmark_dict()[spec["suite"]]()
    task = suite.get_task(spec["task"])
    states = official_init_states(suite, spec["task"])
    if max(initial_states) >= len(states):
        raise ValueError(f"Only {len(states)} official states; cannot fabricate fresh initial states")
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    (out / "observations").mkdir(exist_ok=True)
    rows, receipts, previews = [], [], []
    colors = {"red": [.9, .025, .025, 1.], "blue": [.025, .10, .9, 1.]}
    try:
        env.seed(spec["seed"])
        env.reset()
        for init in initial_states:
            obs = env.set_init_state(np.asarray(states[init]).copy())
            for _ in range(10):
                obs, _, _, _ = env.step([0.] * 6 + [-1.])
            sim = env.env.sim
            sim.forward()
            obs = env.env._get_observations(force_update=True)
            robot_hash = array_hash(np.r_[obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]])
            names = {"bowl": "akita_black_bowl_1", "plate": "plate_1"}
            objects = {k: env.env.get_object(v) for k, v in names.items()}
            bowl_geoms = [i for i in range(sim.model.ngeom)
                          if (sim.model.geom_id2name(i) or "").startswith(names["bowl"] + "_")]
            if not bowl_geoms:
                raise ValueError("Missing bowl geometry")
            rng = np.random.default_rng(spec["seed"] + init)
            x, side_y = float(rng.uniform(.18, .20)), float(rng.uniform(.08, .095))
            for index, obj in enumerate(objects.values()):
                qpos = sim.data.get_joint_qpos(obj.joints[0]).copy()
                qpos[0], qpos[1] = x, side_y if index == 0 else -side_y
                sim.data.set_joint_qpos(obj.joints[0], qpos)
            sim.model.geom_matid[bowl_geoms] = -1
            group = f"goal8_fresh_init{init:02d}"
            for color, rgba in colors.items():
                sim.model.geom_rgba[bowl_geoms] = np.asarray(rgba)
                sim.forward()
                current = env.env._get_observations(force_update=True)
                images, state = native_observation(current)
                pixels = [np.asarray(im) for im in images]
                seg = sim.render(camera_name="agentview", height=256, width=256, segmentation=True)[::-1, ::-1]
                mask = (seg[..., 0] == 5) & np.isin(seg[..., 1], bowl_geoms)
                yy, xx = np.where(mask)
                if len(xx) < 600:
                    raise ValueError(f"Insufficient visible bowl pixels: {group}, {len(xx)}")
                rgb = np.median(pixels[0][mask], axis=0)
                channel = 0 if color == "red" else 2
                if rgb[channel] < max(rgb[k] for k in range(3) if k != channel) + 30:
                    raise ValueError("Material label disagrees with rendered color")
                actual_hash = array_hash(np.r_[current["robot0_eef_pos"], current["robot0_eef_quat"], current["robot0_gripper_qpos"]])
                if actual_hash != robot_hash:
                    raise ValueError("Robot changed during a material-only counterfactual")
                path = out / "observations" / (group + "_" + color + ".npz")
                np.savez_compressed(path, **{"observation/image": pixels[0], "observation/wrist_image": pixels[1], "observation/state": state})
                origin = {"suite": spec["suite"], "task": spec["task"], "init": init,
                          "original_benchmark_instruction": task.language,
                          "intervention": "synthetic material/pose counterfactual, not a closed-loop episode"}
                rows.append({"sample_id": group + "_" + color, "episode_id": group, "scene_id": group,
                             "counterfactual_group_id": group + "_color", "observation_path": str(path),
                             "image_convention": "policy_input", "is_synthetic": True,
                             "original_task_instruction": "pick up the bowl", "task_type": "visual_attribute",
                             "label": color, "labels": ["red", "blue"], "label_source": "mujoco_material_and_visible_geometry_segmentation",
                             "question": "answer en What color is the bowl?\n", "prompt_id": "visual_attribute_v1",
                             "object_text": "bowl", "split": "test", "origin": origin})
                receipts.append({"sample_id": rows[-1]["sample_id"], "visible_pixels": len(xx),
                                 "median_visible_rgb": rgb.tolist(), "robot_state_sha256": robot_hash,
                                 "image_sha256": array_hash(pixels[0]), "origin": origin})
                if len(previews) < 4:
                    previews.append(Image.fromarray(pixels[0]))
            write_json(out / "status.json", {"phase": "running", "gpu": gpu, "scene_groups": len(rows) // 2, "expected": spec["scenes"]})
            print("fresh_pair", init, len(rows), flush=True)
    finally:
        env.close()
    normalized = validate_manifest(rows)
    write_jsonl(out / "manifest.jsonl", normalized)
    write_json(out / "render_receipts.json", {"identity": config["identity"], "gpu": gpu,
              "bddl": str(bddl), "bddl_sha256": hashlib.sha256(bddl.read_bytes()).hexdigest(),
              "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "receipts": receipts})
    preview = Image.new("RGB", (512, 512))
    for i, image in enumerate(previews):
        preview.paste(image, ((i % 2) * 256, (i // 2) * 256))
    preview.save(out / "label_preview.png")
    write_json(out / "complete.json", {"complete": True, "identity": config["identity"],
              "scene_groups": spec["scenes"], "images": len(rows), "no_old_initial_state_overlap": True})
    write_json(out / "status.json", {"phase": "complete", "images": len(rows), "gpu": gpu})
    lease.close()


if __name__ == "__main__":
    main()
