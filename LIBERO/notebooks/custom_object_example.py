"""Register custom LIBERO objects and generate a task BDDL file.

Run from any directory:
    python notebooks/custom_object_example.py
    python notebooks/custom_object_example.py --render
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from robosuite.models.objects import MujocoXMLObject

from libero.libero.envs.base_object import register_object
from libero.libero.utils.bddl_generation_utils import (
    get_xy_region_kwargs_list_from_regions_info,
)
from libero.libero.utils.mu_utils import InitialSceneTemplates, register_mu
from libero.libero.utils.task_generation_utils import (
    generate_bddl_from_task_info,
    register_task_info,
)


NOTEBOOK_DIR = Path(__file__).resolve().parent
ASSET_DIR = NOTEBOOK_DIR / "custom_assets"


class CustomObject(MujocoXMLObject):
    """Base class for XML assets stored outside the LIBERO package."""

    def __init__(self, xml_path: Path, name: str, joints=None):
        xml_path = xml_path.resolve()
        if not xml_path.is_file() or xml_path.suffix != ".xml":
            raise FileNotFoundError(f"Expected a MuJoCo XML file, got: {xml_path}")
        if joints is None:
            joints = [dict(type="free", damping="0.0005")]

        super().__init__(
            str(xml_path),
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()
        self.object_properties = {"vis_site_names": {}}


@register_object
class LiberoMug(CustomObject):
    def __init__(self, name="libero_mug", obj_name="libero_mug"):
        del obj_name  # Kept for compatibility with LIBERO's object constructor API.
        super().__init__(ASSET_DIR / "libero_mug" / "libero_mug.xml", name)
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None


@register_object
class LiberoMugYellow(CustomObject):
    def __init__(self, name="libero_mug_yellow", obj_name="libero_mug_yellow"):
        del obj_name
        super().__init__(
            ASSET_DIR / "libero_mug_yellow" / "libero_mug_yellow.xml", name
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None


@register_mu(scene_type="kitchen")
class KitchenDemoScene(InitialSceneTemplates):
    def __init__(self):
        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info={"kitchen_table": 1, "wooden_cabinet": 1},
            object_num_info={"libero_mug": 1, "libero_mug_yellow": 1},
        )

    def define_regions(self):
        region_specs = (
            ([0.0, -0.30], "wooden_cabinet_init_region", 0.01, (np.pi, np.pi)),
            ([0.0, 0.0], "libero_mug_init_region", 0.025, None),
            ([-0.1, 0.15], "libero_mug_yellow_init_region", 0.025, None),
        )
        for centroid, name, half_len, yaw in region_specs:
            kwargs = dict(
                region_centroid_xy=centroid,
                region_name=name,
                target_name=self.workspace_name,
                region_half_len=half_len,
            )
            if yaw is not None:
                kwargs["yaw_rotation"] = yaw
            self.regions.update(self.get_region_dict(**kwargs))

        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        return [
            ("On", "libero_mug_1", "kitchen_table_libero_mug_init_region"),
            (
                "On",
                "libero_mug_yellow_1",
                "kitchen_table_libero_mug_yellow_init_region",
            ),
            (
                "On",
                "wooden_cabinet_1",
                "kitchen_table_wooden_cabinet_init_region",
            ),
        ]


def generate_task(output_dir: Path) -> Path:
    register_task_info(
        "libero demo behaviors",
        scene_name="kitchen_demo_scene",
        objects_of_interest=[],
        goal_states=[
            ("Open", "wooden_cabinet_1_top_region"),
            ("In", "libero_mug_yellow_1", "wooden_cabinet_1_top_region"),
        ],
    )
    file_names, failures = generate_bddl_from_task_info(folder=str(output_dir))
    if failures:
        print("Generation failures:", failures)
    if not file_names:
        raise RuntimeError("No BDDL file was generated")

    bddl_path = Path(file_names[0]).resolve()
    print(f"Generated {bddl_path}\n")
    print(bddl_path.read_text())
    return bddl_path


def render_task(bddl_path: Path, image_path: Path):
    # Rendering dependencies are imported only when requested.
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from PIL import Image

    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path), camera_heights=256, camera_widths=256
    )
    try:
        observation = env.reset()
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(observation["agentview_image"][::-1]).save(image_path)
        print(f"Saved rendered scene to {image_path.resolve()}")
    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=NOTEBOOK_DIR / "custom_pddl",
        help="Directory for generated BDDL files.",
    )
    parser.add_argument("--render", action="store_true", help="Render the scene.")
    parser.add_argument(
        "--image-output",
        type=Path,
        default=NOTEBOOK_DIR / "custom_scene.png",
        help="Output path used with --render.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    bddl_path = generate_task(args.output_dir)
    if args.render:
        render_task(bddl_path, args.image_output)


if __name__ == "__main__":
    main()
