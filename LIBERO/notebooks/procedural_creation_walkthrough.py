"""Create LIBERO tasks procedurally and write their BDDL definitions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from libero.libero.envs.objects import get_object_dict, get_object_fn
from libero.libero.envs.predicates import get_predicate_fn, get_predicate_fn_dict
from libero.libero.utils.bddl_generation_utils import (
    get_xy_region_kwargs_list_from_regions_info,
)
from libero.libero.utils.mu_utils import InitialSceneTemplates, register_mu
from libero.libero.utils.task_generation_utils import (
    generate_bddl_from_task_info,
    register_task_info,
)


NOTEBOOK_DIR = Path(__file__).resolve().parent


@register_mu(scene_type="kitchen")
class KitchenScene1(InitialSceneTemplates):
    """Initial-state distribution shared by the two example tasks."""

    def __init__(self):
        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info={"kitchen_table": 1, "wooden_cabinet": 1},
            object_num_info={"akita_black_bowl": 1, "plate": 1},
        )

    def define_regions(self):
        region_specs = (
            ([0.0, -0.30], "wooden_cabinet_init_region", 0.01, (np.pi, np.pi)),
            ([0.0, 0.0], "akita_black_bowl_init_region", 0.025, None),
            ([0.0, 0.25], "plate_init_region", 0.025, None),
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
            (
                "On",
                "akita_black_bowl_1",
                "kitchen_table_akita_black_bowl_init_region",
            ),
            ("On", "plate_1", "kitchen_table_plate_init_region"),
            (
                "On",
                "wooden_cabinet_1",
                "kitchen_table_wooden_cabinet_init_region",
            ),
        ]


def show_registry():
    object_dict = get_object_dict()
    print("Available objects:")
    print(object_dict)
    print("\nmoka_pot class:", get_object_fn("moka_pot"))

    print("\nAvailable predicates:")
    print(get_predicate_fn_dict())
    print("\non predicate:", get_predicate_fn("on"))


def register_example_tasks():
    common = dict(
        scene_name="kitchen_scene1",
        objects_of_interest=["wooden_cabinet_1", "akita_black_bowl_1"],
    )
    register_task_info(
        "Your Language 1",
        goal_states=[
            ("Open", "wooden_cabinet_1_top_region"),
            ("In", "akita_black_bowl_1", "wooden_cabinet_1_top_region"),
        ],
        **common,
    )
    register_task_info(
        "Your Language 2",
        goal_states=[
            ("Open", "wooden_cabinet_1_top_region"),
            ("In", "akita_black_bowl_1", "wooden_cabinet_1_bottom_region"),
        ],
        **common,
    )


def generate_tasks(output_dir: Path):
    register_example_tasks()
    file_names, failures = generate_bddl_from_task_info(folder=str(output_dir))
    print("Generated files:", file_names)
    if failures:
        print("Generation failures:", failures)
    if file_names:
        first_file = Path(file_names[0])
        print(f"\nContents of {first_file}:\n")
        print(first_file.read_text())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=NOTEBOOK_DIR / "tmp" / "pddl_files",
        help="Directory for generated BDDL files.",
    )
    parser.add_argument(
        "--show-registry",
        action="store_true",
        help="Print all registered objects and predicates.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.show_registry:
        show_registry()
    generate_tasks(args.output_dir)


if __name__ == "__main__":
    main()
