"""LIBERO 基础 walkthrough（路径、任务、初始状态和 demonstration）。

直接运行时只做只读检查，不会自动下载数据，也不会启动仿真训练。
渲染、下载和视频导出都通过命令行参数显式开启；可运行 ``--help`` 查看参数。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
# 直接执行 notebooks/quick_walkthrough.py 时，Python 默认只把 notebooks/
# 放入模块搜索路径。这里补上仓库根目录，保证 libero 包可以被找到。
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# OffScreenRenderEnv 使用 EGL 做无窗口渲染。robosuite 会在导入时读取
# MUJOCO_GL，因此要在导入 libero / robosuite 之前设置它。
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from termcolor import colored

from libero.libero import benchmark, get_libero_path


def get_paths():
    """读取 LIBERO 配置中的四类目录路径。"""
    return {
        name: Path(get_libero_path(name))
        for name in ("benchmark_root", "init_states", "datasets", "bddl_files")
    }


def print_paths(paths):
    """把实际使用的路径打印出来，便于发现配置指向了错误目录。"""
    print("Configured LIBERO paths:")
    for name, path in paths.items():
        print(f"\033[33m  {name:14s} {path}\033[0m")


def inspect_benchmark(benchmark_name: str, paths):
    """列出 benchmark 的任务，并检查 BDDL/init state 文件是否存在。"""
    # benchmark_dict 是“名称 -> benchmark 类”的映射，例如 libero_10。
    benchmark_dict = benchmark.get_benchmark_dict()
    if benchmark_name not in benchmark_dict:
        choices = ", ".join(sorted(benchmark_dict))
        raise ValueError(f"Unknown benchmark {benchmark_name!r}; choose from: {choices}")

    benchmark_instance = benchmark_dict[benchmark_name]()
    num_tasks = benchmark_instance.get_num_tasks()
    print(f"\n{benchmark_instance.name} contains {num_tasks} tasks:")

    missing_bddl = []
    missing_states = []
    for task_id, task_name in enumerate(benchmark_instance.get_task_names()):
        task = benchmark_instance.get_task(task_id)
        # 每个任务会引用一个 BDDL（任务目标定义）和一个 init state 文件
        # （仿真开始时机器人的初始状态）。
        bddl_path = paths["bddl_files"] / task.problem_folder / task.bddl_file
        state_path = paths["init_states"] / task.problem_folder / task.init_states_file
        print(f"  [{task_id:02d}] {task_name}: {bddl_path}")
        if not bddl_path.is_file():
            missing_bddl.append(bddl_path)
        if not state_path.is_file():
            missing_states.append(state_path)

    for label, missing in (("BDDL", missing_bddl), ("initial-state", missing_states)):
        if missing:
            print(colored(f"[warning] Missing {len(missing)} {label} files", "yellow"))
        else:
            print(colored(f"[ok] All {label} files exist", "green"))
    return benchmark_instance


def validate_task_id(benchmark_instance, task_id: int):
    """检查用户指定的任务编号是否在当前 benchmark 范围内。"""
    num_tasks = benchmark_instance.get_num_tasks()
    if not 0 <= task_id < num_tasks:
        raise ValueError(f"task id must be between 0 and {num_tasks - 1}, got {task_id}")


def render_initial_states(
    benchmark_instance,
    task_id: int,
    paths,
    output: Path,
    num_states: Optional[int] = None,
    gpu_device_id: int = -1,
    settle_steps: int = 5,
):
    """把一个任务的所有初始状态各运行几步，并拼成一张图片。"""
    # 延迟导入这些依赖：只做路径检查时不需要启动 MuJoCo / PyTorch。
    import torch
    import torchvision
    from PIL import Image

    from libero.libero.envs import OffScreenRenderEnv

    validate_task_id(benchmark_instance, task_id)
    task = benchmark_instance.get_task(task_id)
    # BDDL 决定场景中有哪些物体、初始区域以及任务目标。
    bddl_path = paths["bddl_files"] / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=128,
        camera_widths=128,
        # -1 表示 EGL 自动选择第一个可见 GPU；也可以显式传 0、1 等编号。
        render_gpu_device_id=gpu_device_id,
    )
    images = []
    try:
        init_states = benchmark_instance.get_task_init_states(task_id)
        env.seed(0)
        env.reset()
        if num_states is not None:
            if num_states < 1:
                raise ValueError("--num-init-states 必须是正整数")
            init_states = init_states[:num_states]
        if settle_steps < 0:
            raise ValueError("--settle-steps 不能是负数")
        total_states = len(init_states)
        for state_index, initial_state in enumerate(init_states, start=1):
            # 将该条初始状态放入环境，然后执行若干个零动作，让画面稳定下来。
            print(f"渲染初始状态 {state_index}/{total_states}...", flush=True)
            observation = env.set_init_state(initial_state)
            for _ in range(settle_steps):
                observation, _, _, _ = env.step([0.0] * 7)
            images.append(
                torch.from_numpy(observation["agentview_image"]).permute(2, 0, 1)
            )
    finally:
        env.close()

    # LIBERO 返回的相机图像上下方向与普通图片相反，因此保存前做翻转。
    grid = torchvision.utils.make_grid(images, nrow=10, padding=2).permute(1, 2, 0)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid.numpy()[::-1]).save(output)
    print(f"Saved initial-state grid to {output.resolve()}")


def download_datasets(dataset_name: str, paths, use_huggingface: bool = False):
    """下载指定数据集到 LIBERO 配置中的 datasets 目录。

    默认使用项目原有的 Box 链接；``use_huggingface=True`` 时改用 Hugging
    Face 镜像，通常在服务器/容器环境下更稳定。
    """
    import libero.libero.utils.download_utils as download_utils

    download_dir = str(paths["datasets"])
    if download_utils.check_libero_dataset(download_dir=download_dir):
        print(f"Dataset directory is already populated: {download_dir}")
        return
    download_utils.libero_dataset_download(
        download_dir=download_dir,
        datasets=dataset_name,
        use_huggingface=use_huggingface,
    )


def demonstration_paths(benchmark_instance, dataset_dir: Path):
    """根据 benchmark 中的任务定义，生成对应 HDF5 demonstration 路径。"""
    return [
        dataset_dir / benchmark_instance.get_task_demonstration(task_id)
        for task_id in range(benchmark_instance.get_num_tasks())
    ]


def inspect_demo(demo_path: Path, video_output: Optional[Path]):
    """打印 HDF5 demonstration 信息，可选地导出第一条轨迹视频。"""
    import h5py

    from libero.libero.utils.dataset_utils import get_dataset_info

    if not demo_path.is_file():
        raise FileNotFoundError(f"Demonstration file not found: {demo_path}")
    get_dataset_info(str(demo_path))

    if video_output is None:
        return
    import imageio.v2 as imageio

    # demonstration 是 HDF5 文件；这里读取 demo_0 的 agentview 相机序列。
    with h5py.File(demo_path, "r") as dataset:
        images = dataset["data/demo_1/obs/agentview_rgb"][()]
    video_output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(video_output, fps=60) as writer:
        for image in images:
            writer.append_data(image[::-1])
    print(f"Saved demonstration video to {video_output.resolve()}")


def parse_args():
    """定义命令行参数。参数不传时只执行基础检查。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="libero_10")
    parser.add_argument("--task-id", type=int, default=9)
    parser.add_argument(
        "--render-init",
        type=Path,
        metavar="PNG",
        help="将 --task-id 的所有初始状态渲染并保存为 PNG。",
    )
    parser.add_argument(
        "--num-init-states",
        type=int,
        metavar="N",
        help="只渲染前 N 条初始状态；不指定时渲染全部（通常是 50 条）。",
    )
    parser.add_argument(
        "--render-gpu-device-id",
        type=int,
        default=-1,
        help="渲染使用的 GPU 编号；-1 表示自动选择第一个可见 GPU。",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=5,
        metavar="N",
        help="每条初始状态渲染前执行 N 个零动作；默认 5，设为 0 可加快测试。",
    )
    parser.add_argument(
        "--download",
        choices=("all", "libero_goal", "libero_spatial", "libero_object", "libero_100"),
        help="如果数据集不存在，则下载指定的数据集集合。",
    )
    parser.add_argument(
        "--use-huggingface",
        action="store_true",
        help="下载时使用 Hugging Face 镜像，而不是旧的 Box 链接。",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        help="覆盖配置中的 datasets 目录；适合把数据下载到当前项目。",
    )
    parser.add_argument(
        "--inspect-demo",
        action="store_true",
        help="打印 --task-id 对应 demonstration 的 HDF5 元数据。",
    )
    parser.add_argument(
        "--video-output",
        type=Path,
        help="配合 --inspect-demo，将 demo_0 导出为 MP4。",
    )
    return parser.parse_args()


def main():
    # 1. 读取配置并打印路径；2. 检查 benchmark 文件；
    # 3. 按需下载数据、渲染初始状态或查看 demonstration。
    args = parse_args()
    paths = get_paths()
    # 不修改 ~/.libero/config.yaml，仅在本次运行中临时覆盖数据集目录。
    if args.datasets_dir is not None:
        paths["datasets"] = args.datasets_dir.expanduser().resolve()
    print_paths(paths)
    print("\nAvailable benchmarks:", sorted(benchmark.get_benchmark_dict()))
    benchmark_instance = inspect_benchmark(args.benchmark, paths)

    if args.download:
        download_datasets(args.download, paths, args.use_huggingface)
    if args.render_init:
        render_initial_states(
            benchmark_instance,
            args.task_id,
            paths,
            args.render_init,
            args.num_init_states,
            args.render_gpu_device_id,
            args.settle_steps,
        )
    if args.inspect_demo:
        validate_task_id(benchmark_instance, args.task_id)
        demos = demonstration_paths(benchmark_instance, paths["datasets"])
        inspect_demo(demos[args.task_id], args.video_output)


if __name__ == "__main__":
    main()
