import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
from contextlib import suppress
from typing import Any, List, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALVIN_ROOT = PROJECT_ROOT / "calvin"

# 支持直接从 benchmark 根目录运行，无需预先安装本地 calvin_env 包。
sys.path.insert(0, str(CALVIN_ROOT / "calvin_env"))

from calvin_env.envs.play_table_env import get_env
from test_policy import TestPolicy


UInt8Image = NDArray[np.uint8]
FloatArray = NDArray[np.float32]
Observation = Mapping[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在 CALVIN 环境中运行 TestPolicy，并保存静态/手腕相机视频。"
    )
    annotation_selector = parser.add_mutually_exclusive_group()
    annotation_selector.add_argument(
        "--annotation-id",
        type=int,
        default=0,
        help="当前 split 的语言标注编号。",
    )
    annotation_selector.add_argument(
        "--task",
        help="选择当前 split 中该任务的第一条语言标注。",
    )
    parser.add_argument(
        "--list-annotations",
        action="store_true",
        help="列出当前 split 中的语言标注后退出。",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=CALVIN_ROOT / "dataset" / "calvin_debug_dataset",
        help="CALVIN 数据集根目录，或直接指向包含 .hydra 的 split 目录。",
    )
    parser.add_argument(
        "--split",
        choices=("training", "validation"),
        default="validation",
        help="dataset-path 指向数据集根目录时使用的 split。",
    )
    parser.add_argument("--steps", type=int, default=20, help="执行 policy 的步数。")
    parser.add_argument(
        "--settle-steps", type=int, default=5, help="policy 前后保持静止的步数。"
    )
    parser.add_argument(
        "--gripper-action",
        type=int,
        choices=(-1, 1),
        default=1,
        help="CALVIN 要求夹爪动作为 -1（关闭）或 1（打开）。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "policies" / "my_policy" / "calvin_test_policy.mp4",
        help="静态相机视频路径；手腕相机会保存为同名的 *_hand.mp4。",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--resolution",
        type=int,
        help="静态和腕部相机都使用此正方形分辨率。",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=14,
        help="H.264 画质参数，范围 0-51，数值越小画质越高、文件越大。",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_language_annotations(split_path: Path) -> Mapping[str, Any]:
    annotation_path = split_path / "lang_annotations" / "auto_lang_ann.npy"
    if not annotation_path.is_file():
        annotation_path = split_path / "auto_lang_ann.npy"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"找不到 CALVIN language annotations: {annotation_path}")
    return np.load(annotation_path, allow_pickle=True).item()


def select_annotation(
    annotations: Mapping[str, Any], annotation_id: int, task: Optional[str]
) -> Tuple[int, str, str, int, int]:
    language = annotations["language"]
    tasks = np.asarray(language["task"])
    instructions = np.asarray(language["ann"])
    frame_ranges = np.asarray(annotations["info"]["indx"])

    if task is not None:
        matches = np.flatnonzero(tasks == task)
        if not len(matches):
            available = sorted(set(tasks.tolist()))
            raise ValueError(
                f"当前 split 不包含任务 {task!r}，可用任务: {available}"
            )
        annotation_id = int(matches[0])

    if not 0 <= annotation_id < len(instructions):
        raise ValueError(
            f"annotation-id={annotation_id} 越界，当前 split 共有 {len(instructions)} 条标注"
        )

    start_frame, end_frame = frame_ranges[annotation_id]
    return (
        annotation_id,
        str(tasks[annotation_id]),
        str(instructions[annotation_id]),
        int(start_frame),
        int(end_frame),
    )


def print_language_annotations(annotations: Mapping[str, Any]) -> None:
    language = annotations["language"]
    frame_ranges = np.asarray(annotations["info"]["indx"])
    for annotation_id, (task, instruction, frame_range) in enumerate(
        zip(language["task"], language["ann"], frame_ranges)
    ):
        start_frame, end_frame = frame_range
        print(
            f"{annotation_id:2d} | {str(task):24s} | "
            f"frames {int(start_frame)}-{int(end_frame)} | {instruction}"
        )


def resolve_split_path(dataset_path: Path, split: str) -> Path:
    dataset_path = dataset_path.expanduser().resolve()
    split_path = dataset_path if (dataset_path / ".hydra").is_dir() else dataset_path / split
    config_path = split_path / ".hydra" / "merged_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"找不到 CALVIN 环境配置: {config_path}\n"
            "请让 --dataset-path 指向数据集根目录或 training/validation 目录。"
        )
    return split_path


def get_camera_frame(obs: Observation, camera: str) -> UInt8Image:
    try:
        frame = np.asarray(obs["rgb_obs"][camera])
    except KeyError as exc:
        available = list(obs.get("rgb_obs", {}).keys())
        raise KeyError(f"找不到相机 {camera!r}，可用相机: {available}") from exc

    if frame.dtype != np.uint8:
        if frame.size and frame.max() <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def write_video(
    frames: List[UInt8Image], output_path: Path, fps: int, crf: int
) -> None:
    if not frames:
        raise RuntimeError("没有收集到任何视频帧")

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        str(output_path),
    ]

    with tempfile.TemporaryFile() as log_file:
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log_file)
        except FileNotFoundError as exc:
            raise RuntimeError("找不到 ffmpeg，请先安装 ffmpeg。") from exc

        if process.stdin is None:
            process.kill()
            raise RuntimeError("无法打开 ffmpeg stdin")

        with suppress(BrokenPipeError):
            for frame in frames:
                if frame.shape != frames[0].shape:
                    raise ValueError("同一个视频中的帧尺寸不一致")
                process.stdin.write(frame.tobytes())
        with suppress(BrokenPipeError):
            process.stdin.close()

        return_code = process.wait()
        if return_code != 0:
            log_file.seek(0)
            log = log_file.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"ffmpeg 写视频失败，return code = {return_code}\n"
                f"{log or '(ffmpeg 未输出任何日志)'}"
            )


def hand_video_path(output_path: Path) -> Path:
    suffix = output_path.suffix or ".mp4"
    return output_path.with_name(f"{output_path.stem}_hand{suffix}")


def main() -> None:
    args = parse_args()
    if args.steps < 0 or args.settle_steps < 0:
        raise ValueError("--steps 和 --settle-steps 不能为负数")
    if args.fps <= 0:
        raise ValueError("--fps 必须大于 0")
    if args.resolution is not None and args.resolution <= 0:
        raise ValueError("--resolution 必须大于 0")
    if not 0 <= args.crf <= 51:
        raise ValueError("--crf 必须在 0 到 51 之间")

    split_path = resolve_split_path(args.dataset_path, args.split)
    annotations = load_language_annotations(split_path)
    if args.list_annotations:
        print_language_annotations(annotations)
        return
    annotation_id, task, instruction, start_frame, end_frame = select_annotation(
        annotations, args.annotation_id, args.task
    )
    initial_state_path = split_path / f"episode_{start_frame:07d}.npz"
    if not initial_state_path.is_file():
        raise FileNotFoundError(f"找不到标注起始帧: {initial_state_path}")
    initial_state = np.load(initial_state_path)
    observation_space = {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": [],
    }

    print("=" * 80)
    print(f"dataset      : {split_path}")
    print(f"annotation id: {annotation_id}")
    print(f"task         : {task}")
    print(f"\33[33m instruction  : {instruction}\33[0m")
    print(f"episode range: {start_frame}-{end_frame}")
    print(f"steps        : {args.steps}")
    if args.resolution is not None:
        print(f"resolution   : {args.resolution}x{args.resolution} (both cameras)")
    else:
        print("resolution   : static 200x200, gripper 84x84 (default)")
    print(f"H.264 CRF    : {args.crf}")
    print(f"gripper      : {args.gripper_action} ({'open' if args.gripper_action == 1 else 'close'})")
    print(f"output       : {args.output.expanduser().resolve()}")
    print("=" * 80)

    env = get_env(split_path, obs_space=observation_space, show_gui=False)
    try:
        # CALVIN 默认相机尺寸是静态 200x200、腕部 84x84；直接提高仿真渲染
        # 分辨率，避免只在写视频时放大低分辨率图像。
        if args.resolution is not None:
            for camera in env.cameras:
                if camera.name in ("static", "gripper"):
                    camera.width = args.resolution
                    camera.height = args.resolution

        env.seed(args.seed)
        obs: Observation = env.reset(
            robot_obs=initial_state["robot_obs"],
            scene_obs=initial_state["scene_obs"],
        )
        static_frames = [get_camera_frame(obs, "rgb_static")]
        hand_frames = [get_camera_frame(obs, "rgb_gripper")]

        zero_action: FloatArray = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, args.gripper_action],
            dtype=np.float32,
        )

        for _ in range(args.settle_steps):
            obs, _, _, _ = env.step(zero_action)
            static_frames.append(get_camera_frame(obs, "rgb_static"))
            hand_frames.append(get_camera_frame(obs, "rgb_gripper"))

        policy = TestPolicy(
            num_steps=args.steps,
            gripper_action=float(args.gripper_action),
        )
        policy.reset()

        print("\n开始执行 policy：末端执行器沿 +x 方向移动。\n")
        for step in range(args.steps):
            action = policy.get_action(obs)
            obs, reward, done, _ = env.step(action)
            static_frames.append(get_camera_frame(obs, "rgb_static"))
            hand_frames.append(get_camera_frame(obs, "rgb_gripper"))
            tcp_pos = np.asarray(obs["robot_obs"][:3])
            print(
                f"step {step + 1:02d} | tcp_pos={np.round(tcp_pos, 4)} | "
                f"reward={reward} | done={done}"
            )

        for _ in range(args.settle_steps):
            obs, _, _, _ = env.step(zero_action)
            static_frames.append(get_camera_frame(obs, "rgb_static"))
            hand_frames.append(get_camera_frame(obs, "rgb_gripper"))
    finally:
        env.close()
        # PlayTableSimEnv.__del__ 还会再次调用 close；标记为非 owner，避免
        # PyBullet 已断开后重复 disconnect 抛出异常。
        env.ownsPhysicsClient = False

    write_video(static_frames, args.output, args.fps, args.crf)
    hand_output = hand_video_path(args.output)
    write_video(hand_frames, hand_output, args.fps, args.crf)

    print("\n完成。")
    print(f"静态相机: {args.output.expanduser().resolve()}")
    print(f"手腕相机: {hand_output.expanduser().resolve()}")
    print(f"视频帧数: {len(static_frames)}")


if __name__ == "__main__":
    main()
