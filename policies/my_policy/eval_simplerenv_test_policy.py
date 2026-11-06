import argparse
from contextlib import suppress
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
from numpy.typing import NDArray


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIMPLER_ROOT = PROJECT_ROOT / "SimplerEnv"

# Allow the script to find the local checkout even when launched from another cwd.
sys.path.insert(0, str(SIMPLER_ROOT))

import simpler_env
from simpler_env import ENVIRONMENTS

from test_policy import TestPolicy


UInt8Image = NDArray[np.uint8]
FloatArray = NDArray[np.float32]
Observation = Mapping[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在 SimplerEnv 中运行 TestPolicy，并保存观察相机视频。"
    )
    parser.add_argument(
        "--task",
        choices=ENVIRONMENTS,
        default="google_robot_pick_coke_can",
        help="SimplerEnv 预打包任务。",
    )
    parser.add_argument("--steps", type=int, default=20, help="执行 policy 的步数。")
    parser.add_argument(
        "--settle-steps", type=int, default=5, help="policy 前后保持静止的步数。"
    )
    parser.add_argument(
        "--action-scale",
        type=float,
        default=0.1,
        help="缩放 TestPolicy 输出的末端位姿动作，不影响夹爪动作。",
    )
    parser.add_argument(
        "--gripper-action",
        type=float,
        choices=(-1.0, 0.0, 1.0),
        help="夹爪动作；默认根据 Google Robot/WidowX 自动选择保持打开。",
    )
    parser.add_argument(
        "--camera",
        help="主视频相机；默认使用 policy 观察相机，另一个可用相机会单独保存。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "policies"
        / "my_policy"
        / "simplerenv_test_policy.mp4",
        help="主相机视频路径；第二个相机会保存为同名的 *_<camera>.mp4。",
    )
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument(
        "--crf",
        type=int,
        default=14,
        help="H.264 画质参数，范围 0-51，数值越小画质越高、文件越大。",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def default_camera_name(robot_uid: str) -> str:
    if "google_robot" in robot_uid:
        return "overhead_camera"
    if "widowx" in robot_uid:
        return "3rd_view_camera"
    raise ValueError(f"无法为 robot {robot_uid!r} 自动选择观察相机，请传入 --camera")


def default_open_gripper_action(robot_uid: str) -> float:
    # SimplerEnv 的夹爪符号取决于机器人 URDF。
    return -1.0 if "google_robot" in robot_uid else 1.0


def get_camera_frame(obs: Observation, camera: str) -> UInt8Image:
    try:
        frame = np.asarray(obs["image"][camera]["rgb"])
    except KeyError as exc:
        image_obs = obs.get("image", {})
        available = list(image_obs.keys()) if isinstance(image_obs, Mapping) else []
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


def secondary_video_path(output_path: Path, camera: str) -> Path:
    suffix = output_path.suffix or ".mp4"
    return output_path.with_name(f"{output_path.stem}_{camera}{suffix}")


def collect_camera_frames(
    obs: Observation, frames_by_camera: Mapping[str, List[UInt8Image]]
) -> None:
    for camera, frames in frames_by_camera.items():
        frames.append(get_camera_frame(obs, camera))


def step_env(
    env: Any,
    action: FloatArray,
    frames_by_camera: Mapping[str, List[UInt8Image]],
) -> tuple[Observation, float, bool, bool, Mapping[str, Any]]:
    clipped_action = np.clip(action, env.action_space.low, env.action_space.high)
    next_obs, reward, terminated, truncated, info = env.step(clipped_action)
    collect_camera_frames(next_obs, frames_by_camera)
    return next_obs, float(reward), bool(terminated), bool(truncated), info


def main() -> None:
    args = parse_args()
    if args.steps < 0 or args.settle_steps < 0:
        raise ValueError("--steps 和 --settle-steps 不能为负数")
    if args.action_scale < 0:
        raise ValueError("--action-scale 不能为负数")
    if args.fps <= 0:
        raise ValueError("--fps 必须大于 0")
    if not 0 <= args.crf <= 51:
        raise ValueError("--crf 必须在 0 到 51 之间")

    env = simpler_env.make(args.task)
    frames_by_camera: Dict[str, List[UInt8Image]] = {}
    last_info: Mapping[str, Any] = {}
    terminated = False
    truncated = False

    try:
        robot_uid = str(env.unwrapped.robot_uid)
        policy_camera = default_camera_name(robot_uid)
        primary_camera = args.camera or policy_camera
        gripper_action = (
            args.gripper_action
            if args.gripper_action is not None
            else default_open_gripper_action(robot_uid)
        )

        obs, reset_info = env.reset(seed=args.seed)
        instruction = env.unwrapped.get_language_instruction()
        available_cameras = list(obs["image"].keys())
        if primary_camera not in available_cameras:
            raise ValueError(
                f"找不到主相机 {primary_camera!r}，可用相机: {available_cameras}"
            )

        secondary_camera = next(
            (
                camera
                for camera in (policy_camera, "base_camera", *available_cameras)
                if camera != primary_camera and camera in available_cameras
            ),
            None,
        )
        recorded_cameras = [primary_camera]
        if secondary_camera is not None:
            recorded_cameras.append(secondary_camera)
        frames_by_camera = {camera: [] for camera in recorded_cameras}
        collect_camera_frames(obs, frames_by_camera)

        secondary_output = (
            secondary_video_path(args.output, secondary_camera)
            if secondary_camera is not None
            else None
        )

        print("=" * 80)
        print(f"task          : {args.task}")
        print(f"instruction   : {instruction}")
        print(f"robot         : {robot_uid}")
        for index, camera in enumerate(recorded_cameras):
            frame = frames_by_camera[camera][0]
            label = "primary camera" if index == 0 else "second camera"
            print(f"{label:14s}: {camera} {frame.shape[1]}x{frame.shape[0]}")
        print(f"steps         : {args.steps}")
        print(f"action scale  : {args.action_scale}")
        print(f"gripper       : {gripper_action}")
        print(f"seed          : {args.seed}")
        print(f"reset info    : {reset_info}")
        print(f"primary output: {args.output.expanduser().resolve()}")
        if secondary_output is not None:
            print(f"second output : {secondary_output.expanduser().resolve()}")
        print("=" * 80)

        zero_action = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_action],
            dtype=np.float32,
        )

        for _ in range(args.settle_steps):
            obs, _, terminated, truncated, last_info = step_env(
                env, zero_action, frames_by_camera
            )
            if terminated or truncated:
                break

        policy = TestPolicy(
            num_steps=args.steps,
            gripper_action=gripper_action,
        )
        policy.reset()

        print("\n开始执行 policy：输出机器人控制坐标系的 +x 动作。\n")
        for step in range(args.steps):
            if terminated or truncated:
                break

            action = policy.get_action(obs)
            action[:6] *= args.action_scale
            obs, reward, terminated, truncated, last_info = step_env(
                env, action, frames_by_camera
            )
            tcp_pos = np.asarray(obs["extra"]["tcp_pose"][:3])
            success = bool(last_info.get("success", terminated))
            print(
                f"step {step + 1:02d} | tcp_pos={np.round(tcp_pos, 4)} | "
                f"reward={reward:g} | success={success} | truncated={truncated}"
            )

        for _ in range(args.settle_steps):
            if terminated or truncated:
                break
            obs, _, terminated, truncated, last_info = step_env(
                env, zero_action, frames_by_camera
            )
    finally:
        env.close()

    write_video(frames_by_camera[primary_camera], args.output, args.fps, args.crf)
    if secondary_camera is not None and secondary_output is not None:
        write_video(
            frames_by_camera[secondary_camera], secondary_output, args.fps, args.crf
        )
    episode_stats: Optional[Mapping[str, Any]] = last_info.get("episode_stats")

    print("\n完成。")
    print(f"主相机视频: {args.output.expanduser().resolve()}")
    if secondary_camera is not None and secondary_output is not None:
        print(f"第二路视频: {secondary_output.expanduser().resolve()}")
    print(f"视频帧数  : {len(frames_by_camera[primary_camera])}")
    print(f"terminated : {terminated}")
    print(f"truncated  : {truncated}")
    if episode_stats is not None:
        print(f"episode stats: {dict(episode_stats)}")


if __name__ == "__main__":
    main()
