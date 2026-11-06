import argparse
import os
import subprocess
import tempfile
from contextlib import suppress
from typing import Any, Dict, List, Mapping

import cv2
import numpy as np
from numpy.typing import NDArray

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from test_policy import TestPolicy


UInt8Image = NDArray[np.uint8]
FloatArray = NDArray[np.float32]
Observation = Mapping[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--suite",
        type=str,
        default="libero_spatial",
        choices=[
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10","libero_90"],
        help="例如: libero_spatial / libero_object / libero_goal / libero_10",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--init-state-id",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./policies/my_policy/libero_test_policy.mp4",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
    )

    return parser.parse_args()


def get_wrist_camera_frame(obs: Observation) -> UInt8Image:
    """
    LIBERO 的 wrist_camera_image 来自 MuJoCo offscreen renderer。

    原始 observation 中的图像通常需要上下翻转后才是正常方向。
    返回 RGB uint8。
    """
    frame = np.asarray(obs["robot0_eye_in_hand_image"])

    frame = np.flipud(frame)
    frame = np.ascontiguousarray(frame)

    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    return frame

def get_video_frame(obs: Observation) -> UInt8Image:
    # print(obs)

    """
    LIBERO 的 agentview_image 来自 MuJoCo offscreen renderer。

    原始 observation 中的图像通常需要上下翻转后才是正常方向。
    返回 RGB uint8。
    """
    frame = np.asarray(obs["agentview_image"])

    frame = np.flipud(frame)
    frame = np.ascontiguousarray(frame)

    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    return frame


def write_video(
    frames: List[UInt8Image],
    output_path: str,
    fps: int,
) -> None:
    if len(frames) == 0:
        raise RuntimeError("没有收集到任何视频帧")

    os.makedirs(
        os.path.dirname(os.path.abspath(output_path)),
        exist_ok=True,
    )

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
        "18",
        output_path,
    ]

    # ffmpeg 的日志先收进临时文件：正常跑完不打扰终端，失败时再连同
    # return code 一起报出来。
    #
    # 这里用临时文件而不是 stderr=subprocess.PIPE，是为了避免死锁：
    # 管道有缓冲区上限，写满之后 ffmpeg 会卡在 write 上，而父进程此刻
    # 正在 process.stdin.write() 等它读，两边互等。临时文件由内核缓冲，
    # 不会写满阻塞，所以单线程就够，不用再开线程去排空管道。
    with tempfile.TemporaryFile() as log_file:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=log_file,
        )

        stdin = process.stdin

        if stdin is None:
            raise RuntimeError("无法打开 ffmpeg stdin")

        # ffmpeg 提前退出（参数不合法、编码器报错等）会关掉 stdin 的读端，
        # 这时再往里写就会抛 BrokenPipeError。不能让它直接冒出去 —— 那样就
        # 看不到下面那段 ffmpeg 日志了，而那才是真正的失败原因。
        with suppress(BrokenPipeError):
            for frame in frames:
                stdin.write(frame.tobytes())

        with suppress(BrokenPipeError):
            stdin.close()

        return_code = process.wait()

        if return_code != 0:
            log_file.seek(0)
            log = log_file.read().decode("utf-8", errors="replace").strip()

            raise RuntimeError(
                f"ffmpeg 写视频失败，return code = {return_code}\n"
                f"{log or '(ffmpeg 未输出任何日志)'}"
            )


def cv2_write_video(
    frames: List[UInt8Image],
    output_path: str,
    fps: int,
) -> None:
    if len(frames) == 0:
        raise RuntimeError("没有收集到任何视频帧")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    height, width = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            "OpenCV 无法创建 MP4。"
            "请检查当前环境中的 ffmpeg / OpenCV 编码支持。"
        )

    for frame_rgb in frames:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()


def main() -> None:
    args = parse_args()

    benchmark_dict: Dict[str, Any] = benchmark.get_benchmark_dict()

    if args.suite not in benchmark_dict:
        raise ValueError(
            f"未知 suite: {args.suite}\n"
            f"可选项: {list(benchmark_dict.keys())}"
        )

    task_suite = benchmark_dict[args.suite]()

    task = task_suite.get_task(args.task_id)

    bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )

    print("=" * 80)
    # print("33m[INFO]")
    print(f"suite       : {args.suite}")
    print(f"task id     : {args.task_id}")
    print(f"task name   : {task.name}")
    print(f"\33[33m instruction : {task.language}\33[0m")
    print(f"bddl file   : {bddl_file}")
    # print(" 33[0m ")
    print("=" * 80)
    

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=args.height,
        camera_widths=args.width,
    )

    env.seed(0)

    obs: Observation = env.reset()

    init_states = task_suite.get_task_init_states(args.task_id)

    if not 0 <= args.init_state_id < len(init_states):
        env.close()
        raise ValueError(
            f"init-state-id={args.init_state_id} 越界，"
            f"该任务共有 {len(init_states)} 个 initial states"
        )

    obs = env.set_init_state(init_states[args.init_state_id])

    frames: List[UInt8Image] = []
    frames_hand: List[UInt8Image] = []

    # 记录初始画面
    frames.append(get_video_frame(obs))
    # 记录腕部相机画面
    frames_hand.append(get_wrist_camera_frame(obs))

    # 让物理系统先稳定几步，不移动机械臂
    zero_action: FloatArray = np.zeros(7, dtype=np.float32)

    for _ in range(5):
        obs, reward, done, info = env.step(zero_action)
        frames.append(get_video_frame(obs))
        frames_hand.append(get_wrist_camera_frame(obs))
    policy = TestPolicy(num_steps=20)
    policy.reset()

    print("\n开始执行 policy:")
    print("连续 10 步，每一步 action = [1, 0, 0, 0, 0, 0, 0]")
    print("即默认 OSC_POSE 下，每一步要求末端沿 +x 移动约 5 cm。\n")

    for step in range(20):
        action: FloatArray = policy.get_action(obs)

        obs, reward, done, info = env.step(action)
        frames.append(get_video_frame(obs))
        frames_hand.append(get_wrist_camera_frame(obs))
        eef_pos = obs.get("robot0_eef_pos", None)

        if eef_pos is None:
            print(
                f"step {step + 1:02d} | "
                f"action={action.tolist()} | "
                f"reward={reward} | done={done}"
            )
        else:
            print(
                f"step {step + 1:02d} | "
                f"eef_pos={np.round(eef_pos, 4)} | "
                f"reward={reward} | done={done}"
            )

    # 再静止几帧，方便视频最后看清机器人停在哪里
    for _ in range(5):
        obs, reward, done, info = env.step(zero_action)
        frames.append(get_video_frame(obs))
        frames_hand.append(get_wrist_camera_frame(obs))

    env.close()

    write_video(
        frames=frames,
        output_path=args.output,
        fps=args.fps,
    )
    write_video(
        frames=frames_hand,
        output_path=args.output.replace(".mp4", "_hand.mp4"),
        fps=args.fps,
    )

    print("\n完成。")
    print(f"MP4 已保存到: {os.path.abspath(args.output)}")
    print(f"视频帧数: {len(frames)}")


if __name__ == "__main__":
    main()
