用于测试不同 benchmark 的模拟 policy。

CALVIN（默认使用仓库内的 debug validation split）：

```bash
calvin/.venv/bin/python policies/my_policy/eval_calvin_test_policy.py
```

静态和腕部相机默认都以 `512x512` 原生渲染，视频使用 H.264 CRF 14，
以改善原先 `200x200` / `84x84` 导致的模糊和分辨率不一致。可通过
`--resolution` 和 `--crf` 调整；分辨率越高渲染越慢、视频越大，CRF 越低画质越高。
脚本从当前 dataset split 的 `auto_lang_ann.npy` 读取任务、instruction 和对应 episode
范围，并用标注起始帧的机器人/场景状态 reset 环境。使用 `--list-annotations` 查看可用
标注，通过 `--annotation-id` 或 `--task` 选择；还可通过 `--dataset-path`、`--steps`、
`--gripper-action`、`--output` 调整数据集、回合长度、夹爪状态和输出视频路径。

SimplerEnv（默认运行 Google Robot 拿可乐罐任务）：

```bash
SimplerEnv/.venv/bin/python policies/my_policy/eval_simplerenv_test_policy.py
```

脚本在同一次 rollout 中录制 policy 观察相机和另一个可用相机；默认 Google Robot
会输出 `overhead_camera` 主视频和同名的 `*_base_camera.mp4`。可通过 `--task` 选择
其他预打包任务，通过 `--action-scale`、`--gripper-action`、`--camera`、`--steps`
和 `--output` 调整 rollout。使用 `--help` 可查看完整任务列表。
