# Agent 协作入口

这是具身模型实验工作区的轻量源码与结果仓库，各版本的设计与最终分析均保留；v1/v2 中间结果与机器审计不提交。轻量运行汇总以 v3+ 为主，重复源码快照不提交。先读 `readme.md`、`CONTRIBUTING.md`、`docs/code/README.md` 和 `docs/setup/remote-reproduction.md`，再读目标实验的方案与报告。

- 各 `policies/` 与 benchmark 源码已展开到根仓库，包含首次导出时的本地修改；上游来源在 `docs/repository/upstreams.json`。子目录的 AGENTS.md 继续适用于对应实现。
- 每个 agent 在自己的分支或 worktree 中工作，明确负责的模块，保留他人的修改。实验输出使用新的 run ID；不要重写历史证据或冻结输入。
- 对评测规划、吞吐和多 GPU 实验，按需要使用 `.agents/skills/embodied-eval-acceleration/SKILL.md`。
- 模型、数据、视频、激活、完整 trace、环境与下载缓存不上 Git。使用 `shared/` 或外部存储；记录真实下载源或生成命令。
- 部分历史脚本包含原机器绝对路径。运行前检查入口、解释器、资产、端口和可用设备；历史 PID/lease/running 状态不表示当前任务。
- 提交前运行与改动相称的检查及 `python3 scripts/check_repo_content.py`。不要启动与当前任务无关的 GPU 实验、下载所有数据或变更共享环境。
