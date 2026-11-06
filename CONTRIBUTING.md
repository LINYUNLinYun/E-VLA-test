# 并行开发与实验记录

先阅读 [readme](readme.md)、[代码索引](docs/code/README.md)、[远程复现说明](docs/setup/remote-reproduction.md)和目标实验的方案与报告。实验协议中有关样本划分、噪声、成功率分母、计时边界和冻结配置的约束继续适用。

## 分工与分支

每位开发者或 agent 从 `main` 建立自己的分支，例如 `experiment/<实验名>` 或 `fix/<模块>`。可以使用独立 clone 或 `git worktree`。开始时约定负责的模块与文件，修改共享模型、评测器或调度器前确认其他实验是否依赖它。向本仓库发 PR，说明改动、验证和实验影响。

项目已经包含各上游源码快照与本地修改，直接在相应目录开发。根目录 Git 记录统一管理；`docs/repository/upstreams.json` 只记录首次导出的来源，不应把其 commit 误认为后续开发的当前状态。各上游的贡献说明适用于该项目的开发方式，本仓库的协作约定适用于这里的分支和 PR。

## 新实验与结果

各版本设计与最终分析报告均可保留。不要补入 v1/v2 的中间结果、机器审计或重复运行源码快照；轻量运行汇总以 v3+ 为主。

- 方案、状态和报告写入 `docs/experiments/<实验名>/`；小型独立实现放 `experiments/<实验名>/`，既有路线沿用 `preparations/<实验名>/`。
- 每次运行使用独立 run ID 和输出目录，记录 Git commit、本地 diff、输入来源、seed、模型与归一化版本、依赖、计时范围及完成状态。
- 保留失败、阴性和中断结果；工程冒烟通过、离线误差改善、真实闭环成功率与 GPU 加速分别报告。
- 上传报告、配置、manifest、汇总 CSV/JSON 和小图。数据集、模型、视频、激活张量、逐案例原始 receipt、完整逐请求日志与大型 trace 留在外部存储或实验机器，并在文档中记录真实来源或生成命令。
- 历史 receipt、冻结输入和已发表结果应视作证据。新实验另存，不覆盖原记录。

## 提交检查

```bash
git diff --check
git add <本次文件>
python3 scripts/check_repo_content.py
```

内容检查限制普通文件为 10 MiB、图/PDF 为 1 MiB，并拒绝常见大产物与疑似 token。它检查 Git 已跟踪或暂存的路径；提交前先暂存新增文件。它不代替对应实验的测试，也不能保证发现所有敏感信息。

不要为绕过子项目 `.gitignore` 而对整个工作区使用 `git add -f .`。确认文件是小型源码/文档且检查通过后，可针对该文件使用 `git add -f <路径>`。GitHub 的 Repository content 工作流会重复体积和内容检查。

`docs/repository/snapshot_files.tsv` 保存首次源码导出时的 hash，仅用于追溯。正常开发后不要求它与新代码一致；`--verify-snapshot` 是首次导出完整性检查，CI 不启用该选项。

源码与资产的许可证按各上游目录中的 LICENSE/NOTICE 执行；本仓库没有将所有上游重新授予同一个许可证。
