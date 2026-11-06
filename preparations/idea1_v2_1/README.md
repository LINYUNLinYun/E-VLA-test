# E1 v2.1 实验前准备

本目录仅做前期准备；本轮禁止正式实验。`e1_preflight.py` 没有模型推理、GPU 查询、服务器启动、环境 rollout、下载或安装入口。它不证明 E1 已可运行。

从工作区根目录执行（仅 Python 标准库）：

```bash
python3 preparations/idea1_v2_1/e1_preflight.py
python3 preparations/idea1_v2_1/e1_preflight.py --model xiaomi-robotic0
python3 -m unittest discover -s preparations/idea1_v2_1 -p 'test_*.py' -v
```

preflight 输出真实本地状态并以 **exit 2** 表示仍有阻塞，这是预期行为。默认只输出 stdout；metadata 默认读取本目录 `metadata/<model>/`，也可用 `--metadata-dir /absolute/local/path` 指定。只读取配置、权重索引及 Python 源码，不载入 tensor、不执行远程模型代码。索引层号只证明键存在，不能证明运行时实际执行深度。

`native_protocol.template.json` 提供待填的原生协议字段。复制为后续真实模型协议，填写各 suite 的 R 与证据后可通过 `--protocol-file /absolute/path/native_protocol.json` 校验；未知 R、R>H、必需字段缺失或来源 hash 不符均明确报错。模板所有未知值为 null，schema 完整本身也不授权实验或替代真实架构/输出 parity 验证。

显式生成可复核文件（目标父目录需已存在；目标文件不得存在）：

```bash
python3 preparations/idea1_v2_1/e1_preflight.py --model db-cogact --write-manifest /tmp/db-cogact-e1-preflight.json --write-plan /tmp/db-cogact-e1-plan.json
```

权重下载完成后，用独立工具显式读取全部大文件并校验（默认 preflight 不做这一步）：

```bash
python3 preparations/idea1_v2_1/verify_weights.py --checkpoint shared/checkpoints/xiaomi-robotic0/fc04bbdcc53e498695cfad95d1199091396ef55b --hub-file-info preparations/idea1_v2_1/metadata/xiaomi-robotic0/hub_file_info.json --index preparations/idea1_v2_1/metadata/xiaomi-robotic0/model.safetensors.index.json --revision fc04bbdcc53e498695cfad95d1199091396ef55b --write-manifest /tmp/xiaomi-v21-weights-verified.json
python3 preparations/idea1_v2_1/verify_weights.py --checkpoint shared/checkpoints/db-cogact/2cb20bcaa1e2e39d431b7e3d40a2d7a696388a10 --hub-file-info preparations/idea1_v2_1/metadata/db-cogact/hub_file_info.json --index preparations/idea1_v2_1/metadata/db-cogact/model.safetensors.index.json --revision 2cb20bcaa1e2e39d431b7e3d40a2d7a696388a10 --write-manifest /tmp/db-cogact-v21-weights-verified.json
```

`verify_weights.py` 只读，8MiB 分块计算 SHA256，按 `hub_file_info.json` 的 `siblings[].lfs.size/sha256` 检查所有 safetensors（含 index 引用的每个 shard），并用 index 的 Git blob SHA1 与同 revision 的 API 元数据绑定。checkpoint 目录名必须等于指定 revision，API `sha` 也必须一致；远程名称路径穿越、目录外 symlink、hash 过程中被修改、缺失/大小/hash 不符均失败。成功返回0，任一错误返回2；目标 manifest 不覆盖。此检查校验所提供 API 元数据与本地字节的一致性，不加载 tensor 或启动模型。

计划文件是不可调度的预登记槽位：两个主模型各 Full + 12 个待真实探针产生 IDs 的 mask，每条件 4 suite × task0–9 × init0–4 = 200，合计 2600；所有条件先 init0，再统一 init1–4。mask 明确为 null，不能把条件名称当作已冻结层号。额外 8 条 task0/5、init5 的 Full 校准和 4 条官方参考诊断单独列出，不混入正式分母。CogACT 的同规模计划仅为取得真实 LIBERO checkpoint 后的条件计划；GR00T 补充计划为 Full+4、1000正式，不能计作第二个深层模型。没有自动写入正式结果目录。

来源记录包括 Git revision/origin/dirty 状态、完整 tracked diff 的 hash、静态源码行号与文件 hash、依赖声明 hash、环境 prompt、已安装 distribution 版本和指定本地 checkpoint metadata。Git 的 safe.directory 仅按本次目标仓库显式配置，不改全局设置。上游未跟踪文件只记录路径状态，其内容以及未检查的权重均不能视为冻结 provenance。

环境检查通过项目 `.venv/bin/python -I -S` 启动标准库脚本，直接搜索该虚拟环境 site-packages，禁止处理可执行 `.pth`、导入模型包或初始化 CUDA。`module_discoverable` 只是模块静态可定位；它不能证明共享库/ABI/原生扩展可成功导入，也不能保证 editable `.pth` 包被识别。需要后续受控的 CPU import 检查，不能以本工具结果宣称环境就绪。

必需服务端依赖、可选服务端依赖与独立客户端依赖分别列出：DB的 `flash_attn` 为可选检查，其缺失不产生服务端必需依赖阻塞；Xiaomi 的 `tyro/libero/imageio` 记录于客户端字段，不要求与server同环境。`metadata/<model>/weights_verified.json` 如已存在，会连同来源/hash及原记录收入manifest；preflight不重新hash权重、不据receipt声称当前字节再次验证。归一化、运行时加载及真实parity仍独立阻塞。

`validate_mask` 提供纯逻辑层号/重复/保护层检查，以及与当前adapter一致的声明分类。Xiaomi 前3层保护 DeepStack；Xiaomi 与 DB-CogACT 对所有skip层（含早层）均声明保留 KV shell，`fully_eliminated_layers=[]`。这不是实测成本或算子消除证明；原CogACT/GR00T 的实际读出保护层及跳层契约仍须由架构证据显式提供。

`validate_candidates` 检查12（或GR00T的4）个候选的实际 IDs、去重、各预算数量、生成算法/seed/理由及真实探针来源 hash；结构通过不证明探针质量或模型 parity，不会生成候选来补齐空槽。

`adapters.py` 提供隔离、可恢复的核心准备实现，import 时不导入 torch；使用时须先由后续加载器取得已审核模型：

- `xiaomi_hidden_bypass(model, skip, reference=False)` 对固定 source hash 的36层/16DiT架构临时替换被选层 forward。它保守地对所有跳层（包括早层）保留 KV shell，保持 DynamicCache 原索引；该 context audit 与 `validate_mask` 的快速adapter契约一致，`fully_eliminated_layers=[]`。`reference=True` 执行原块后丢弃 hidden update，另记reference分类，供同输入同噪声比对。前三层、外层 DeepStack循环、最终norm与动作头保留。架构/源hash改变时拒绝。
- `db_cogact_hidden_bypass(model, skip, reference=False)` 是DB-CogACT专用实现，匹配已锁定Dexbotic源码和Transformers v4.57.6 Qwen2源码hash、28层与原始layer_idx。保留原 `use_cache=True`，对所有skip层保留KV shell；最终norm、末token cognition读出、视觉塔与动作扩散保持原调用。支持慢参考；真实checkpoint/action parity及实际DynamicCache/backend验证仍需后续完成。
- `hidden_update_bypass(layers, skip, contract, reference=False)` 是显式 class/source hash/返回协议 contract 的通用无cache实现；遇到cache/attention请求拒绝，不能替代上述DB专用adapter。
- `layer_probes(layers, token_masks, records)` 在显式排除padding的视觉/文本/读出token分组上记录逐层cosine、更新RMS、attn/FFN分支量级，只保留CPU标量。未实现真实轨迹阶段采样与单层动作影响探针调度。

以上 temporary forward override 仅用于eval模式，不支持训练、compile、并发模型调用或其它forward override。`test_adapters.py` 以CPU toy及固定源码提取算子检查shell/reference与调用消除，不能替代真实checkpoint/真实观测/实际服务端parity。

后续在已审核加载器/推理进程内使用，`model` 是对应已加载且 `.eval()` 的模型对象，`frozen_layer_ids` 必须来自真实校准与预登记，不在这里臆造：

```python
from preparations.idea1_v2_1.adapters import db_cogact_hidden_bypass

with db_cogact_hidden_bypass(model, frozen_layer_ids) as skip_audit:
    actions = model.inference_action(input_ids, image_tensor, native_inference_args)
```

使用相同固定输入、真实噪声状态与原生参数，依次比较原模型、空mask、`reference=True`、快速实现；将context返回的skip/KV/reference分类写入审计。此示例不是正式闭环runner。DB固定官方源码快照在 `metadata/db-cogact/modeling_qwen2_v4_57_6.py`，来源为适配器常量中的官方v4.57.6 URL，SHA256 `a59fa06524227361fb401baf4d177124a27aec146bc01ec931235a9abbab17cb`；只供离线AST toy提取与来源核查，不被生产入口动态import。默认测试不联网。

已静态核查的适配风险：

- Xiaomi 原客户端默认 50 init、10 worker、R10、按观测内容产生 seed、视频 fps10，原 launch 脚本激活固定 Conda；必须改用显式 init/task 队列、条件配对随机源、独立 invalid、20Hz 控制时间与完整遥测。checkpoint 配置 H30/文本36/DiT16 以及远程 forward 默认 D5 只是来源证据，须同实际服务端覆盖参数一起冻结。
- DB 模型配置 H16/文本28，发布推理路径 D10/CFG1.5。独立官方 `policies/dexbotic-benchmark/evaluation/configs/libero/` 的 suite YAML 给出 R：Spatial12/Object16/Goal16/Long15；`example_libero.yaml` 为11，π0样例为5。不能把任一个样例值广播到全部 suite，也需核对这些 suite YAML 是否与选用的 DB-CogACT checkpoint 匹配。
- CogACT 未验证可复现官方 LIBERO 权重，不能拿 OXE 或 DB-CogACT 替代。GR00T loader 删除超过 select_layer 的层，必须逐 suite 确认真实 checkpoint 和有效深度。

以上为准备阶段状态；2026-10-16 执行实现与真实证据见下节。preflight 的旧模板字段不替代实际服务端协议。

## 实际模型服务与独占性能测试

`runtime_models.NativePolicy` 使用原生图像/状态处理与动作解码，`runtime_server.py` 是仅监听127.0.0.1的可信本地RPC。帧为4字节big-endian长度加pickle dict；请求`op=predict`携带原始LIBERO `obs`、`instruction`、`seed`、1-based `skip`，可选`reference/probe/cache_path/category/native`。返回`actions[H,7]`、CUDA event分段及服务wall毫秒、输入/真实采样噪声hash、逐层特征和算子调用计数。`health/reset/parity`分别读协议、重置无状态策略、运行5次官方/empty/repeat/reference/fast诊断。每个预测起始和失败也写attempt账本；成功响应与账本携带加载epoch/hash。`cache_path`额外保存真实noise和decoded actions，可独立重算hash。

实测 Xiaomi checkpoint虽名义`action_length=30`，原生processor的LIBERO mask/归一化为`[1,10,32]`，官方forward实际生成H10；本轮冻结H10/R10/D5。DB冻结H16/D10/CFG1.5和suite R12/16/16/15，官方suite默认仅base相机、state不使用、gripper取sign。两模型真实checkpoint parity及q/o/FFN算子消除证据位于本轮`e1/<model>/readiness/parity.json`，不是toy单测结论。

启动示例（对应项目`.venv`，单卡绑定）：

```bash
CUDA_VISIBLE_DEVICES=1 HF_HOME="$PWD/shared/caches/huggingface" HF_HUB_OFFLINE=1 policies/xiaomi-robotic0/.venv/bin/python preparations/idea1_v2_1/runtime_server.py --model xiaomi-robotic0 --port 9101 --output-dir <attempt>/e1/xiaomi-robotic0/readiness
```

`runtime_benchmark.py` 在真实校准与12个mask冻结后运行；调度器先独占释放对应GPU/server，再传`--exclusive-note`说明租约。固定32个真实校准输入，13条件循环轮换，每输入/条件10次warmup与50次measurement，共24960逻辑预测/模型，category=benchmark独立记账。任何外来模型调用、epoch变化、noise不一致或未决调用均停止，不静默重试。输出`manifest.json`、`samples.jsonl`、分warmup/measurement的`latency_breakdown.csv`、`memory_snapshots.jsonl`、`recording_costs.jsonl`和完成审计；GPU event与server/client wall及记录开销分开解释。

```bash
policies/libero-eval-v21/.venv/bin/python preparations/idea1_v2_1/runtime_benchmark.py --model xiaomi-robotic0 --port 9101 --gpu 1 --probes <model>/calibration/probes.json --masks <model>/candidates.json --output-dir <model>/benchmark --exclusive-note <scheduler-lease-id>
```
