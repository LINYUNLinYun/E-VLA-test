# 原生资产可用性与资格审计（不是运行结果）

本审计只使用公开一手仓库元数据、小文件及已批准的两个权重下载。没有训练、GPU推理或新环境实验。每个 Hub 子目录的 receipt.json 给出不可变提交、文件大小、SHA256、固定下载 URL；模型卡的性能数字不是本项目复现实验。

| 场景 | 已定位资产 | 资格与限制 |
|---|---|---|
| CALVIN ABC→D | `RLinf/RLinf-Pi0-CALVIN-ABC-D-SFT`，提交 `b8bc9ed031037dae5415279d870c55e9080c882b` | 官方 RLinf 文档推荐的原生 π0；目标域训练的正控，不是 LIBERO draft 的匹配消融。权重 7,011,543,960 字节，SHA256 `b9c80cdbf248344d09bcc054577828ba408a4ede049c61b03f68e894049e1c33`。 |
| CALVIN ABCD→D | 本次检查的 RLinf 发布及 Hub 搜索未定位原生 π0 ABCD 权重 | RLinf 的 ABCD→D 配置和权重示例是 π0.5；不能用它冒充 π0，也不能把 ABC 权重重新命名为 ABCD。两者共用 D 测试协议，训练来源必须分别证明。 |
| Simpler Bridge | `juexzz/INTACT-pi0-finetune-bridge`，提交 `c569ed57ba821efdb5caa522734920159c02ecb2` | LeRobot π0 BridgeV2 微调；模型卡明确用于 Simpler 四个 WidowX Bridge 任务。原生7D、单图、H4/D10；不是 Google Fractal 专家。未批准/未下载该大权重，尚需核实完整原生归一化与控制契约。模型卡披露按四任务平均表现选择检查点，不是无选择的任意检查点。 |
| Simpler Fractal/Bridge | `allenzren/open-pi-zero`，提交 `8518347d4ae0c6cfc69fbdda970b3f38c6ff76ca` | 作者明确说明是论文驱动的独立重实现，不等于 Physical Intelligence 官方 π0，不能不加标签地替代。仅获取元数据。 |
| PushT 原生2D | `qualia-robotics/pi0-pusht-30930d2b`，提交 `0b7c4aec769727469283937fd752c3d4d5acdbbc` | config 为 π0/gemma，状态和动作都2D，H50/D10；但保存的 processor 是 SmolVLM2 tokenizer/smolvla_new_line，与模型配置不一致。下载获准不等于兼容性通过；需权重/原生工厂审计，不得把候选直接宣称可运行正控。权重7,011,548,640字节，SHA256 `dd232f08d73ed7954f4e60a9eb8d387bc94e3b94508e9ba76369729da1b8cdb7`。 |
| FLASH 目标域 draft | 官方 `Dexmal/RealtimeVLA-Flash`，提交 `77b9a6f88fb100230bc78cb4cb361bd2e586f9fb` | 完整6文件清单只有 README/.gitattributes 与四个 `draft_libero_{10,goal,object,spatial}.pt`。该发布没有 CALVIN/Simpler/PushT draft；不能把目标域微调 Full 与无关 LIBERO draft 当成同权重匹配比较。 |

## CALVIN 原生配置证据

公开 RLinf checkout 固定于 `44418454146c1d89e428d3d1b2c1ac5928258238`，保存于 `policies/RLinf`。

- `rlinf/models/embodiment/openpi/dataconfig/__init__.py`：`pi0_calvin` 使用 `Pi0Config(action_horizon=5)`、`InternRobotics/InternData-Calvin_ABC`、`extra_delta_transform=False`。原生正控不能硬改为旧 LIBERO 的50步假定。
- `policies/calvin_policy.py`：位置3维、欧拉角3维、夹爪开口1维；原生双摄像头，无图像翻转；第三个虚拟相机被 mask。
- `toolkits/standalone_eval_scripts/openpi/calvin_eval.py`：实际取 `robot_obs[6:7]`，不是状态末尾的开闭符号；输出夹爪按大于0映射+1，否则−1。
- 原生加载器 `toolkits/standalone_eval_scripts/openpi/__init__.py` 使用 OpenPi PyTorch 加载器、数据变换及检查点 norm stats。其 `strict=False` 本身不是权重完全匹配证明，后续必须额外审计 missing/unexpected keys。
- 环境组的最小 D 资产审计见 `../native_preflight/calvin_audit/receipt.json`。它证明本地官方 checkout 的几何/相机一致性，不声称已下载并逐文件验证完整 D 验证归档。

## 可复现获取与安全边界

获取清单为 `native_pi0_acquisition_v1.json`，清单哈希 `ac712c10d8dc68b7be40f014232d3dcbd0afe43290b5d2956ab20be91f9f932d`，总计14,023,118,230字节/12文件。当前下载状态以 `native_pi0_download_v1.json` 为准，不能用本静态文本替代完成证明。

下载工具只允许显式批准的 repo，固定提交+字节数+SHA256，保留部分文件和身份文件，通过 Range 恢复；拒绝覆盖不一致的旧文件。元数据模式有字节上限且只取文本白名单，不会下载权重。8项 CPU 单元测试覆盖篡改、路径逃逸、上限、恢复偏移、错误哈希保留、授权与旧文件保护。

一手入口：[RLinf CALVIN](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/calvin.html)、[RLinf π0 ABC权重](https://huggingface.co/RLinf/RLinf-Pi0-CALVIN-ABC-D-SFT)、[Bridge π0模型卡](https://huggingface.co/juexzz/INTACT-pi0-finetune-bridge)、[独立 open-pi-zero](https://github.com/allenzren/open-pi-zero)、[PushT候选](https://huggingface.co/qualia-robotics/pi0-pusht-30930d2b)、[官方 FLASH发布](https://huggingface.co/Dexmal/RealtimeVLA-Flash)。

## 后续已核实的新增证据

- CALVIN 完整权重已经通过 SHA256；以下载 receipt 的逐文件记录为准。
- PushT 的部分文件仅取安全长度的 safetensors header，得到777个张量：`model.paligemma_with_expert.gemma_expert...`、状态投影[1024,32]、动作输入[1024,32]、动作输出[32,1024]。这支持权重确为π0结构而非SmolVLA；不能因错误processor命名丢弃它。完整下载SHA及原生工厂兼容性仍单独核验，见 `pusht_header_v1.json`。
- Bridge 正控的大权重后续已获批准，独立清单 `bridge_pi0_acquisition_v1.json`、状态 `bridge_pi0_download_v1.json`。INT-ACT官方源码固定 `66ae3ed4719c68cb2f3f22868015a7a171b04e1f`，literal `/policies/INT-ACT`；其LeRobot子模块固定 `35f6e02315dcb3fa25f3a740478265c6c793a95e`。Bridge归一化在外部适配器里完成，所以模型config中的IDENTITY不能解释为物理量无需归一化。
- RLinf检查点创建于2025-11-29；接近时间的v0.1源码 `9df6dc80dc729a6caccab92dd676004e51b1d3a2`同样使用H5和`extra_delta_transform=False`，但其openpi依赖未固定commit。本项目不能反推精确训练源码版本；必须额外审计权重键匹配。
- 新增 LIBERO-Plus 官方repo固定 `4976dc30028e805ff8094b55501d532c48fec182`，literal `/policies/LIBERO-plus`。官方评测资产zip6,395,849,578字节、数据集提交 `dd2bd61b7d9a6fef1abc52d606e983b41886a149`、SHA256 `96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf`，不下载训练数据、不替换旧安装或全局config。7类扰动、四suite共10,030任务；原始task映射及有效BDDL全部静态解析通过，见 `libero_plus_static_v2/task_mapping.csv`。121个goal任务难度为空，必须单列未知；官方每任务1次trial，不能误当每原始任务1次。

## 重要：CALVIN夹爪通道的源码与训练数据矛盾

上面“官方eval取robot_obs[6:7]”只描述官方代码，**不证明该宽度通道匹配训练语义**。新增一手数据审计：`InternRobotics/InternData-Calvin_ABC`固定版本 `fd844a43912c1823ff8355873b6f013a5cce127a`，meta/info.json报告`state.gripper`1维和10Hz；meta/stats.json值域±1。读取一个15,445字节的公开episode0 parquet并核验LFS SHA256 `dba51547f5edc8b2b021244b9541cacb19b98c178164dccf6460507ea71f247c`后，实际46帧的状态夹爪也只有±1，不是米制宽度；3个切换帧的state与当前action不同，全部45个可比较后续帧state与上一帧action相同。证据保存 `calvin_training_episode0_evidence_v1.json`，没有获取大训练数据。

这直接表明训练数据state.gripper与官方eval的width索引不一致；不能只靠归一化统计推断、也不能照抄官方eval后声称语义已验证。CALVIN官方数据文档将robot_obs末尾通道定义为gripper_action(close−1/open+1)，具有独立符号语义支持；精确转换源码仍未定位，单episode不能证明整个数据集所有采样的时序关系。后续实验必须保留官方width复制条件、训练一致符号条件的清晰标签/预注册，不能看结果后静默切换。

## 获取完成与原生PushT加载边界

三个独立获取receipt现均为complete：CALVIN+PushT合计14,023,118,230字节，Bridge6,482,733,212字节，Plus资产6,395,849,578字节，逐文件SHA通过。Plus官方ZIP含长相对wrapper，安全预检最初拒绝且未解包；随后显式记录/剥离同一固定wrapper，全部457,675成员逐文件CRC验证后原子安装新目录 `/policies/LIBERO-plus/libero/libero/assets`，解压8,953,192,654字节。见 `libero_plus_extraction_v2.json`；旧LIBERO安装和全局配置未改。

PushT新CPU适配准备在 `policy_pusht.py`：只接收真实原生96×96图像+像素XY状态，使用实际两维normstats，输出50×2、执行50、D10；没有七维投影或虚构Z。保存的两份normalizer文件SHA相同，state mean=[229.1110,293.3112]/std=[101.8547,96.4895]，action mean=[228.2397,293.9891]/std=[101.5996,96.0393]。兼容实现固定LeRobot v0.4.3 `0b067df57d21d3a02d6c511f1609172fa39ac29b`；这**不是已知训练保存版本**，发布config没有记录该版本。权重的777张量结构已核验，但真实模型构造后的全key/shape严格gate尚未执行，不能把7项CPU契约测试当成原生模型加载成功。

原生LeRobot PI0工厂源码明确要求PaliGemma tokenizer、结尾换行、右侧补齐48tokens。因此拒绝执行保存文件中错配的SmolVLM处理器有源码依据，但不能伪造HF配置替代受限资产。Google官方固定提交 `35e4f46485b4d07967e7e9935bc3786aad50687c` 的两个小配置文件，在标准huggingface_hub默认正常认证(token=None)下也返回GatedRepoError401，见 `paligemma_standard_client_v1.json`。未读取/打印凭证、未使用镜像。等待正常授权或用户给出的真实本地tokenizer目录；当前原生模型import还缺accelerate，未安装新依赖，也未GPU运行。
