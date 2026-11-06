# idea1 v3.1：准备状态

2026-10-20准备快照。**本文件保留历史准备状态，不是当前运行状态。** idea2已先完成；10-21正式进入v3.1，当前进度见[执行状态](idea1_v3_1_execution_status.md)。下面原manifest之后已有不可覆盖v2，正式运行使用v2。

## 已准备

- 独立代码目录：`preparations/idea1_v3_1/`；按用户要求由gpt-6.1-sol medium实现，未改用户 `idea1.md` 或现有模型源码。
- 9项CPU测试通过，包括mask/采样边界、promotion规则、真实bank选择及27层vision原生scan／完整保留selected-scan的严格数值检查；不是已完成真实模型GPU验收。
- 冻结初始manifest：`results/idea1_v3_1/manifest.json`，SHA身份字段 `759fdea316c8125254ec3f1bab94ba1b9dd7a7675bd5f973884ea53526facd16`。源码哈希受检查，不能运行中随意编辑或覆盖manifest。
- 初始筛选：π0 vision 21条件×40、π0.5 vision 21×40、XR0 VLM 27×40、CogACT VLM 22×40，共3640个screen键。另有2个benchmark-only完整vision selected-scan参考，不混入成功率分母。
- 扩展前沿按执行方案条件触发，另立不可覆盖manifest，不偷偷扩写初始矩阵。
- 完整模型均保留原AE结构；XR0 D2/H10/R10；CogACT原生DDIM D10/H16及原套件执行长度；π0／π0.5 D10、原H50／H10、R5。
- 输入bank：`results/idea1_v3_1/input_bank.pkl`，SHA256 `2c2e90a3e1fcac3f02226a433c6db9e19a0f3c169deea77707a9a1fbe8996d55`。100个全局唯一真实观测，四套件各25个，从idea2 Full的16个完整场景按全轨迹分位点采样；原NPZ／推理记录SHA、input digest和种子已核对。来源详见同名 `.receipt.json`。
- lean评测不写非必要图像／NPZ，但策略相机仍渲染；保留真实XYZ、动作、可用接触／抓取证据与启发式阶段。两worker共享串行模型服务，不冒充batch推理。
- 已加入端口、manifest、condition、服务PID一致性检查；只停止自有客户端和模型进程。技术失败保留并有界重试，不记策略失败。

## 正式执行前还要做

1. idea2全部数据、补充微基准、报告与完成审计结束。
2. 检查空闲卡，给不同GPU分配独立端口；只对等待空闲的目标检查，不频繁扫描自有运行卡。
3. 先E2小批量真实checkpoint验收与吞吐对照：完整I/O、lean I/O、环境复用、两worker。以相同8场景的实际墙钟和输出／初态一致性选择配置。
4. π系vision原生Full与完整selected-scan Full做同真实输入GPU数值验收，并保存匹配的parity artifact；未通过不跑mask成功率。
5. E3高优先级vision筛选与E1两模型跳层筛选分卡运行，所有mask直接做套件评测。候选按冻结规则补量，不用离线相似度先淘汰大部分闭环候选。
6. 所有质量／性能数据仍须真实完成后再汇总。当前代码与CPU通过不等于模型可运行、已有速度收益或实验已完成。

命令与入口见 `preparations/idea1_v3_1/README.md`；总体边界见[执行方案](idea1_v3_1_experiment_plan.md)。
