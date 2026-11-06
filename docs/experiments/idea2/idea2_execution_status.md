# idea2 执行进度

日期：2026-10-21。当前阶段：**idea2全部完成、最终审计通过**。下文保留历史执行过程，旧进度数字不是当前状态。

## 最终状态

- E0／E1／E2及E1b六臂完整；E3共112/112有效，缺失0、错误0。
- Full16/16；串行与异步复用间隔1／2／4均分别11/16、8/16、4/16，每个scene的成功标记一致。
- `artifacts/final_audit.json`：passed=true，graph_required=true；112视频全部完整解码并核对帧数／FPS、动作与XYZ、输入／输出hash、cache来源及reset身份。
- 24张固定XYZ图＋4张汇总图，112视频，184个可读索引链接全部目标存在。
- 客户端47748和E1b42259均exit0；模型服务31479接收到shutdown并exit0，相关PID均已退出。
- [最终报告](idea2_exploratory_report.md)、[可读媒体索引](artifacts/media_index.md)、[浏览器播放器](artifacts/media_index.html)。
- 接下来单独执行idea1 v3.1，不能把idea2完成写成两个目标均已完成。

## 已完成

- 已读用户 `idea2.md`、`idea2_探索性实验.md`，将模块级双 stream 与后续层级流水线明确分开。
- 原始 OpenPI commit：`215abfb217dbac7d5f1273282331b9b1866c0479`。模型仓库已有用户 vision compression 修改，本轮使用独立 PyTorch路径，不覆盖它们。
- 官方 JAX→PyTorch 转换已正常 exit0：源 `shared/checkpoints/openpi/openpi-assets/checkpoints/pi0_libero`，目标 `shared/checkpoints/openpi/pi0_libero_pytorch_idea2`；原 normalization assets 已复制到目标。
- 采用已有 `policies/openpi/.venv`：torch2.7.1+cu126，transformers4.53.2；按仓库说明安装其自带 transformers_replace，未升级依赖。
- 按用户先前要求，代码由 gpt-6.1-sol medium 实现，位于 `preparations/idea2/`，未编辑用户 idea 文档或模型源码。
- 8项CPU协议／采样单测通过：双槽背压、reader租约、ready发布、pending更新、reset世代隔离、重复release、原始10步时间递推及稳定输出。
- 首次Spatial真实观测E0通过：归一化动作、实际输出动作、velocity的max-abs误差均为0；原始路径确实调用10次denoise，全部KV保持不变。
- 转换映射审计通过：776个应映射训练张量无缺失、无形状错误；保存的777张量包含tied embedding别名与未用于动作推理的AE lm_head。该检查证明映射覆盖，不声称证明JAX与PyTorch逐值等价。原始证据：`results/idea2/run_20261020/converted_mapping_audit.json`。
- 四套件E0均通过，动作max-abs均为0。E2共96个请求配置（4套件×8真实观测×3刷新间隔），同版本串行／并发动作max-abs均为0；间隔1的离线设定消费当前KV，不等于闭环异步的年龄分布。
- E1各100次正式测量：串行动作就绪P50=222.90ms、两任务全部完成297.56ms；双stream分别290.24ms和290.24ms。总完成时间约1.025×，动作响应约0.768×（变慢）。kernel重叠仅0.0167ms／66.09ms kernel union≈0.025%；这是当前eager提交方式的结果，不是流水线方法的理论上限。
- 后续针对已观测CPU dispatch瓶颈，新增独立E1b CUDA Graph微基准代码。它不改变E3科学组。为缩短等待，23:30确认GPU2空闲后，准备在GPU2单独做同卡eager／graph、串行／并发四臂对照；不将GPU0与GPU2绝对耗时混合计算加速比。
- 另立 [idea1 v3.1执行方案](../idea1/idea1_v3_1_experiment_plan.md)，尚未启动v3.1，先做idea2。

## 历史运行过程（均已退出）

GPU0：E3闭环服务9012端口，输出 `results/idea2/run_20261020/episodes/`。当前先运行28个smoke键（4套件task0/init0×7条件），再以同目录断点续跑补齐112个唯一键，不重复已完成样本。

模型会话31479、PID2250491（微基准完成后保留同一模型进入服务）。smoke客户端53444已正常exit0，28个唯一键全部完成，无技术异常：fresh 4/4；串行／异步间隔1均3/4、间隔2均2/4、间隔4均1/4。此小样本不足以形成总体成功率结论。

2026-10-20 23:19已启动补量客户端会话47748、PID2264737，复用28个完成键，补齐其余84个到112个总键；不重启模型、不重复已完成样本。输出仍为同一episodes目录。23:32左右已完成57/112个有效episode；Full已16/16、串行间隔1已11/16，其余尚未齐，不能混用当前不同分母作最终排名。

E1b首次会话26979、PID2288521已明确exit1：原始sampler在capture中创建CUDA scalar常量，触发不允许的H2D操作。GPU0闭环未受影响。完整eager参考和错误已恢复性移动到 `results/idea2/run_20261020/attempts/e1b_graph_gpu2_attempt1/`，没有删除原数据。

第二次E1b会话42259、PID2296854已正常exit0，输出 `e1b_graph_gpu2/`。显式graph-compatible适配器仅预计算固定scalar／mask／position，保持全部NN计算和10步原始CUDA FP32时间递推；真实GPU E0全部误差0。原始eager、静态元数据eager、graph各有串行／并发两臂、20warmup＋100measure，同GPU2 UUID，数值／地址／独立pool审计通过。匹配静态eager串行全任务P50=297.91ms，graph串行64.95ms，graph并发58.18ms；graph动作就绪则由串行27.64ms变成并发58.17ms。仅是固定输入微基准，不代表graph闭环部署完成。

23:50之后的原始数据进度审计：95/112条已完整检查，缺17条，错误0；`audit_progress_graph_20261020T1550Z.json` 已开启 `--require-graph`，E1b通过。由于闭环仍在继续，此receipt为进度快照、passed=false，不当作最终验收。

分析稿已保存：[idea2探索报告（进行中）](idea2_exploratory_report.md)。包含E0/E1/E2、首轮28条表格，以及Spatial腕部图像出现3个uint8通道值相差1的渲染不完全确定性证据；不将自由闭环轨迹差异全部归因于并发。

idea1 v3.1当前只有独立代码与真实输入bank准备，9项CPU测试通过；初始manifest已冻结，未进行模型GPU调用或闭环评测，仍按先完成idea2的顺序执行。详见[v3.1准备状态](../idea1/idea1_v3_1_preparation_status.md)。

第一次诊断位于 `results/idea2/e0_spatial_first/`，保留为启动诊断。正式矩阵再次检查并保存源码身份。只有E0通过才开始并发；四套件检查及固定版本一致性通过后才开始闭环E3。

## 历史待办（已全部完成）

1. E3三模式×刷新间隔的小规模闭环，112个唯一配对episode。
2. E1b独立CUDA Graph微基准已完成，待并入最终图表与完成审计。
3. 数据审计、性能表、成功率、XYZ／视频索引和最终分析报告。

GPU1检查时有他人占用，未触碰。闭环始终只使用GPU0；仅独立微基准短时使用空闲GPU2（UUID `GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25`）。这不是多GPU流水线，也没有启动idea1 v3.1的GPU实验。
