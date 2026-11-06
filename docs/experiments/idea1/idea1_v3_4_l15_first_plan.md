# E1：L15–18仅首轮复用

用户指定将复用起点提前到1-based L15，并只在Full后的第一轮draft复用。条件D15_first：Full-D10 → L15–18旧Full-KV draft-D2 → 当前V14 draft-D2，随后刷新Full。当前L1–14 KV按V14计算；旧快照独立保留，不以draft覆盖。第二轮不读取旧KV，也不提前Full。

沿用π0 checkpoint、BF16/Triton、V14 skip[3,4,9,11]、H50/R5、P=2、20Hz、官方init、seed7、10步settling、原prompt/请求噪声与各suite上限；无verify或夹爪保护。原实验代码和结果不改写。

先四suite×10task×init14，共40条。延续上轮冻结扩样门：首40条至少35成功，且同卡同输入周期模型路径耗时不高于无复用周期V14，合格则固定补init15–18至200条。有效失败不重试；技术中断保留并有界恢复。未达到门槛则报告40条结果，不声称200条结论。

对照复用periodic_v14、periodic_old16及上一轮B16_first、C17_first。合并前核验同输入Full初始输出，atol=rtol=1e-5；L15截断须等价于完整计算当前V14后替换L15–18。实际路由逐请求核验F-H-V、旧层数4、第一轮age5；第二轮旧层数0。保存全H50 raw/decoded、实际动作、EEF姿态、请求seed/输入hash/年龄/时延，不将H50未执行后缀误差解释成闭环质量。

同输入测速沿用上轮冻结80对，预热后AB/BA三次，测Full10、V14、old15、old16、old17。D15周期=(Full10+old15+V14)/3；同时给B16/C17及无复用周期成本。该成本含Full刷新，不含RPC/仿真/加载。与闭环策略成本、吞吐分别报告。

GPU0/1/2/3均为候选。用户要求换其他卡后，在尚未启动模型、完成0条时，将原GPU0/3范围扩至本机全部4张卡；原冻结清单、方案与代码已归档至resource_revisions/gpu03_original。科学配置、case/seed、扩样门及数值容差不变。

约30秒检查占用；没有其他compute进程且显存≤1500MiB才启动，禁止停止外部任务。空闲一张即开始；其他候选卡释放后可加入待处理task分片。task/seed与完成顺序无关。此为探索评测，非官方50-init全量基准。
