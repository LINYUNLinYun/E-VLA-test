# v3.4 E1补充：复用层与复用轮次独立验证

2026-10-29。用户指定两项独立改动，有效果再组合。原v3.4代码、manifest及结果不改写。

共同协议：π0原checkpoint、BF16/Triton、V14原始1-based skip[3,4,9,11]、H50/R5、draft D2、Full D10；P=2仍为Full→draft1→draft2。无verify/夹爪保护。20Hz、官方init、seed7/10步settling、原prompt与请求noise规则，沿用原各suite上限。GPU0/3启动前验实际占用；GPU1/2不使用。

独立A：L17–18在draft1/2均复用最近Full；当前L16 KV保留。
独立B：L16–18仅draft1复用，draft2完整计算当前V14 KV，不提前Full、不用上次draft KV、不覆盖独立Full快照。
组合C（条件触发）：L17–18仅draft1复用，draft2当前V14。

每项先四suite×10task×init14，共40条；有效失败不重试。复用原init14的periodic_v14(37/40)、periodic_old16(31/40)，扩样时复用init14–18共200条对照。先通过原4suite同输入门（新增cutoff17与完整计算后换KV对照）、原跨服务参考输出与每case初始输入/Full原始动作对照才合并旧成绩，容差固定atol=rtol=1e-5；失败停止，不调容差。标明历史对照与新服务差异；成本不直接比跨运行的绝对wall。

独立效果门：首40条较旧L16两轮复用净多成功≥2，且同卡同输入热态周期成本低于无复用周期V14。两项均达门才跑C的40条；否则不触发组合，记明原因。扩样门：候选首40条至少35成功（较周期V14最多少2），且周期成本不高于V14，合格者固定补init15–18至200条。A/B独立评估；C同门。不得按扩样结果回调条件。新闭环上限600条，非官方50-init全量基准。

同输入测速：复用原Full首40轨迹每episode最早/最晚已冻结机制pair，共80对；各卡按task%2分片，先在该shape/route预热，AB/BA三次，记录full10/v14/old16/old17完整wall和模块时间。周期成本为(Full+draft1+draft2)/3；B为(Full+old16+v14)/3，C为(Full+old17+v14)/3。单请求快与周期/自由闭环每动作成本、完整评测吞吐分开。

每条件逐case保存成功/步数/轨迹/EEF姿态/原始decoded动作/seed/Full年龄/路径与耗时。新分支route必须验证F-H-H或F-H-V序列，draft1年龄5步/draft2年龄10步，B/C第二轮不读取旧KV。40/200口径和Spatial独列，配对恢复/损失及task聚类探索区间。

交付：冻结manifest、gate与历史复用收据、条件触发决策、逐案例与逐请求数据、时延/质量表、中文报告与最终验收。所有技术错误单列且有界恢复；不中断其他GPU任务。
