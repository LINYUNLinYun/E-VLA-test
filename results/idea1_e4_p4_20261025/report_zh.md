# E4 P4刷新周期补测

主要观察：无verify时P4相对P2每动作成本变化-2.06%；保留verify时变化+2.15%，因此本轮未降低带验证策略的开销。四臂均40/40，同周期有/无验证的动作与结果完全相同；小幅成本差受固定suite顺序限制，仅作本批观察。

四suite各10task、init8，四臂各40例；同物理GPU3/同模型服务、每case四臂平衡顺序，每臂新环境。Xiaomi skip12、Full/draft D2、H10/R10、无夹爪保护。P=连续draft轮数；零接受Full重置，部分接受计一轮；验证保持tau=.8、q=.10/.05、原生1-q方向。旧300仅历史参照。

| 条件 | 成功 | policy ms/action | predict | ack | 动作 | 请求 | 每例平均动作/成功例均值 | episode worker秒 | episode/小时 | 动作/秒 |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| p2_periodic | 40/40 | 15.370 | 15.198 | 0.1722 | 5931 | 613 | 148.28/148.28 | 383.76 | 375.2 | 15.46 |
| p4_periodic | 40/40 | 15.053 | 14.881 | 0.1717 | 6011 | 619 | 150.28/150.28 | 373.77 | 385.3 | 16.08 |
| p2_verified | 40/40 | 18.222 | 18.054 | 0.1681 | 5931 | 613 | 148.28/148.28 | 389.51 | 369.7 | 15.23 |
| p4_verified | 40/40 | 18.614 | 18.444 | 0.1697 | 6011 | 619 | 150.28/150.28 | 397.20 | 362.5 | 15.13 |

policy=完整服务端predict+ack /实际动作，同卡同case；不是RPC wall或纯GPU时间。共享ack仍调用原Kalman.observe，四臂共同保留，不称最简Xiaomi实现。公用Xiaomi准备/重复Full检查、模型加载、环境/reset/RPC/仿真/IO均单列。

## 同卡配对

| 候选/参考 | 恢复 | 新失败 | 成本比 | 成本比bootstrap95% | 成功率差描述 |
|---|---:|---:|---:|---|---|
| p4_periodic/p2_periodic | 0 | 0 | 0.9794 | [0.9695,0.9893] | 样本无差异，bootstrap退化 |
| p4_verified/p2_verified | 0 | 0 | 1.0215 | [1.0117,1.0318] | 样本无差异，bootstrap退化 |
| p2_verified/p2_periodic | 0 | 0 | 1.1856 | [1.1789,1.1923] | 样本无差异，bootstrap退化 |
| p4_verified/p4_periodic | 0 | 0 | 1.2366 | [1.2264,1.2459] | 样本无差异，bootstrap退化 |

Bootstrap按suite分层重采样task/case（每task仅init8），同一case两臂一起重采样，2000次；每次重新计算各臂sum wall/sum实际动作再求比值，不把同episode的请求当独立样本。区间含1时只报观察值，不宣称稳定加速；区间也不消除固定suite顺序等系统混杂。

四臂全部40/40成功、所有配对结局差均为0，因此成功差的经验bootstrap得到[0,0]，这是边界退化，不是可用于总体差异推断的95%区间，不能证明质量等价或非劣。成本区间仍按上述任务分层计算。

## 每40任务累计策略wall（描述性补充）

| 模式 | P2累计policy秒 | P4累计policy秒 | 累计wall变化 | P2/P4实际动作 |
|---|---:|---:|---:|---|
| periodic | 91.1598 | 90.4822 | -0.743% | 5931/6011 |
| verified | 108.0760 | 111.8899 | +3.529% | 5931/6011 |

P4实际动作5931→6011（+1.35%），所以每动作成本变化不等于完成同一批40任务的累计策略耗时变化。以上仅服务端predict+ack，不含环境、公用warm或加载；主成本仍为每实际动作wall。

同周期verified与periodic逐case全部decoded动作及实际执行长度相等：P2: 40/40；P4: 40/40。实际已执行动作轨迹/Full-draft请求节奏/成功标签相等分别为P2: 40/40/40（各40）；P4: 40/40/40（各40）。behavior_comparisons.json保存不等案例首处分歧与输入/种子/路由；不能把轨迹差异仅凭总数直接归因验证器。验收长度使用原accepted_length；episode成功或步数上限可截短actual executed_length，两者不混算。

## 套件与路由

| 条件 | Spatial | Object | Goal | Long | Full请求/全部 | Full原因 | verify零/部分/全10 | draft KV轮数 max/mean | draft KV步数 max/mean |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| p2_periodic | 10/10 | 10/10 | 10/10 | 10/10 | 219/613 | {'startup': 40, 'periodic': 179} | 0/0/0 | 2/1.49 | 20/14.90 |
| p4_periodic | 10/10 | 10/10 | 10/10 | 10/10 | 136/619 | {'startup': 40, 'periodic': 96} | 0/0/0 | 4/2.41 | 40/24.14 |
| p2_verified | 10/10 | 10/10 | 10/10 | 10/10 | 219/613 | {'startup': 40, 'periodic': 179} | 0/0/394 | 2/1.49 | 20/14.90 |
| p4_verified | 10/10 | 10/10 | 10/10 | 10/10 | 136/619 | {'startup': 40, 'periodic': 96} | 0/0/483 | 4/2.41 | 40/24.14 |

上表draft KV年龄只统计实际正接受draft；验证年龄单独统计所有有distance/verify调用的请求（包括随后零接受Full）：
p2_verified: verify KV轮数max/mean=2/1.49，控制步数max/mean=20/14.90。
p4_verified: verify KV轮数max/mean=4/2.41，控制步数max/mean=40/24.14。
startup/periodic Full请求开头的旧age未用于本轮验证，不混入verify年龄；P4第4次draft可用到4轮/40步，下一periodic Full行的5轮/50步是刷新前旧age，不称验证KV年龄。

Full减少比例不能当完整推理加速比例；原Full与skip12单请求成本接近，且verify成本仍存在。不同策略自由闭环可能改变轨迹和总动作数。40例仅探索筛选，未校正多重比较，不证明总体非劣或部署优势。

## 准备与模块耗时

| 条件 | 公用warm秒 | env/reset秒 | RPC predict/ack秒 | physics/save秒 | verify ms/实际动作 | Full AE ms/Full请求 | draft AE ms/draft提案 |
|---|---:|---|---|---|---:|---:|---:|
| p2_periodic | 33.05 | 63.73/72.71 | 90.85/1.38 | 108.34/7.04 | 0.000 | 44.43 | 44.20 |
| p4_periodic | 32.85 | 58.37/71.39 | 90.17/1.39 | 109.15/5.91 | 0.000 | 44.10 | 43.99 |
| p2_verified | 32.64 | 56.96/71.12 | 107.83/1.36 | 107.75/7.22 | 2.912 | 44.05 | 44.06 |
| p4_verified | 32.99 | 57.58/73.29 | 111.61/1.39 | 109.27/7.13 | 3.514 | 44.22 | 43.97 |

请求级描述（不同自由闭环观测，不是固定同输入模块因果对照；Full条件均值包含拒绝后接管请求，拒绝项本身包括draft+verify+Full）：

| 条件 | Full次数/占比 | Full完整policy ms/request | 正接受draft次数 | draft完整policy ms/request | verify次数/AE均值ms | 零接受次数/完整policy均值ms |
|---|---|---:|---:|---:|---|---|
| p2_periodic | 219/35.73% | 156.52 | 394 | 144.37 | 0/0.00 | 0/0.00 |
| p4_periodic | 136/21.97% | 155.22 | 483 | 143.63 | 0/0.00 | 0/0.00 |
| p2_verified | 219/35.73% | 155.17 | 394 | 188.06 | 394/43.83 | 0/0.00 |
| p4_verified | 136/21.97% | 155.48 | 483 | 187.88 | 483/43.74 | 0/0.00 |

P4增加draft占比，也可能增加verify次数；若draft+verify比Full贵，即使Full减少，verified总成本仍可能上升。routing_summary.json额外保存零接受请求内draft AE/verify AE/Full AE及完整policy wall总和，不把这些模块条件均值解释为相同输入的因果benchmark。

顺序只在全局40case中平衡：每task恰有4suite、旋转依赖case index mod4，故同一suite内部四臂顺序固定。公用暖机与startup门禁减少冷态混杂，但不消除所有顺序/热态历史影响；若速度仅约1%，不称稳定提速，需未来独立反序复验。本轮不更改在途顺序、不追加矩阵。不同P轨迹/步数不同本身不证明验证部分接受；验证增量依据同P配对及真实accepted_length。

execution_audit.json核对160例的实际执行前缀、真实P计数/拒绝重置/部分接受、KV轮数与控制步年龄、periodic零verify、无夹爪保护及predict+ack总成本。40组初态RGB/state/启动Full raw+decoded在四臂逐位相等。原始观测首帧、动作、EEF世界XYZ/四元数、种子、噪声hash与逐请求时延保存在episode目录；无全量视频。

paired_results.json提供逐case恢复/新失败与同卡比率；routing_summary.json保存模块耗时总和与KV年龄；requests.csv保存逐请求计数/年龄。

## 结果解释

periodic: P4 40/40、P2 40/40，配对0恢复/0新失败；同卡完整policy ms/action比0.9794，观察成本变化-2.06%，配对task分层95%区间[0.9695,0.9893]。区间未包含1，但仍受固定suite条件顺序等限制。不能仅凭Full次数减少断言等幅加速，也不能以这40例宣布总体非劣。

verified: P4 40/40、P2 40/40，配对0恢复/0新失败；同卡完整policy ms/action比1.0215，观察成本变化+2.15%，配对task分层95%区间[1.0117,1.0318]。区间未包含1，但仍受固定suite条件顺序等限制。不能仅凭Full次数减少断言等幅加速，也不能以这40例宣布总体非劣。

## 完成与资源

160/160有效回合，技术异常attempt记录0；全量2464请求审计通过，40组初态/启动Full签名一致，16项冻结源码/依赖hash相同。GPU3 worker从资源核验至正常退出27.00分钟，模型加载24.27秒，吞吐355.5有效episode/小时；公用warm共131.52 worker秒另列。末次C/G为空，本轮租约inactive，GPU0/1/2未使用。无自动扩样或额外GPU实验。
