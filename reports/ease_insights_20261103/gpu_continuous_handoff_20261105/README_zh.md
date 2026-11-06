# GPU 任务连续接续（资源执行修订）

用户要求：只等待本机自然空闲卡；连续空闲门槛现为15秒（先前为25秒）、每2秒检查；一旦开始使用GPU，任务结束后立即继续下一项，避免任务间主动释放。

当前状态（2026-11-05更新）：v3已在自然释放后获得并持续保留GPU0/1，两份真实冷门记录为15.494/15.514秒。原session与64MiB keeper跨发现任务、CPU筛选和独立复验保留；720条发现已独立检查通过。复验Full/local_long各200条完成，Jac已通过零案例端口恢复接上并开始提交；GPU1保持占用。跨任务持有已有实际边界证据，600条复验的科学结论仍待完成。

## 接续方式

原调度器仍是唯一分配GPU的进程，最多同时保留两张卡。每个资源会话在通过15秒空闲门后，先创建本会话拥有的64MiB CUDA分配，再加载原科学服务。该CUDA上下文和调度器继承的GPU锁跨任务保留。当前任务完成后立即取下一项；待CPU选择候选、准备确认实验或等待新的任务时也保持占用。

技术错误转为需要处理的状态并保留资源，不会重跑已经有效的成功或失败。修复后必须显式恢复，重试任务使用新的任务身份；旧记录保持不变。只有显式释放文件才在任务边界回收本会话的服务与keeper，随后结束资源槽位。进程或GPU上下文本身丢失会如实记录，无法保证抵御其他用户直接使用同一设备。

## 科学条件

本次修改属于资源编排修订。原冻结的模型、mask、采样日程、任务与初态、数值检验、逐案例产物及完成标记继续使用。每个条件仍启动新的原服务，避免跨条件残留；既有120条Jac结果（99成功、21有效失败）保留。新资源槽位的完成标记只表示资源释放，绝不充当科学成功标记。

CPU broker将原队列中已明确识别的科学任务一一转成不可改写的描述文件。描述文件只有在对应队列变更提交后才激活；实际执行仍读取冻结manifest并校验全部原科学源文件。

## 安装与验证

v1准备包从未部署，保留为旧计划记录。最终资源源文件冻结后创建v2；安装只替换经身份校验的本轮CPU等待门和CPU分配器，使用同一份原调度器源码以2秒间隔扫描。任何已经进入模型执行的任务都不得被这一安装流程终止。

安装准备时，真实CUDA启动、显存与任务连续接续尚待空闲卡验证，CPU测试不能代替运行证据。后续实际获得及交接结果见下方；这句保留安装阶段的证据边界。

2026-11-05 03:00 UTC，25秒门已触发一次Full启动，但模型初始化遇到OOM，没有新增episode。旧PID与监督器状态因此变化，v2自动拒绝安装，未写签署授权。故另行保留技术失败原产物，并准备当前身份对应的v3。

## v3部署结果

最终96项CPU检查通过，独立审阅完整源文件、31条保留队列记录、5项科学任务映射与CPU进程身份。安装事务已完成，原CPU分配器及两CPU等待门已退出并reap；没有终止在途模型。原调度器源码不变，新PID600088、扫描2秒、最多2卡。监督器595772确认采用新调度器，broker600807运行。

计划、签核和部署证据均在 `results/ease_insights_20261103/gpu_resource_campaign_v3/`。v1/v2从未部署，原包完整保留。安装使用系统Python的pidfd；模型venv缺该API，第一次尝试在发送任何信号前即拒绝，随后同一签核包通过系统Python完成。

安装结束当时两个槽位均为pending，任务激活并不证明科学模型已启动。后续实际GPU获得、keeper与冻结数值门、案例生成及连续换任务已另行记录如下。

## 用户后续调整为15秒

2026-11-05，用户要求25秒缩短为15秒。两资源槽位当时均pending，没有已启动的旧等待门。仅将两行显式idle env与对应metadata共4个值改为15，poll仍为2。gate实际配置解析15/2；原调度器与broker均在提交后继续运行，监督器非terminal，冻结plan/source pins继续匹配。两个槽位是当前唯一冷分配入口，后续科学任务在原会话内接续。原始计划和先前部署记录的25秒保留历史原值。新审计在 `results/ease_insights_20261103/gate_policy_refreshes_15s/user15s_persistent_slots_1793848974166872136/`。

## 首次真实获得GPU与科学启动

2026-11-05 05:00:31UTC，四卡自然释放。原调度器分配GPU0/1，真实15秒门通过；session631907/631915与keeper632159/632160的PID/start/parent/nonce/UUID经独立[首次获得审计](first_acquisition/README_zh.md)核对。keeper于05:00:49附近分别呈460MiB/1compute，随后原科学服务加载并开始生成episode。GPU2/3后来被其他进程占用，本轮保留GPU0/1。

此刻已证明首次keeper-backed CUDA持有和新科学提交，尚未声称跨条件交接已验证、整卡独占或两个insight已完成。新case抽样验收与最终全阶段校验继续进行，所有有效策略失败保留。

## 首批科学案例抽样验收

[有界CPU验收](../vlm_local_search_runtime_acceptance/first_resource_session_cases/audit.json)为sample_pass：Full与local_long各3个已提交案例，加同case的3条Jac对照。身份、冻结source/manifest、输入/噪声、逐请求动作长度与导出schedule/mask/history、artifact hashes均通过；24组gate action数组与Jac参考逐项一致。Jac样本中的失败原样保留，成功不作为验收门。独立覆盖共126条condition-case记录（原Jac120+新6），不是全部正在增长的提交数；没有重算全部332检验或720案例，内部actual_NFE计数仍未独立导出。

## 首次任务边界与端口恢复

Full与local_long各完成120案例后，两个后继任务在新server/case启动前遇到Address already in use。两个session保留原keeper，端口随后ss无记录且plain bind可用。显式一次技术恢复只新增task ID及activation路径，原科学参数/输出位置不变；旧complete与attention outcomes保留，valid success/failure不重跑。resume收据均通过source/plan/scheduler/lease/keeper复核，新的Goal/Spatial服务已健康启动。

[双slot边界审计](first_handoff/README_zh.md)确认同session/keeper PID、start、PPID、nonce及64MiB context跨首个任务边界保留；watcher无目标UUID idle事件。这里验证CUDA持有连续，科学服务交接确有技术暂停，未声称无采样间隔瞬态或整卡独占。恢复完整事务在 `results/ease_insights_20261103/gpu_resource_campaign_v3/transactions/technical_port_retry_1793856599463613971/`。

## 发现完成后接入独立复验

720条发现已由[全量CPU检查](../vlm_local_search_runtime_acceptance/full_discovery/audit.json)接受：28,396次请求、140,775动作行、6个实际服务的保存数值门均通过。随后原监督器按冻结规则选中Jac/local_long并自动准备600条新初态复验。原两个session/keeper跨CPU筛选和准备阶段存活，Full/local_long复验在相同两张卡上启动，现各完成200条。

Jac首次接续在模型和案例启动前遇到端口占用，原任务0案例。端口恢复工具等待所有可领取任务槽位都plain-bind-ready后，通过[已审阅的一次性工具](port_retry_helper/README.md)追加`__port_retry1`描述文件、匹配activation并原子发布严格resume请求；只改变资源任务id与activation路径。原端口失败outcome、Full/Long完成outcome及所有有效策略成功/失败都保留。

slot01的resume收据确认`resumed`且source复核通过；重试被已空闲的slot00原子领取，在GPU0启动Jac服务。GPU1继续idle_holding。新服务的数值门、protocol和首案例输入身份已核对，案例继续增长；原session631907/631915、keeper632159/632160及其start/nonce均未更换。完整[恢复确认](../../../results/ease_insights_20261103/gpu_resource_campaign_v3/transactions/port_retry_helper_2ff6702a4c393369b9f8a6413b78429a/acknowledged.json)保留原attention会话与实际执行会话的区别。

本次等待端口期间保持GPU占用，科学服务确有暂停。只观察到了当时的bind失败及稍后的bind成功；`ss`取样时已无连接记录，不能据此断言历史错误由TIME_WAIT造成。最终600条记录检查和独立统计复算已准备，尚未将探索互补性写成复现结论。
