# E2闭环：最多连续跳两层（经初始观测修复）

120条主实验已完成。原首观测验收发现4个Long case在条件间不一致；补跑这些case的全部8个新旧对照，共32条。各条件构造前固定seed7并新建环境，其他协议不变。原记录未删除，4case所有条件统一用修复数据，其余36case复用原数据。全部8条件×40case的首观测及初态/BDDL/prompt/controller身份逐位核对通过；没有放宽hash门。

|指标|原mask|原成功（修复后）|max2 mask|新成功（修复后）|
|---|---|---|---|---|
|cosine4|[3, 4, 5, 6]|2/40|[2, 3, 5, 6]|10/40|
|jacobian4|[3, 15, 16, 17]|23/40|[3, 14, 16, 17]|19/40|
|action_mse4|[2, 3, 4, 5]|7/40|[2, 3, 5, 17]|21/40|

同一修复矩阵内V14为26/40，Full-D2为39/40。

|对照|恢复/新增失败|差值pp|探索95%区间pp|
|---|---|---|---|
|old_cosine4→new_cosine4|8/0|+20.00|[+7.50,+32.50]|
|old_jacobian4→new_jacobian4|2/6|-10.00|[-22.50,+2.50]|
|old_action_mse4→new_action_mse4|14/0|+35.00|[+22.50,+50.00]|
|new_cosine4→new_jacobian4|12/3|+22.50|[+5.00,+40.00]|
|new_cosine4→new_action_mse4|13/2|+27.50|[+10.00,+42.50]|

主120条修复前原始计数：余弦10/40、Jacobian19/40、动作影响21/40；它们是执行记录，不应替代上表经相同首观测核对的结果。修复涉及的原旧对照也统一更新，不能混用原旧聚合数。
这是针对既有结果提出约束后的探索性配对检查，非全新独立确认集。区间按suite分层task聚类bootstrap，未证明总体非劣，也不把单个mask输赢推广为指标普遍有效/失效。

[原冻结方案](idea1_v3_4_E2_closedloop_max2_plan.md) · [技术修复方案](idea1_v3_4_E2_max2_reset_repair_plan.md) · [全部数据来源](../../../results/idea1_v3_4_E2_max2_repair/episodes.csv) · [身份验收](../../../results/idea1_v3_4_E2_max2_repair/identity_audit.json)
