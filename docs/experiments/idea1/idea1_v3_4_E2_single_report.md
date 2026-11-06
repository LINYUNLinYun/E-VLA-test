# E2：PyTorch 单层 gate Jacobian 证据

40个真实请求、四suite各10task、init14；每请求独立检查L1–17三档gate。D2完整两步FM，固定H50/R5与生成噪声，参考与候选全部为同一FP32 PyTorch路径。E1/E3未重跑。

**主要发现：局部扰动预测准确；直接整层跳过时，平移和旋转排序仍优于层间余弦，但向量预测误差显著增大，夹爪没有优势。**

|gate|动作|Jacobian排序ρ|层间余弦排序ρ|方向余弦|向量相对误差|
|---|---|---|---|---|---|
|0.9|translation|0.978|-0.043|0.976|0.185|
|0.9|rotation|0.969|-0.001|0.978|0.191|
|0.9|gripper|0.941|0.053|0.960|0.259|
|0.5|translation|0.774|0.028|0.710|0.809|
|0.5|rotation|0.769|0.057|0.694|0.804|
|0.5|gripper|0.626|0.137|0.646|1.119|
|0.0|translation|0.525|0.163|0.395|1.193|
|0.0|rotation|0.478|0.269|0.405|1.204|
|0.0|gripper|0.353|0.372|0.358|1.490|

gate=0时Jacobian相对层间余弦的排序ρ配对差及suite分层task聚类探索95%区间：
- translation：+0.362，[+0.270, +0.450]。
- rotation：+0.209，[+0.119, +0.305]。
- gripper：-0.020，[-0.135, +0.093]。

热态完整forward平均69.45ms，完整35维Jacobian（含forward）3783.71ms；后者约为前者54.5倍。这是离线诊断证据，尚不能称廉价runtime跳层规则。

gate=0与实际bypass的680项检查通过；四suite真实输入上的autograd/有限差分检查通过。层间余弦与残差范数分别比较；连续夹爪敏感度不代替夹爪符号或任务成败。不能将单层证据外推成多层组合有效。

PyTorch/native Full-D2的前五步平移差代理平均0.195mm，逐请求差异保存在E2_native_compatibility.csv；两个后端保持分开，不要求FP32与BF16逐位等价。

[共同冻结方案](idea1_v3_4_E2_E4_single_plan.md) · [E2验收](../../../results/idea1_v3_4_E2_E4_single/E2_standalone_audit.json) · [原始单层数组](../../../results/idea1_v3_4_E2_E4_single/E2)

排序和向量相对预测误差在normalized raw动作空间内计算；decoded平移/旋转/夹爪指标逐请求另存，不把normalized排序ρ当成闭环成功率预测。
