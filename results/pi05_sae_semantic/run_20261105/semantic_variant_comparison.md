# 同一 64 观测上的语义读出对照

三种读出都使用冻结的 π0.5、同一对发布 SAE 和同一发现/校准划分。特征关联仅由发现集计算，校准集只检查分数阈值；这些是探索性结果，不是独立测试或成功率保证。

| 读出 | 概念 | 层 | 特征数 | 校准 precision | recall | FPR | 支持 |
|---|---|---:|---:|---:|---:|---:|---|
| released_all | object/bowl | 5 | 1 | — | 0.000 | 0.000 | False |
| released_all | object/bowl | 16 | 0 | — | — | — | False |
| released_all | object/stove | 5 | 0 | — | — | — | False |
| released_all | object/stove | 16 | 0 | — | — | — | False |
| released_all | spatial/on | 5 | 1 | 0.600 | 0.375 | 0.250 | False |
| released_all | spatial/on | 16 | 0 | — | — | — | False |
| dictionary_all | object/bowl | 5 | 3 | 1.000 | 0.500 | 0.000 | True |
| dictionary_all | object/bowl | 16 | 1 | — | 0.000 | 0.000 | False |
| dictionary_all | object/stove | 5 | 1 | 0.500 | 0.250 | 0.083 | False |
| dictionary_all | object/stove | 16 | 1 | — | 0.000 | 0.000 | False |
| dictionary_all | spatial/on | 5 | 1 | 1.000 | 0.500 | 0.000 | True |
| dictionary_all | spatial/on | 16 | 1 | 0.500 | 0.125 | 0.125 | False |
| dictionary_instruction | object/bowl | 5 | 1 | 1.000 | 0.500 | 0.000 | True |
| dictionary_instruction | object/bowl | 16 | 1 | — | 0.000 | 0.000 | False |
| dictionary_instruction | object/stove | 5 | 1 | 1.000 | 0.500 | 0.000 | True |
| dictionary_instruction | object/stove | 16 | 1 | — | 0.000 | 0.000 | False |
| dictionary_instruction | spatial/on | 5 | 1 | 1.000 | 0.500 | 0.000 | True |
| dictionary_instruction | spatial/on | 16 | 1 | 0.500 | 0.125 | 0.125 | False |

released_all 直接使用公开候选池；dictionary_all 在同一 SAE 字典中重新关联特征；dictionary_instruction 只用已保存的指令 token 读出。没有新增 VLM 前向或 SAE 训练。

指令 token 含有视觉上下文，但当前标签是静态任务语义；不能把分类表现直接解释为操作阶段识别。Prompt-only 的额外价值对照留待策略阶段。

后续 FFN 矩阵使用冻结 L5 触发选样，L16 不做逐样本筛选；物理干预按观测/mask 缓存，各种语义读出可复用同一次前向。
