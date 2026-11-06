# 已确认发布 draft 跨 suite 结果的科研图

[PNG](draft_transfer_confirmation.png)、[PDF](draft_transfer_confirmation.pdf)、[SVG](draft_transfer_confirmation.svg)与[逐条件CSV](confirmed_metrics.csv)来自已完成的400条独立初态确认记录，未读取当前fast path实验的成功率。

四个面板分别展示纯head成功数、完整FLASH与Full成功数、动作接受率/请求回退比例，以及原生推理总毫秒除以实际执行动作数。目标suite内保持任务、初态和推理合同不变；外来来源为另一个冻结suite。纯draft没有验证/Full回退，接受率为不适用，CSV中留空。

接受率按验证动作预算加权；拒绝→Full为请求比例，两者分母不同。计时采用全部init21–24，包含执行0动作的拒绝请求及冷编译、轨迹与共享负载影响，不等于独占部署加速。匹配FLASH也并未在两个目标的全部初态中都快于Full；repeat-task子集没有被改称完全热态。

图只支持发布head在共同Pi0推理合同下的有限迁移性。原训练/归一化溯源不完整，不能排除所有历史预处理混淆。图上的Holm2是原预定case-level检验；[任务级事后敏感性分析](../draft_task_sensitivity/README_zh.md)另外报告聚类口径，不替代原主分析，也不增加独立案例。

[生成脚本](make.py)核对独立审阅与统计文件哈希及其成功/计时数据一致性，[provenance.json](provenance.json)记录导出物哈希。Full的`source_suite`原字段不代表draft来源，因此派生CSV对Full将该列留空；原数据保持不变。
