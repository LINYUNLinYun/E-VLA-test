# E4 输入偏移诊断：相同 SVD 随机数控制

主方案与主数据保持冻结。代码核查发现，Full-input 和 shifted-input 的两个低秩拟合调用采用了不同的固定随机SVD种子；在输出质量揭盲/选参前增加此控制，以隔离输入分布与随机截断近似的影响。原参数与原始指标均保留。

仍用原80个偶数task校准请求及相同token选择、ridge、rank8/32、FP32 PyTorch和同一权重。只将shifted-input拟合的随机SVD种子改成Full-input分支的340400+layer；不调整rank或其他科学参数，不使用留出任务拟合。

重新计算Full/shifted输入上的hidden误差，不执行新动作、闭环或多层策略，不重跑E2或五种方法的动作比较。保留所有160请求，留出48请求作为主汇总；相同rank下比较Full-input拟合与matched-seed shifted-input拟合。L1–3输入尚未受V14上游skip影响，须作为负对照验证两组拟合一致。

单独保存本补充的源hash、参数、校准身份、匹配种子的adapter、逐请求hidden误差、留出汇总和验收。不能用hidden误差降低声称闭环成功率提高。
