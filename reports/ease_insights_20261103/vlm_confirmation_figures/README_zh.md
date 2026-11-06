本图仅使用本轮已完成的600条正式VLM报告，不重新运行统计或筛选方向。

A：四个suite的Full/Jac/MSE成功率，直接标注成功数/50，不附假定案例独立的误差条。
B：原主分析Spatial和Long，以及发现后、独立复验前登记的Goal补充方向；展示正式报告已有的任务聚类95%区间。正值顺着预定方向。三方向Holm3均为1；原主假设及补充假设未复现，不意味着路径等价。

可导出文件：`vlm_independent_confirmation.png`（300 dpi）、`.pdf`和`.svg`；CSV为绘图的全部数据。英文完整图注见`caption.txt`。

| 条件 | 成功数 | 执行动作数 | 策略请求数 | policy ms/执行动作 | RPC ms/执行动作 |
|---|---:|---:|---:|---:|---:|
| Full | 182/200 | 35418 | 7163 | 28.987 | 29.322 |
| Jac | 170/200 | 40433 | 8150 | 15.763 | 16.069 |
| MSE | 170/200 | 40169 | 8097 | 16.245 | 16.532 |

该成本表统一使用正式报告全部200个episode的累计policy/RPC毫秒数，除以累计实际执行动作数，保留成功与失败案例；不是逐case比值的均值。policy沿用本轮VLM报告的策略调用墙钟字段；RPC为客户端往返。没有对历史native计时作跨实验等同。
不同路径产生的轨迹、动作数、冷启动和外部争用可能不同。数值不是独占GPU速度、质量非劣性或闭环加速的证明；episode墙钟之和也不是并行总耗时。

复现（在benchmark根目录运行）：

```bash
python3 reports/ease_insights_20261103/vlm_confirmation_figures/generate_figures.py
```

输入报告、manifest/selection身份及输出SHA见`provenance.json`。原主分析的Holm2没有改变；图B统一呈现另行登记的三检验家族Holm3。
