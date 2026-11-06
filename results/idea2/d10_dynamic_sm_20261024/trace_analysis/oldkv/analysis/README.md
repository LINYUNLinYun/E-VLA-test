# Old-KV D10 trace analysis

CPU-only analysis of existing traces. GPU intervals are never clipped to the Host response. No layer/iteration labels inferred.

| Trace / group | Requests | AE busy ms | Vision busy ms | VLM busy ms | AE×Vision ms/request | AE×VLM ms/request |
|---|---:|---:|---:|---:|---:|---:|
| d10_oldkv_graph_serial_20261024_v1 / cewp/continuous/graph_serial | 8 | 25.864 | 9.807 | 31.026 | 0.000 | 0.000 |
| d10_oldkv_graph_encoder_overlap_20261024_v1 / cewp/continuous/graph_encoder_overlap | 8 | 29.595 | 12.961 | 31.532 | 11.103 | 0.000 |
| d10_oldkv_graph_ae_first_20261024_v1 / cewp/continuous/graph_ae_first | 8 | 34.105 | 12.602 | 34.336 | 10.776 | 10.397 |

All measured and unscoped kernels are retained in kernels.csv. Completeness, trailing VLM kernels, Graph launch mapping, resources and API statistics are in each report.json. Prefill_other is retained rather than guessed as Vision/VLM. A missing module is not silently treated as measured zero.

Matched signature comparisons use phase + module + kernel name + launch geometry; verify unmatched signatures and call-count differences before interpreting slowdown. These short profile measurements are not formal latency/throughput results.

Host request identity is a submission identity, not the KV observation identity; numerical equivalence and old-KV freshness need the benchmark result.json, not this trace alone.
