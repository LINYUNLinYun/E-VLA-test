# Old-KV D10 trace analysis

CPU-only analysis of existing traces. GPU intervals are never clipped to the Host response. No layer/iteration labels inferred.

| Trace / group | Requests | AE busy ms | Vision busy ms | VLM busy ms | AE×Vision ms/request | AE×VLM ms/request |
|---|---:|---:|---:|---:|---:|---:|
| cewp_d10_vision_wait_modulation.2 / cewp/continuous/graph_ae_high | 5 | 29.724 | 12.968 | 30.888 | 12.420 | 0.000 |

All measured and unscoped kernels are retained in kernels.csv. Completeness, trailing VLM kernels, Graph launch mapping, resources and API statistics are in each report.json. Prefill_other is retained rather than guessed as Vision/VLM. A missing module is not silently treated as measured zero.

Matched signature comparisons use phase + module + kernel name + launch geometry; verify unmatched signatures and call-count differences before interpreting slowdown. These short profile measurements are not formal latency/throughput results.

Host request identity is a submission identity, not the KV observation identity; numerical equivalence and old-KV freshness need the benchmark result.json, not this trace alone.
