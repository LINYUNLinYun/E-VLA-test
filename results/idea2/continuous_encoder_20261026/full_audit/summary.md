# Completed continuous-runtime audit

Status: passed. 300 episodes; 53928 predictions; 69763 actual actions. Bootstrap max error: 0.0.

All target seeds, R5 consumption, discarded-candidate nonexecution, actual-frame digest chains, pending admissions, deferred Event counts, terminal resets, stopped workers and frozen sources passed. Valid task failures remain included.

| Arm | Success | Actor Hz mean | Next Encoder jobs | Positive Event windows |
|---|---:|---:|---:|---:|
| fresh18_no_encoder | 75/100 | 19.917 | 0 | 0 |
| sm16_no_encoder | 72/100 | 19.914 | 0 | 0 |
| sm16_encoder | 84/100 | 19.910 | 19098 | 19066 |

**Event envelope overlap is not actual Kernel overlap.** No per-episode Kernel intersection is claimed. Cold bootstrap ages and pauses are retained with separate steady-input summaries.
