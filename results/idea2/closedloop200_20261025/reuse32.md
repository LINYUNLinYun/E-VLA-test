# Reuse proof: 32 existing episodes

Status: **passed**. CPU-only verification; no environment or GPU was started.

Reuse 8 cases per arm: fresh18, sm16_tail0, sm32_tail0, sm32_tail2. Case IDs: 0, 25, 50, 75, 100, 125, 150, 175. The four suites follow object, spatial, goal, 10 order; each uses task 0/5, init 0.

All recorded source hashes and checkpoint file size/mtime metadata match. D10/H10/R5, environment/model seed rules, settle steps, limits, SM/lead/tail semantics, old/new source identities, all input/action hashes, completions and clean exit/render guards pass. Initial observation, input, and decoded action hashes agree across all four arms for each of the eight cases.

The JSON contains absolute source directories and per-request evidence. Keep these original artifacts unchanged. Reuse success/coverage only; do not treat their old elapsed time as throughput of the new run. No full weight hash or GPU noise-byte hash is claimed.

| Arm | Case | Suite | Task | Init | Success | Steps | Requests |
|---|---:|---|---:|---:|---|---:|---:|
| fresh18 | 0 | libero_object | 0 | 0 | True | 154 | 31 |
| fresh18 | 25 | libero_object | 5 | 0 | True | 130 | 26 |
| fresh18 | 50 | libero_spatial | 0 | 0 | True | 110 | 22 |
| fresh18 | 75 | libero_spatial | 5 | 0 | True | 101 | 21 |
| fresh18 | 100 | libero_goal | 0 | 0 | True | 114 | 23 |
| fresh18 | 125 | libero_goal | 5 | 0 | True | 129 | 26 |
| fresh18 | 150 | libero_10 | 0 | 0 | True | 262 | 53 |
| fresh18 | 175 | libero_10 | 5 | 0 | True | 181 | 37 |
| sm16_tail0 | 0 | libero_object | 0 | 0 | True | 156 | 32 |
| sm16_tail0 | 25 | libero_object | 5 | 0 | True | 124 | 25 |
| sm16_tail0 | 50 | libero_spatial | 0 | 0 | True | 84 | 17 |
| sm16_tail0 | 75 | libero_spatial | 5 | 0 | True | 96 | 20 |
| sm16_tail0 | 100 | libero_goal | 0 | 0 | True | 125 | 25 |
| sm16_tail0 | 125 | libero_goal | 5 | 0 | True | 137 | 28 |
| sm16_tail0 | 150 | libero_10 | 0 | 0 | True | 289 | 58 |
| sm16_tail0 | 175 | libero_10 | 5 | 0 | True | 206 | 42 |
| sm32_tail0 | 0 | libero_object | 0 | 0 | True | 137 | 28 |
| sm32_tail0 | 25 | libero_object | 5 | 0 | True | 131 | 27 |
| sm32_tail0 | 50 | libero_spatial | 0 | 0 | True | 91 | 19 |
| sm32_tail0 | 75 | libero_spatial | 5 | 0 | True | 192 | 39 |
| sm32_tail0 | 100 | libero_goal | 0 | 0 | True | 143 | 29 |
| sm32_tail0 | 125 | libero_goal | 5 | 0 | True | 136 | 28 |
| sm32_tail0 | 150 | libero_10 | 0 | 0 | True | 308 | 62 |
| sm32_tail0 | 175 | libero_10 | 5 | 0 | True | 204 | 41 |
| sm32_tail2 | 0 | libero_object | 0 | 0 | True | 139 | 28 |
| sm32_tail2 | 25 | libero_object | 5 | 0 | True | 132 | 27 |
| sm32_tail2 | 50 | libero_spatial | 0 | 0 | True | 89 | 18 |
| sm32_tail2 | 75 | libero_spatial | 5 | 0 | True | 102 | 21 |
| sm32_tail2 | 100 | libero_goal | 0 | 0 | True | 168 | 34 |
| sm32_tail2 | 125 | libero_goal | 5 | 0 | True | 143 | 29 |
| sm32_tail2 | 150 | libero_10 | 0 | 0 | True | 285 | 57 |
| sm32_tail2 | 175 | libero_10 | 5 | 0 | True | 174 | 35 |
