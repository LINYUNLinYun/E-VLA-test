# Evidence-based quota screen

Written after the completed B2 SM16 run and the actual hardware descriptor probe.

B2 ordinary full-GPU pipeline:17.4175 samples/s; same-B serial:16.4743. Green16:16.9729, with AE105.986ms and VLM76.323ms. SM16 therefore does not transfer its B1 advantage to B2; AE remains exposed after VLM/next Encoder. This argues for trying a moderately larger AE quota, rather than assuming more overlap or a smaller quota is beneficial.

The resource descriptor probe reports exact splits8/120,16/112,24/104,32/96,48/80,64/64. Test24 first for B2 and B4, keeping lead3 and all model/input/precision settings fixed. It is a midpoint between starving AE at16 and unnecessarily reducing VLM resources at32. Screen with short runs; promote promising candidates to100 groups and test32 as a neighboring control. Do not expand to64 unless remaining evidence warrants it. The8-SM case is not prioritized because the measured B2 AE16 branch is already the bottleneck.

All B>1 results retain failed strict B1-equivalence and are same-B scheduling diagnostics, not deployment claims. Per-lane same-B MAE/max tolerances stay unchanged.

Evidence: `stage2_b2_full_prepared_full/result.json`, `stage2_b2_sm16_prepared_full/result.json`, `stage2_sm_granularity_probe/result.json`.
