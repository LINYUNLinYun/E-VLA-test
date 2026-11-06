# COMPLETED — full runtime NVTX ranges

All3 requested files pass completion.json and canonical_execution.json. GPU3 released; no pending jobs or watchers. Do not restart model runners.

- Each file has8 whole GPU projections per requested label,72 verified core envelopes overall; exact first/last operation endpoints, individual kernels retained.
- Names: preprocess,encoder,vlm,AE. Graph replay scopes replace capture-node aliases; node:nvtx-live retains kernel detail.
- Native52520 kernels, strict10 decoded numerical comparisons against original whole JIT max/MAE0. Still a split-stage diagnostic.
- Each Torch file104936 kernels; no additional owner-thread synchronization or model/runtime edits.
- Strict idle guard stopped before pipeline startup on stale process-free96% utilization; initial native/fresh reused, only pending pipeline resumed. Original stop retained.
- Download: nsys/idea2_experiment/runtime_ranges_20261026/*.nsys-rep, same3 basenames. Close old UI reports and reopen; view GPU NVTX projection, CPU scopes indicate submission.
