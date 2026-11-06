# Early-return serial control: CPU trace audit

Control: `d10_async_serial_profile_20261024_v1` / `graph_async_serial`; comparator: `d10_oldkv_graph_encoder_overlap_20261024_v1` / `graph_encoder_overlap`.
Control contract passed: **True**. Exact signature/count comparison eligible: **True**.

| Check | Value |
|---|---|
| nonempty | True |
| all_requests_have_ae_vision_vlm | True |
| ae_x_vision_exactly_zero | True |
| ae_x_vlm_exactly_zero | True |
| ae_x_all_prefill_exactly_zero | True |
| ae_completed_by_api_return | True |
| every_request_prefill_continues_after_api_return | True |
| vision_waits_same_request_ae | True |
| vlm_follows_same_request_vision | True |
| graph_module_attribution_complete | True |
| launch_correlation_complete | True |

Actual AE×Vision: 0.000000000 ms; AE×VLM: 0.000000000 ms.
Prefill continues after API return: 8/8 measured requests.
Matched signatures: 136; unmatched control/overlap: 0/0.

The comparator/control ratio in slowdown_summary.csv is based on exact phase/module/kernel-name/geometry signatures. It is not an end-to-end speedup. Model output correctness, snapshot age, two-slot ownership and same-Graph capture owner must also pass the benchmark result checks.

Full GPU kernels are retained after the Host request ends. Vision means the complete prefill/vision Graph (including embedding/metadata), not pure Encoder arithmetic. No layer or iteration labels are inferred.
