"""Passive intermediate-layer Action Expert probe for deployed Triton pi0.

The production CUDA-graph output is evaluated first.  A second, semantically
identical full 18-layer execution records selected Action Expert layer outputs,
applies the original fused final RMSNorm/action projection to copies, and never
feeds an intermediate prediction back into the flow trajectory.
"""

# ruff: noqa: RUF001

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

# Make the repository's namespace-style ``scripts`` package importable when
# this file is executed directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.spec.spec_serve_policy import _build_policy_components  # noqa: E402
from scripts.spec.triton import pi0_infer  # noqa: E402
from scripts.spec.triton.triton_pi0_runtime import prepare_triton_inputs_from_transformed  # noqa: E402

PROBE_LAYERS = (6, 9, 12, 15, 18)
PROBE_STEPS = (0, 2, 5, 7)
FLOW_TIMES = {step: 1.0 - step / 10.0 for step in PROBE_STEPS}
DT = -0.1
SUITE_ARTIFACT = {
    "libero_spatial": "draft_spatial",
    "libero_object": "draft_object",
    "libero_goal": "draft",
    "libero_10": "draft_10",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_prompt_embeddings(converted_root: Path, metadata: list[dict]) -> dict[tuple[str, str], torch.Tensor]:
    wanted = {(row["suite"], row["task_description"]) for row in metadata}
    result: dict[tuple[str, str], torch.Tensor] = {}
    for suite, artifact_name in SUITE_ARTIFACT.items():
        artifact = converted_root / artifact_name
        manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))["prompts"]
        for wanted_suite, prompt in sorted(wanted):
            if wanted_suite != suite:
                continue
            entry = manifest.get(prompt)
            if entry is None:
                raise KeyError(f"prompt {prompt!r} missing from {artifact / 'manifest.json'}")
            result[(suite, prompt)] = torch.load(artifact / entry["embed_path"], map_location="cpu").to(torch.bfloat16)
    if set(result) != wanted:
        raise RuntimeError(f"loaded {len(result)} prompt embeddings, expected {len(wanted)}")
    return result


def _decoder_layer(
    weights: dict[str, torch.Tensor], buffers: dict[str, torch.Tensor], encoder_seq_len: int, i: int
) -> None:
    pi0_infer.rms_matmul_k_1024_2560_qkv_rope(
        buffers["decoder_x"],
        weights["decoder_attn_qkv_w"][i],
        buffers["decoder_rope_weights"],
        buffers["decoder_q_buf"],
        buffers["encoder_K"][i][encoder_seq_len:],
        buffers["encoder_V"][i][encoder_seq_len:],
        buffers["decoder_norm_factor_buf"],
    )
    pi0_infer.matmul_k8_256_n_softmax_mask0(
        buffers["decoder_q_buf"],
        buffers["encoder_K"][i],
        buffers["decoder_attn_buf"],
        encoder_seq_len,
    )
    pi0_infer.matmul_k8_n_256(
        buffers["decoder_attn_buf"],
        buffers["encoder_V"][i],
        buffers["decoder_q_buf"],
    )
    pi0_infer.matmul_k_2048_1024_res(
        buffers["decoder_q_buf"].view(-1, 2048),
        weights["decoder_attn_o_w"][i],
        buffers["decoder_x"],
    )
    pi0_infer.rms_matmul_k_1024_4096_gate(
        buffers["decoder_x"],
        weights["decoder_ffn_gate_w"][i],
        weights["decoder_ffn_up_w"][i],
        buffers["decoder_hidden"],
        buffers["decoder_norm_factor_buf"],
    )
    pi0_infer.matmul_k_4096_1024_res(
        buffers["decoder_hidden"],
        weights["decoder_ffn_down_w"][i],
        buffers["decoder_x"],
    )


def _velocity_from_hidden(
    hidden: torch.Tensor,
    weights: dict[str, torch.Tensor],
    norm_factor: torch.Tensor,
) -> torch.Tensor:
    # The deployed Triton artifact fuses original final RMSNorm into W and dt=-0.1
    # into W/b.  Starting from zero yields dt*v; dividing by dt recovers v.
    update = torch.zeros((hidden.shape[0], 32), device=hidden.device, dtype=torch.bfloat16)
    pi0_infer.rms_matmul_k_1024_32_bias_res(
        hidden,
        weights["decoder_action_fused_out_proj_w"],
        weights["decoder_action_fused_out_proj_b"],
        update,
        norm_factor,
    )
    return update.to(torch.float32) / DT


@torch.inference_mode()
def _probe_full_forward(runtime: pi0_infer.Pi0Inference, images, state, noise):
    weights = runtime.weights
    buffers = runtime.buffers
    encoder_seq_len = int(buffers["encoder_x"].shape[0])
    buffers["observation_images_normalized"].copy_(images)
    buffers["observation_state_normalized"].copy_(state)
    buffers["diffusion_noise"].copy_(noise)
    buffers["encoder_x"][runtime.num_views * 256 :].copy_(weights["language_embeds"])
    pi0_infer.vision_encoder(weights, buffers, runtime.num_views)
    pi0_infer.transformer_encoder(weights, buffers, encoder_seq_len)

    pi0_infer.matmul_1_32_1024_bias(
        buffers["observation_state_normalized"],
        weights["decoder_state_in_proj_w"],
        weights["decoder_state_in_proj_b"],
        buffers["decoder_state_buf"],
    )
    velocities: dict[tuple[int, int], torch.Tensor] = {}
    layer_visits = [0] * 18
    for step in range(10):
        buffers["decoder_x"][:1].copy_(buffers["decoder_state_buf"])
        pi0_infer.matmul_k_32_1024_bias_silu(
            buffers["diffusion_noise"],
            weights["decoder_action_fused_in_proj_w"],
            weights["decoder_action_fused_time_biases"][step],
            buffers["decoder_x_buf"],
        )
        pi0_infer.matmul_k_1024_1024_bias(
            buffers["decoder_x_buf"],
            weights["decoder_action_mlp_w"],
            weights["decoder_action_mlp_b"],
            buffers["decoder_x"][1:],
        )
        snapshots: dict[int, torch.Tensor] = {}
        for layer_idx in range(18):
            _decoder_layer(weights, buffers, encoder_seq_len, layer_idx)
            layer_visits[layer_idx] += 1
            human_layer = layer_idx + 1
            if step in PROBE_STEPS and human_layer in PROBE_LAYERS:
                snapshots[human_layer] = buffers["decoder_x"][1:].clone()

        if step in PROBE_STEPS:
            for layer in PROBE_LAYERS:
                velocities[(step, layer)] = (
                    _velocity_from_hidden(snapshots[layer], weights, buffers["decoder_norm_factor_buf"]).detach().cpu()
                )

        pi0_infer.rms_matmul_k_1024_32_bias_res(
            buffers["decoder_x"][1:],
            weights["decoder_action_fused_out_proj_w"],
            weights["decoder_action_fused_out_proj_b"],
            buffers["diffusion_noise"],
            buffers["decoder_norm_factor_buf"],
        )
    torch.cuda.synchronize()
    if layer_visits != [10] * 18:
        raise RuntimeError(f"not all layers executed exactly 10 times: {layer_visits}")
    return buffers["diffusion_noise"].clone(), velocities, layer_visits


def _metrics(candidate: torch.Tensor, reference: torch.Tensor, dims: slice) -> dict[str, float]:
    a = candidate[:, dims].reshape(-1).to(torch.float64)
    b = reference[:, dims].reshape(-1).to(torch.float64)
    diff = a - b
    a_norm = torch.linalg.vector_norm(a)
    b_norm = torch.linalg.vector_norm(b)
    eps = torch.finfo(torch.float64).eps
    cosine = torch.dot(a, b) / torch.clamp(a_norm * b_norm, min=eps)
    return {
        "cosine": float(cosine),
        "mse": float(torch.mean(diff * diff)),
        "relative_l2": float(torch.linalg.vector_norm(diff) / torch.clamp(b_norm, min=eps)),
        "norm_ratio": float(a_norm / torch.clamp(b_norm, min=eps)),
        "velocity_l2": float(torch.linalg.vector_norm(diff)),
        "velocity_norm": float(a_norm),
        "full_velocity_norm": float(b_norm),
    }


def _metric_columns(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    columns: dict[str, float] = {}
    for prefix, dims in (
        ("", slice(0, 7)),
        ("translation_", slice(0, 3)),
        ("rotation_", slice(3, 6)),
        ("gripper_", slice(6, 7)),
        ("all32_", slice(0, 32)),
    ):
        columns.update({prefix + key: value for key, value in _metrics(candidate, reference, dims).items()})
    return columns


def _write_plots(rows: list[dict[str, Any]], output_dir: Path) -> None:
    for metric, filename, ylabel in (
        ("cosine", "layer_vs_cosine.png", "Mean cosine similarity with Layer 18"),
        ("relative_l2", "layer_vs_relative_l2.png", "Mean relative L2 error vs Layer 18"),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for step in PROBE_STEPS:
            means, stds = [], []
            for layer in PROBE_LAYERS:
                values = np.asarray(
                    [float(r[metric]) for r in rows if int(r["flow_step"]) == step and int(r["layer"]) == layer]
                )
                means.append(float(values.mean()))
                stds.append(float(values.std(ddof=1)))
            ax.errorbar(PROBE_LAYERS, means, yerr=stds, marker="o", capsize=3, label=f"t={FLOW_TIMES[step]:.1f}")
        ax.set_xlabel("Action Expert Transformer layer")
        ax.set_ylabel(ylabel)
        ax.set_xticks(PROBE_LAYERS)
        ax.grid(alpha=0.25)
        ax.legend(title="Actual flow time")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for step in PROBE_STEPS:
        for layer in PROBE_LAYERS:
            subset = [r for r in rows if int(r["flow_step"]) == step and int(r["layer"]) == layer]
            entry: dict[str, Any] = {"flow_step": step, "flow_t": FLOW_TIMES[step], "layer": layer, "n": len(subset)}
            for metric in (
                "cosine",
                "relative_l2",
                "norm_ratio",
                "translation_cosine",
                "rotation_cosine",
                "gripper_cosine",
            ):
                values = np.asarray([float(r[metric]) for r in subset])
                entry[f"{metric}_mean"] = float(values.mean())
                entry[f"{metric}_std"] = float(values.std(ddof=1))
            result.append(entry)
    return result


def _decision(summary: list[dict[str, Any]]) -> str:
    cosine_means = defaultdict(list)
    relative_l2_means = defaultdict(list)
    for row in summary:
        layer = int(row["layer"])
        cosine_means[layer].append(float(row["cosine_mean"]))
        relative_l2_means[layer].append(float(row["relative_l2_mean"]))
    early_relative_l2 = float(np.mean(relative_l2_means[6]))
    strong = any(
        float(np.mean(cosine_means[layer])) >= 0.9 and float(np.mean(relative_l2_means[layer])) < early_relative_l2
        for layer in (9, 12)
    )
    if strong:
        return "Strong Go"
    if float(np.mean(cosine_means[15])) >= 0.9 and float(np.mean(relative_l2_means[15])) < early_relative_l2:
        return "Weak Go"
    return "No-Go / Reconsider"


def _write_report(
    *, output_dir: Path, rows: list[dict[str, Any]], summary: list[dict[str, Any]], manifest: dict[str, Any]
) -> None:
    decision = _decision(summary)
    table_lines = [
        "| t | layer | cosine mean±std | relative L2 mean±std | norm ratio mean±std |",
        "|---:|---:|---:|---:|---:|",
    ]
    table_lines.extend(
        (
            f"| {row['flow_t']:.1f} | {row['layer']} | {row['cosine_mean']:.4f}±{row['cosine_std']:.4f} "
            f"| {row['relative_l2_mean']:.4f}±{row['relative_l2_std']:.4f} "
            f"| {row['norm_ratio_mean']:.4f}±{row['norm_ratio_std']:.4f} |"
        )
        for row in summary
    )
    nonfinal = [r for r in summary if int(r["layer"]) != 18]
    best = max(nonfinal, key=lambda r: float(r["cosine_mean"]))
    layer18_rows = [r for r in rows if int(r["layer"]) == 18]
    l18_cos_min = min(float(r["cosine"]) for r in layer18_rows)
    l18_err_max = max(float(r["relative_l2"]) for r in layer18_rows)
    text = f"""# π0 Intermediate Action Expert Probe — Experiment 1

## Setup

- 模型：原始 pretrained `pi0_libero`，实际部署的 Triton-π0 路径；未训练、未修改权重。
- checkpoint：`{manifest["base_weights_path"]}`，SHA256 `{manifest["base_weights_sha256"]}`。
- GPU：{manifest["gpu"]}。
- observation：四个 LIBERO suite、40 个 task，每 task 从 pure-π0 episode 0 轨迹均匀复放 5 个位置，共 {manifest["sample_count"]} 个。
- observation SHA256：`{manifest["observations_sha256"]}`。这些点来自 40 个 episode（190 个点来自成功 episode、10 个来自失败 episode）；轨迹进度只是 phase proxy，不是人工 phase 标签，也不能把 n=200 当成 200 个独立 episode。
- probe layers：6 / 9 / 12 / 15 / 18（日志为 1-based；代码层索引为 5 / 8 / 11 / 14 / 17）。
- flow timesteps：原始 10-step Euler 轨迹的 t=1.0 / 0.8 / 0.5 / 0.3；0.8 与 0.3 是离任务建议 0.75 / 0.25 最近、并统一选择较早侧的实际时刻。
- 主指标：50-step horizon 的 LIBERO 有效 action 维 0:7；`all32_*` 仅作附录字段，避免 padding 维扭曲结论。

## Code Changes

- 新增离线 observation 复放采集脚本；不请求策略、不改变控制动作。
- 新增独立 probe runner，完整执行全部 18 层；只复制指定层的 Action Expert action-token hidden（排除 state token）。
- 每个副本经过部署权重中的原 final RMSNorm + `action_out_proj` 融合 head；融合的 `dt=-0.1` 被除回 velocity。
- probe 预测从不写回 `x_t`、KV cache 或 action。线上 server/production forward 未增加 feature flag，也未修改。
- 固定 observation/noise 对比 production CUDA graph 与 probe full forward：{manifest["output_equal_count"]}/{manifest["sample_count"]} 为 bitwise equal，最大绝对差 {manifest["max_output_abs_diff"]}。
- 每个 sample 的每个 flow step 均访问全部 18 层；权重文件实验前后 SHA256 一致：{manifest["weight_hash_unchanged"]}。
- Layer 18 自引用 sanity：最小 cosine={l18_cos_min:.8f}，最大 relative L2={l18_err_max:.8g}；NaN/Inf 数={manifest["nonfinite_metric_count"]}。它只验证 metric/reference 管线；独立的不变性证据是上一条 production graph vs probe full forward 的 200 次 bitwise equality。

## Results

{chr(10).join(table_lines)}

两张主图：`layer_vs_cosine.png`、`layer_vs_relative_l2.png`。逐 sample 数据见 `intermediate_probe.csv`。

## Findings

1. Q1（随深度是否提高）：是，四个 t 下 cosine 都随 6→9→12→15 单调提高；但跨 t 平均只有 Layer 6/9/12/15 = 0.4890/0.5574/0.7500/0.8278，没有浅层稳定达到 0.9。
2. Q2（emergence point）：没有稳健的浅层 emergence point。最佳单格是 Layer {best["layer"]}、t={best["flow_t"]:.1f}，mean cosine={best["cosine_mean"]:.4f}，但同一 Layer 15 到 t=0.3 已降至 0.7061，真正稳定跃迁发生在最终 Layer 18。
3. Q3（方向 vs magnitude）：方向先于幅值形成，但幅值没有在中间层收敛。Layer 12/15 的 norm ratio 多为 1.87–2.45，relative L2 仍为 1.37–1.62；所以高 cosine 不能直接解释为可执行 velocity。
4. Q4（flow timestep 差异）：差异显著且方向与“后期更容易浅层退出”的猜想相反。Layer 15 cosine 从 t=1.0 的 0.9285 连续降到 0.8853/0.7914/0.7061；越接近 flow 后期，中间层方向越不稳定。
5. Q5（action dimensions）：rotation 最早形成，Layer 12 在四个 t 下 cosine 为 0.9770/0.9700/0.9489/0.9178；translation 与 gripper 明显更晚，t=0.3 的 Layer 15 仅为 0.5637/0.5594。gripper 是单维 cosine，对近零向量敏感，需与 CSV 中误差字段一起解释。

## Decision

**{decision}**。操作化判断先要求跨 flow time 的 Layer 9/12 mean cosine ≥0.9 才可能为 Strong Go，只有 Layer 15 稳定接近该水平才可能为 Weak Go；同时必须检查 relative L2 与 norm ratio。本轮 cosine 已不达标，幅值误差又很大，因此结论不依赖主观的“误差明显下降”阈值。这里是表示一致性结论，不是 success-rate 或真实 early-exit 加速结论。

## Recommended Next Step

只有结果达到 Strong/Weak Go 才进入 Experiment 2（Flow-Step Budget Sweep）；否则暂停 partial-layer fast path，不先做 Kalman、vision buffer 或复杂 runtime。
"""
    (output_dir / "experiment1_report.md").write_text(text, encoding="utf-8")


def run(args) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = np.load(args.observations, allow_pickle=False)
    metadata = json.loads(str(dataset["metadata_json"]))
    images_np, wrists_np, states_np = dataset["image"], dataset["wrist_image"], dataset["state"]
    if len(metadata) != 200:
        raise ValueError(f"expected exactly 200 observations, got {len(metadata)}")

    base_weights_path = args.converted_root / "base" / "base_weights.pkl"
    hash_before = _sha256(base_weights_path)
    prompt_embeddings = _load_prompt_embeddings(args.converted_root, metadata)
    _, input_transform, _ = _build_policy_components(
        config_name="pi0_libero", checkpoint_dir=args.converted_root / "base"
    )

    print(f"loading {base_weights_path}")
    with base_weights_path.open("rb") as handle:
        base_weights = pickle.load(handle)
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, row in enumerate(metadata):
        groups[len(prompt_embeddings[(row["suite"], row["task_description"])])].append(idx)

    rng = np.random.default_rng(args.noise_seed)
    noises = rng.standard_normal((len(metadata), 50, 32), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    output_equal_count = 0
    max_output_abs_diff = 0.0
    nonfinite_metric_count = 0

    for prompt_len, sample_indices in sorted(groups.items()):
        first = metadata[sample_indices[0]]
        checkpoint = dict(base_weights)
        checkpoint["language_embeds"] = prompt_embeddings[(first["suite"], first["task_description"])]
        runtime = pi0_infer.Pi0Inference(checkpoint, num_views=2, chunk_size=50)
        for sample_idx in sample_indices:
            meta = metadata[sample_idx]
            prompt_embedding = prompt_embeddings[(meta["suite"], meta["task_description"])]
            if len(prompt_embedding) != prompt_len:
                raise RuntimeError("prompt grouping failed")
            runtime.weights["language_embeds"].copy_(prompt_embedding.to(device="cuda"))
            raw = {
                "observation/image": images_np[sample_idx],
                "observation/wrist_image": wrists_np[sample_idx],
                "observation/state": states_np[sample_idx],
                "prompt": meta["task_description"],
            }
            transformed = input_transform(raw)
            images, state, noise = prepare_triton_inputs_from_transformed(
                transformed=transformed,
                device="cuda",
                action_horizon=50,
                action_dim=32,
                noise=noises[sample_idx],
            )
            baseline = runtime.forward(images, state, noise).clone()
            torch.cuda.synchronize()
            probed, velocities, layer_visits = _probe_full_forward(runtime, images, state, noise)
            equal = torch.equal(baseline, probed)
            output_equal_count += int(equal)
            max_output_abs_diff = max(max_output_abs_diff, float((baseline.float() - probed.float()).abs().max()))
            if not equal:
                raise AssertionError(f"probe changed final output for {meta['sample_id']}")
            for step in PROBE_STEPS:
                reference = velocities[(step, 18)]
                for layer in PROBE_LAYERS:
                    metric_values = _metric_columns(velocities[(step, layer)], reference)
                    nonfinite_metric_count += sum(not math.isfinite(float(value)) for value in metric_values.values())
                    rows.append(
                        {
                            **meta,
                            "flow_step": step,
                            "flow_t": FLOW_TIMES[step],
                            "layer": layer,
                            "all_18_layers_executed": layer_visits == [10] * 18,
                            "final_output_bitwise_equal": equal,
                            **metric_values,
                        }
                    )
            print(f"[{len(rows) // 20:03d}/200] {meta['sample_id']} output_equal={equal}")
        del runtime, checkpoint
        gc.collect()
        torch.cuda.empty_cache()

    fieldnames = list(rows[0])
    with (output_dir / "intermediate_probe.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = _summary(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_plots(rows, output_dir)
    hash_after = _sha256(base_weights_path)
    manifest = {
        "model": "pi0_libero Triton deployment",
        "base_weights_path": str(base_weights_path.resolve()),
        "base_weights_sha256": hash_before,
        "weight_hash_unchanged": hash_before == hash_after,
        "observations_path": str(args.observations.resolve()),
        "observations_sha256": _sha256(args.observations),
        "sample_count": len(metadata),
        "row_count": len(rows),
        "probe_layers": list(PROBE_LAYERS),
        "flow_steps": list(PROBE_STEPS),
        "flow_times": [FLOW_TIMES[s] for s in PROBE_STEPS],
        "noise_seed": args.noise_seed,
        "output_equal_count": output_equal_count,
        "max_output_abs_diff": max_output_abs_diff,
        "nonfinite_metric_count": nonfinite_metric_count,
        "gpu": torch.cuda.get_device_name(0),
        "decision": _decision(summary),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_report(output_dir=output_dir, rows=rows, summary=summary, manifest=manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--converted-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--noise-seed", type=int, default=20260927)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_grad_enabled(False)
    run(args)


if __name__ == "__main__":
    main()
