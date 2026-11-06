# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
import statistics
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
LAYER_ORDER = ("first3_last3", "first6", "last6")
FLOW_ORDER = ("full10", "first3", "last3")


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    p = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _mcnemar_exact_p(gains: int, losses: int) -> float | None:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(gains, losses) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _safe_relpath(root: Path, value: Any, *, label: str) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{label} must be relative to run_dir, got absolute path {value!r}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{label} escapes run_dir: {value!r}")
    return resolved


def _read_jsonl(path: Path, *, label: str, errors: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{label}: invalid JSON on line {line_number}: {exc}")
                    continue
                if not isinstance(value, dict):
                    errors.append(f"{label}: line {line_number} is not a JSON object")
                    continue
                rows.append(value)
    except OSError as exc:
        errors.append(f"{label}: cannot read file: {exc}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _condition_rows(manifest: dict[str, Any], errors: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    conditions = manifest.get("conditions")
    if not isinstance(conditions, list):
        errors.append("manifest.conditions must be an array")
        return [], []
    if len(conditions) != 10:
        errors.append(f"expected ten conditions, found {len(conditions)}")

    all_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    expected_trials = int(manifest.get("num_trials_per_condition", 50))
    expected_task_id = int(manifest.get("task_id", 5))
    condition_ids: set[str] = set()

    for condition in conditions:
        condition_id = str(condition.get("condition_id", "<missing>"))
        if condition_id in condition_ids:
            errors.append(f"duplicate condition_id: {condition_id}")
        condition_ids.add(condition_id)
        run_dir_value = condition.get("run_dir")
        if not isinstance(run_dir_value, str):
            errors.append(f"{condition_id}: missing run_dir")
            continue
        run_dir = (REPO_ROOT / run_dir_value).resolve()
        if not run_dir.is_relative_to(REPO_ROOT.resolve()):
            errors.append(f"{condition_id}: run_dir escapes repository: {run_dir_value}")
            continue
        client_manifest_path = run_dir / "manifest.json"
        episode_log_path = run_dir / "episode_log.json"
        try:
            client_manifest = json.loads(client_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{condition_id}: missing/invalid client manifest: {exc}")
            client_manifest = {}
        try:
            episode_records = json.loads(episode_log_path.read_text(encoding="utf-8"))
            if not isinstance(episode_records, list):
                raise ValueError("episode_log.json must contain an array")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"{condition_id}: missing/invalid episode_log.json: {exc}")
            episode_records = []

        if client_manifest:
            if client_manifest.get("run_id") != condition.get("run_id"):
                errors.append(f"{condition_id}: client manifest run_id does not match master manifest")
            if client_manifest.get("task_suite_name") != manifest.get("task_suite_name"):
                errors.append(f"{condition_id}: client manifest suite mismatch")
            if client_manifest.get("selected_task_ids") != [expected_task_id]:
                errors.append(f"{condition_id}: client manifest must select only task {expected_task_id}")
            if int(client_manifest.get("num_trials_per_task", -1)) != expected_trials:
                errors.append(f"{condition_id}: client manifest trial count mismatch")

        indexed: dict[int, dict[str, Any]] = {}
        for raw in episode_records:
            if not isinstance(raw, dict):
                errors.append(f"{condition_id}: episode_log contains a non-object record")
                continue
            if int(raw.get("task_id", -1)) != expected_task_id:
                errors.append(f"{condition_id}: episode record has unexpected task_id={raw.get('task_id')}")
                continue
            try:
                episode_idx = int(raw["episode_idx"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{condition_id}: episode record has invalid episode_idx")
                continue
            if episode_idx in indexed:
                errors.append(f"{condition_id}: duplicate episode_idx={episode_idx}")
                continue
            indexed[episode_idx] = raw
        missing = sorted(set(range(expected_trials)) - set(indexed))
        unexpected = sorted(set(indexed) - set(range(expected_trials)))
        if missing:
            errors.append(f"{condition_id}: missing episode indices {missing}")
        if unexpected:
            errors.append(f"{condition_id}: out-of-range episode indices {unexpected}")
        if len(episode_records) != expected_trials:
            errors.append(f"{condition_id}: expected {expected_trials} episode records, found {len(episode_records)}")

        for episode_idx, raw in sorted(indexed.items()):
            if not isinstance(raw.get("success"), bool):
                errors.append(f"{condition_id}/ep{episode_idx:03d}: success must be boolean")
            file_paths: dict[str, Path | None] = {}
            for field in ("video_path", "trace_path", "infer_path"):
                try:
                    file_paths[field] = _safe_relpath(run_dir, raw.get(field), label=f"{condition_id}/ep{episode_idx}/{field}")
                except ValueError as exc:
                    errors.append(str(exc))
                    file_paths[field] = None
                if file_paths[field] is None:
                    errors.append(f"{condition_id}/ep{episode_idx:03d}: missing {field}")
                elif not file_paths[field].is_file() or file_paths[field].stat().st_size <= 0:
                    errors.append(f"{condition_id}/ep{episode_idx:03d}: {field} file is missing or empty")

            trace_rows = []
            infer_rows = []
            if file_paths["trace_path"] and file_paths["trace_path"].is_file():
                trace_rows = _read_jsonl(
                    file_paths["trace_path"],
                    label=f"{condition_id}/ep{episode_idx:03d}/trace.jsonl",
                    errors=errors,
                )
            if file_paths["infer_path"] and file_paths["infer_path"].is_file():
                infer_rows = _read_jsonl(
                    file_paths["infer_path"],
                    label=f"{condition_id}/ep{episode_idx:03d}/infer.jsonl",
                    errors=errors,
                )
            if len(trace_rows) != int(raw.get("trace_record_count", -1)):
                errors.append(f"{condition_id}/ep{episode_idx:03d}: trace record count mismatch")
            if len(infer_rows) != int(raw.get("infer_record_count", -1)):
                errors.append(f"{condition_id}/ep{episode_idx:03d}: infer record count mismatch")
            if int(raw.get("infer_calls", -1)) != len(infer_rows):
                errors.append(f"{condition_id}/ep{episode_idx:03d}: infer_calls does not equal infer.jsonl rows")
            executed = sum(1 for row in trace_rows if row.get("executed_action") is not None)
            if int(raw.get("executed_action_count", -1)) != executed:
                errors.append(f"{condition_id}/ep{episode_idx:03d}: executed_action_count does not match trace")
            if int(raw.get("trajectory_length", -1)) != executed:
                errors.append(f"{condition_id}/ep{episode_idx:03d}: trajectory_length does not match trace actions")

            record = {
                "condition_id": condition_id,
                "layer_pattern": condition.get("layer_pattern"),
                "flow_pattern": condition.get("flow_pattern"),
                "is_baseline": bool(condition.get("is_baseline", False)),
                "run_id": condition.get("run_id"),
                "task_id": expected_task_id,
                "task_description": raw.get("task_description", manifest.get("task_description")),
                "episode_idx": episode_idx,
                "success": raw.get("success"),
                "failure_reason": raw.get("failure_reason"),
                "trajectory_length": raw.get("trajectory_length"),
                "executed_action_count": raw.get("executed_action_count"),
                "infer_calls": raw.get("infer_calls"),
                "infer_latency_mean_ms": _finite_float(raw.get("infer_latency_mean_ms")),
                "infer_latency_sum_ms": _finite_float(raw.get("infer_latency_sum_ms")),
                "avg_latency_per_action_ms": _finite_float(raw.get("avg_latency_per_action_ms")),
                "policy_time_mean_ms": _finite_float(raw.get("policy_time_mean_ms")),
                "serve_time_mean_ms": _finite_float(raw.get("serve_time_mean_ms")),
                "video_path": str(file_paths["video_path"].relative_to(REPO_ROOT)) if file_paths["video_path"] else None,
                "trace_path": str(file_paths["trace_path"].relative_to(REPO_ROOT)) if file_paths["trace_path"] else None,
                "infer_path": str(file_paths["infer_path"].relative_to(REPO_ROOT)) if file_paths["infer_path"] else None,
            }
            all_rows.append(record)
            video_rows.append(
                {
                    "condition_id": condition_id,
                    "layer_pattern": record["layer_pattern"],
                    "flow_pattern": record["flow_pattern"],
                    "episode_idx": episode_idx,
                    "success": record["success"],
                    "video_path": record["video_path"],
                }
            )

    return all_rows, video_rows


def _condition_summary(condition: dict[str, Any], rows: list[dict[str, Any]], expected_trials: int) -> dict[str, Any]:
    valid_rows = [row for row in rows if isinstance(row.get("success"), bool)]
    successes = sum(bool(row["success"]) for row in valid_rows)
    total = len(valid_rows)
    lower, upper = _wilson_interval(successes, total)
    per_episode_latency = [
        float(row["infer_latency_mean_ms"])
        for row in valid_rows
        if row.get("infer_latency_mean_ms") is not None
    ]
    latency_sum = sum(
        float(row["infer_latency_sum_ms"])
        for row in valid_rows
        if row.get("infer_latency_sum_ms") is not None
    )
    infer_count = sum(int(row.get("infer_calls") or 0) for row in valid_rows)
    per_action = [
        float(row["avg_latency_per_action_ms"])
        for row in valid_rows
        if row.get("avg_latency_per_action_ms") is not None
    ]
    return {
        "condition_id": condition["condition_id"],
        "layer_pattern": condition["layer_pattern"],
        "flow_pattern": condition["flow_pattern"],
        "is_baseline": bool(condition.get("is_baseline", False)),
        "run_id": condition.get("run_id"),
        "run_dir": condition.get("run_dir"),
        "status": condition.get("status"),
        "episode_count": total,
        "expected_episode_count": expected_trials,
        "successes": successes,
        "failures": total - successes,
        "success_rate": successes / total if total else None,
        "success_rate_percent": 100.0 * successes / total if total else None,
        "wilson_95_percent": [100.0 * lower, 100.0 * upper] if lower is not None else None,
        "infer_call_count": infer_count,
        "infer_latency_weighted_mean_ms": latency_sum / infer_count if infer_count else None,
        "episode_infer_latency_mean_ms": statistics.mean(per_episode_latency) if per_episode_latency else None,
        "episode_infer_latency_median_ms": _percentile(per_episode_latency, 0.5),
        "episode_infer_latency_p95_ms": _percentile(per_episode_latency, 0.95),
        "mean_latency_per_action_ms": statistics.mean(per_action) if per_action else None,
    }


def _paired_summary(
    summaries: list[dict[str, Any]], records_by_condition: dict[str, dict[int, dict[str, Any]]], baseline_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline = records_by_condition.get(baseline_id, {})
    paired_rows: list[dict[str, Any]] = []
    paired_summaries: list[dict[str, Any]] = []
    for summary in summaries:
        candidate_id = str(summary["condition_id"])
        if candidate_id == baseline_id:
            continue
        candidate = records_by_condition.get(candidate_id, {})
        gains = losses = same_success = same_failure = 0
        latency_deltas: list[float] = []
        for episode_idx, base_record in sorted(baseline.items()):
            candidate_record = candidate.get(episode_idx)
            if candidate_record is None:
                continue
            base_success = bool(base_record["success"])
            candidate_success = bool(candidate_record["success"])
            if candidate_success and not base_success:
                gains += 1
                discordance = "candidate_only_success"
            elif base_success and not candidate_success:
                losses += 1
                discordance = "baseline_only_success"
            elif base_success:
                same_success += 1
                discordance = "both_success"
            else:
                same_failure += 1
                discordance = "both_failure"
            base_latency = _finite_float(base_record.get("infer_latency_mean_ms"))
            candidate_latency = _finite_float(candidate_record.get("infer_latency_mean_ms"))
            latency_delta = (
                candidate_latency - base_latency
                if candidate_latency is not None and base_latency is not None
                else None
            )
            if latency_delta is not None:
                latency_deltas.append(latency_delta)
            paired_rows.append(
                {
                    "condition_id": candidate_id,
                    "baseline_condition_id": baseline_id,
                    "episode_idx": episode_idx,
                    "baseline_success": base_success,
                    "candidate_success": candidate_success,
                    "discordance": discordance,
                    "baseline_infer_latency_mean_ms": base_latency,
                    "candidate_infer_latency_mean_ms": candidate_latency,
                    "latency_delta_ms": latency_delta,
                }
            )
        paired_count = same_success + same_failure + gains + losses
        paired_summaries.append(
            {
                "condition_id": candidate_id,
                "baseline_condition_id": baseline_id,
                "paired_episode_count": paired_count,
                "both_success": same_success,
                "both_failure": same_failure,
                "candidate_only_success": gains,
                "baseline_only_success": losses,
                "success_rate_difference_percentage_points": (
                    100.0 * (gains - losses) / paired_count if paired_count else None
                ),
                "mcnemar_exact_p_value": _mcnemar_exact_p(gains, losses),
                "mean_paired_latency_delta_ms": statistics.mean(latency_deltas) if latency_deltas else None,
            }
        )
    return paired_rows, paired_summaries


def _heatmap(path: Path, title: str, values: dict[tuple[str, str], float | None], counts: dict[tuple[str, str], int], *, success: bool) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = np.array(
        [[values.get((layer, flow), np.nan) for flow in FLOW_ORDER] for layer in LAYER_ORDER],
        dtype=float,
    )
    masked = np.ma.masked_invalid(matrix)
    fig, ax = plt.subplots(figsize=(8.0, 4.3), constrained_layout=True)
    cmap = plt.get_cmap("YlGnBu" if success else "magma").copy()
    cmap.set_bad(color="#d9d9d9")
    if success:
        image = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        colorbar_label = "Success rate (%)"
    else:
        finite = matrix[np.isfinite(matrix)]
        maximum = float(finite.max()) if finite.size else 1.0
        image = ax.imshow(masked, cmap=cmap, vmin=0, vmax=max(maximum, 1e-6), aspect="auto")
        colorbar_label = "Mean inference latency per call (ms)"
    ax.set_xticks(range(len(FLOW_ORDER)), FLOW_ORDER)
    ax.set_yticks(range(len(LAYER_ORDER)), LAYER_ORDER)
    ax.set_xlabel("Denoising pattern")
    ax.set_ylabel("Action Expert layer pattern")
    ax.set_title(title)
    for row_index, layer in enumerate(LAYER_ORDER):
        for column_index, flow in enumerate(FLOW_ORDER):
            value = values.get((layer, flow))
            count = counts.get((layer, flow), 0)
            label = "N/A" if value is None else (f"{value:.1f}%" if success else f"{value:.2f} ms")
            ax.text(column_index, row_index, f"{label}\n(n={count})", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _fmt_percent(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.1f}%"


def _fmt_ms(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.2f} ms"


def _report(manifest: dict[str, Any], summaries: list[dict[str, Any]], paired: list[dict[str, Any]], errors: list[str]) -> str:
    expected = int(manifest.get("num_trials_per_condition", 50))
    lines = [
        "# π0 Layer × Denoising 消融实验",
        "",
        f"- 实验 ID：`{manifest.get('experiment_id', 'unknown')}`",
        f"- 状态：`{manifest.get('status', 'unknown')}`；task：LIBERO-Goal {manifest.get('task_id', 5)}，每条件 {expected} 集。",
        f"- 对照：`full18 × full10`；噪声种子：`{manifest.get('ablation_noise_seed', 'unknown')}`；环境 seed：`{manifest.get('environment_seed', 'unknown')}`。",
        "- 成功判据：LIBERO 原生任务完成条件。各 rollout 视频、trace 与 infer 文件按 `manifest.json`/`episode_log.json` 路径索引。",
        "",
        "## 配置结果",
        "",
        "| Layers | Denoising | 成功数/集数 | 成功率（Wilson 95% CI） | 平均推理延迟/调用 | episode 延迟 P50 / P95 | 每动作延迟 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        ci = item.get("wilson_95_percent")
        ci_text = "N/A" if ci is None else f"[{ci[0]:.1f}%, {ci[1]:.1f}%]"
        lines.append(
            "| {layer} | {flow} | {success}/{count} | {rate} {ci} | {latency} | {p50} / {p95} | {per_action} |".format(
                layer=item["layer_pattern"],
                flow=item["flow_pattern"],
                success=item["successes"],
                count=item["episode_count"],
                rate=_fmt_percent(item["success_rate_percent"]),
                ci=ci_text,
                latency=_fmt_ms(item["infer_latency_weighted_mean_ms"]),
                p50=_fmt_ms(item["episode_infer_latency_median_ms"]),
                p95=_fmt_ms(item["episode_infer_latency_p95_ms"]),
                per_action=_fmt_ms(item["mean_latency_per_action_ms"]),
            )
        )
    lines.extend(
        [
            "",
            "## 配对基线比较",
            "",
            "| Layers | Denoising | 成对集数 | 同成功 | 同失败 | 消融独有成功 | 基线独有成功 | Δ成功率 | McNemar exact p | 配对延迟变化 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    paired_index = {row["condition_id"]: row for row in paired}
    for item in summaries:
        if item["is_baseline"]:
            continue
        compare = paired_index.get(item["condition_id"], {})
        delta = compare.get("success_rate_difference_percentage_points")
        p_value = compare.get("mcnemar_exact_p_value")
        lines.append(
            "| {layer} | {flow} | {n} | {same_s} | {same_f} | {gain} | {loss} | {delta} | {p} | {latency} |".format(
                layer=item["layer_pattern"],
                flow=item["flow_pattern"],
                n=compare.get("paired_episode_count", 0),
                same_s=compare.get("both_success", 0),
                same_f=compare.get("both_failure", 0),
                gain=compare.get("candidate_only_success", 0),
                loss=compare.get("baseline_only_success", 0),
                delta="N/A" if delta is None else f"{float(delta):+.1f} pp",
                p="N/A" if p_value is None else f"{float(p_value):.4f}",
                latency=_fmt_ms(compare.get("mean_paired_latency_delta_ms")),
            )
        )
    lines.extend(
        [
            "",
            "## 热图与数据",
            "",
            "- 成功率：`success_rate_heatmap.png`。",
            "- 推理延迟：`latency_heatmap.png`；使用客户端 episode_log 中 sample-actions latency 的加权每次调用均值。",
            "- 全部逐集结果：`episode_results.csv`；与控制组按相同 `episode_idx` 配对：`paired_vs_baseline.csv`。",
            "- Rollout 视频索引：`video_index.csv`；每集视频及 `trace.jsonl`、`infer.jsonl` 保存在 `data/test/` 对应 run 目录。",
            "",
            "## 完整性检查",
            "",
        ]
    )
    if errors:
        lines.append(f"未通过：发现 {len(errors)} 个问题。")
        lines.extend(f"- {error}" for error in errors[:100])
        if len(errors) > 100:
            lines.append(f"- 另有 {len(errors) - 100} 个问题未在此处展开，完整列表见 `manifest.json`。")
    else:
        lines.append(f"通过：10 个条件各有 {expected} 个唯一 episode；视频、trace、infer 文件及其计数一致。")
    lines.append("")
    return "\n".join(lines)


def summarize_experiment(manifest_path: Path, output_dir: Path) -> list[str]:
    manifest_path = manifest_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    rows, video_rows = _condition_rows(manifest, errors)
    expected_trials = int(manifest.get("num_trials_per_condition", 50))
    conditions = manifest.get("conditions", [])

    rows_by_condition: dict[str, list[dict[str, Any]]] = {}
    raw_by_condition: dict[str, dict[int, dict[str, Any]]] = {}
    for row in rows:
        rows_by_condition.setdefault(str(row["condition_id"]), []).append(row)
    for condition in conditions:
        condition_id = str(condition.get("condition_id", "<missing>"))
        run_dir = REPO_ROOT / str(condition.get("run_dir", ""))
        try:
            episode_records = json.loads((run_dir / "episode_log.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            episode_records = []
        raw_by_condition[condition_id] = {
            int(record["episode_idx"]): record
            for record in episode_records
            if isinstance(record, dict) and "episode_idx" in record
        }

    summaries = [
        _condition_summary(condition, rows_by_condition.get(str(condition["condition_id"]), []), expected_trials)
        for condition in conditions
    ]
    baseline_id = str(manifest.get("baseline_condition_id", "full18__full10"))
    if sum(bool(condition.get("is_baseline")) for condition in conditions) != 1:
        errors.append("manifest must mark exactly one baseline condition")
    if baseline_id not in {str(condition.get("condition_id")) for condition in conditions}:
        errors.append(f"baseline condition {baseline_id!r} is absent")
    paired_rows, paired_summaries = _paired_summary(summaries, raw_by_condition, baseline_id)
    if len(paired_rows) != 9 * expected_trials:
        errors.append(f"expected {9 * expected_trials} paired rows, found {len(paired_rows)}")

    episode_fields = [
        "condition_id", "layer_pattern", "flow_pattern", "is_baseline", "run_id", "task_id",
        "task_description", "episode_idx", "success", "failure_reason", "trajectory_length",
        "executed_action_count", "infer_calls", "infer_latency_mean_ms", "infer_latency_sum_ms",
        "avg_latency_per_action_ms", "policy_time_mean_ms", "serve_time_mean_ms", "video_path",
        "trace_path", "infer_path",
    ]
    paired_fields = [
        "condition_id", "baseline_condition_id", "episode_idx", "baseline_success", "candidate_success",
        "discordance", "baseline_infer_latency_mean_ms", "candidate_infer_latency_mean_ms", "latency_delta_ms",
    ]
    video_fields = ["condition_id", "layer_pattern", "flow_pattern", "episode_idx", "success", "video_path"]
    _write_csv(output_dir / "episode_results.csv", rows, episode_fields)
    _write_csv(output_dir / "paired_vs_baseline.csv", paired_rows, paired_fields)
    _write_csv(output_dir / "video_index.csv", video_rows, video_fields)

    success_values: dict[tuple[str, str], float | None] = {}
    latency_values: dict[tuple[str, str], float | None] = {}
    counts: dict[tuple[str, str], int] = {}
    for item in summaries:
        if item["is_baseline"]:
            continue
        key = (str(item["layer_pattern"]), str(item["flow_pattern"]))
        success_values[key] = item["success_rate_percent"]
        latency_values[key] = item["infer_latency_weighted_mean_ms"]
        counts[key] = int(item["episode_count"])
    try:
        _heatmap(output_dir / "success_rate_heatmap.png", "LIBERO-Goal Task 5 Success Rate", success_values, counts, success=True)
        _heatmap(output_dir / "latency_heatmap.png", "Mean Inference Latency by Ablation", latency_values, counts, success=False)
    except ImportError as exc:
        errors.append(f"heatmap dependency unavailable: {exc}")

    if errors and manifest.get("status") == "completed":
        manifest["status"] = "incomplete"
    summary_payload = {
        "experiment_id": manifest.get("experiment_id"),
        "generated_at": _utc_now(),
        "status": manifest.get("status"),
        "task_suite_name": manifest.get("task_suite_name"),
        "task_id": manifest.get("task_id"),
        "expected_trials_per_condition": expected_trials,
        "baseline_condition_id": baseline_id,
        "condition_count": len(conditions),
        "episode_record_count": len(rows),
        "paired_record_count": len(paired_rows),
        "integrity_passed": not errors,
        "integrity_errors": errors,
        "conditions": summaries,
        "paired_vs_baseline": paired_summaries,
    }
    manifest["analysis"] = {
        "generated_at": summary_payload["generated_at"],
        "integrity_passed": not errors,
        "integrity_errors": errors,
        "episode_record_count": len(rows),
        "paired_record_count": len(paired_rows),
        "summary_file": "summary.json",
        "report_file": "experiment_report.md",
    }
    _atomic_json(output_dir / "summary.json", summary_payload)
    _atomic_json(output_dir / "manifest.json", manifest)
    _atomic_text(output_dir / "experiment_report.md", _report(manifest, summaries, paired_summaries, errors))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and summarize a π0 layer × denoising rollout sweep.")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "results/pi0_layer_denoise_ablation/manifest.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/pi0_layer_denoise_ablation")
    args = parser.parse_args()
    errors = summarize_experiment(args.manifest, args.output_dir)
    if errors:
        print(f"Artifact integrity check failed with {len(errors)} issue(s):")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Artifacts validated and summarized in {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
