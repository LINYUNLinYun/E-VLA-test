"""Audit existing Experiment 9 artifacts without running policies or the full analyzer."""
from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/experiment9_standard_vlm_pruning_20261004"


def edge_rows(path):
    with path.open("rb") as stream:
        first = json.loads(stream.readline())
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 65536))
        last = json.loads(stream.read().splitlines()[-1])
    return first, last


def main():
    config = json.loads((ROOT / "scripts/experiment9/experiment9_config.json").read_text())
    conditions = list(config["conditions"]) + ["AE-D7", "AE-D5", "AE-D3"]
    suites = config["suites"]
    selected, missing, partial, duplicates = {}, [], [], []
    for condition in conditions:
        for suite in suites:
            for start in range(0, 50, 10):
                expected = {(t, i) for t in range(10) for i in range(start, start + 10)}
                good = []
                for path in sorted(RESULTS.glob(f"exp9_*_{condition.replace('-', '_')}_{suite}_i{start:02d}_{start+10:02d}_a*"), reverse=True):
                    try:
                        rows = json.loads((path / "episode_log.json").read_text())
                        manifest = json.loads((path / "manifest.json").read_text())
                        keys = [(r["task_id"], r["episode_idx"]) for r in rows]
                        valid = len(keys) == 100 and set(keys) == expected and manifest["episode_index_start"] == start and manifest["episode_index_end"] == start + 10
                    except (OSError, ValueError, KeyError):
                        rows, valid = [], False
                    if valid:
                        good.append((path, rows))
                    else:
                        partial.append({"run_dir": path.name, "episodes": len(rows)})
                if len(good) > 1:
                    outcomes = [{(r["task_id"], r["episode_idx"]): r["success"] for r in rows} for _, rows in good]
                    duplicates.append({"runs": [p.name for p, _ in good], "identical_success_labels": all(o == outcomes[0] for o in outcomes), "selected": good[0][0].name})
                if good:
                    selected[condition, suite, start] = good[0]
                else:
                    missing.append({"condition": condition, "suite": suite, "start": start, "end": start + 10, "episodes": 100})

    issues, checked, metadata_issues, noise = [], 0, [], {}
    timing_fields = Counter()
    video_samples = []
    summaries = []
    for condition in conditions:
        for suite in suites:
            rows = []
            for start in range(0, 50, 10):
                if (condition, suite, start) not in selected:
                    continue
                path, chunk = selected[condition, suite, start]
                for row in chunk:
                    ident = (condition, suite, row["task_id"], row["episode_idx"])
                    checked += 1
                    if checked % 2000 == 0:
                        print(f"Audited {checked} episodes", flush=True)
                    for field in ("telemetry_path", "infer_path", "trace_path", "video_path"):
                        asset = path / row.get(field, "_missing_")
                        if not asset.is_file() or asset.stat().st_size == 0:
                            issues.append({"episode": ident, "issue": "missing_or_empty", "field": field})
                    try:
                        initial, final = edge_rows(path / row["telemetry_path"])
                        expected_n = int(row["executed_action_count"]) + 1
                        if row["frame_count"] != expected_n or initial["step"] != 0 or final["step"] != expected_n - 1:
                            issues.append({"episode": ident, "issue": "endpoint_or_declared_alignment"})
                        for state in (initial, final):
                            if len(state.get("eef_pos", [])) != 3 or not all(math.isfinite(v) for v in state["eef_pos"]):
                                issues.append({"episode": ident, "issue": "eef_xyz"})
                        first, last = edge_rows(path / row["infer_path"])
                        for inference in (first, last):
                            meta = inference.get("server_experiment", {})
                            denoise = int(condition[4:]) if condition.startswith("AE-D") else 10
                            pattern = config["conditions"].get(condition, "vlm_full18")
                            if meta.get("vlm_layer_pattern") != pattern or meta.get("denoise_steps") != denoise or meta.get("action_layer_pattern") != "full18" or meta.get("ablation_noise_seed") != 12345 or not meta.get("noise_sha256"):
                                metadata_issues.append({"episode": ident, "metadata": meta})
                        noise[ident] = first["server_experiment"].get("noise_sha256")
                        for field in ("action_denoise_ms", "full_fallback_ms", "vlm_prefill_ms", "policy_time_ms"):
                            timing_fields[field] += first.get(field) is not None
                    except (OSError, ValueError, KeyError, IndexError) as exc:
                        issues.append({"episode": ident, "issue": str(exc)})
                    rows.append(row)
                if start == next(s for s in range(0,50,10) if (condition,suite,s) in selected):
                    video_samples.append((condition, suite, path / chunk[0]["video_path"], chunk[0]["frame_count"]))
            baseline = { (r["task_id"],r["episode_idx"]): r for st in range(0,50,10) for r in selected.get(("Full18",suite,st),(None,[]))[1] }
            paired = [(r, baseline[r["task_id"],r["episode_idx"]]) for r in rows]
            successes = sum(bool(r["success"]) for r in rows)
            base_successes = sum(bool(b["success"]) for _,b in paired)
            summaries.append({"condition":condition,"suite":suite,"successes":successes,"episodes":len(rows),"success_rate":successes/len(rows) if rows else None,"paired_baseline_successes":base_successes,"paired_delta_pp":100*(successes-base_successes)/len(rows) if rows else None,"new_failures":sum(bool(b["success"]) and not r["success"] for r,b in paired),"recoveries":sum(bool(r["success"]) and not b["success"] for r,b in paired),"policy_episode_mean_p50_ms":statistics.median(r["policy_time_mean_ms"] for r in rows) if rows else None,"failure_stage_counts":dict(Counter(r.get("furthest_semantic_phase","unknown") for r in rows if not r["success"]))})
    noise_mismatches = [key for key, value in noise.items() if key[0] != "Full18" and value != noise.get(("Full18",*key[1:]))]
    probes, decodes = [], []
    for condition, suite, path, expected_frames in video_samples:
        result = subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=codec_name,width,height,nb_frames,avg_frame_rate","-of","json",str(path)],capture_output=True,text=True)
        stream = json.loads(result.stdout).get("streams", [{}])[0] if result.returncode == 0 else {}
        probes.append({"condition":condition,"suite":suite,"path":str(path),"stream":stream,"expected_frames":expected_frames,"ok":result.returncode==0 and stream.get("codec_name")=="h264" and int(stream.get("nb_frames",-1))==expected_frames})
        if condition in ("Full18","V13-Custom"):
            decoded = subprocess.run(["ffmpeg","-v","error","-i",str(path),"-f","null","-"],capture_output=True,text=True)
            decodes.append({"path":str(path),"ok":decoded.returncode==0 and not decoded.stderr.strip(),"error":decoded.stderr[:1000]})
    output = RESULTS / "audit"
    output.mkdir(exist_ok=True)
    audit = {"created_at_utc":datetime.now(timezone.utc).isoformat(),"expected_episodes":len(conditions)*len(suites)*500,"selected_episodes":checked,"selected_chunks":len(selected),"missing_chunks":missing,"excluded_partial_attempts":partial,"duplicate_complete_attempts":duplicates,"asset_issues":issues,"metadata_endpoint_issues":metadata_issues,"first_infer_noise_mismatches":noise_mismatches,"first_infer_nonnull_timing_counts":dict(timing_fields),"video_header_samples":probes,"video_decode_samples":decodes,"suite_results":summaries,"selected_runs":[str(path) for path,_ in selected.values()],"scope":"Every selected episode: four asset paths/nonzero sizes, telemetry and inference endpoints, declared N+1 alignment, finite endpoint XYZ, metadata at first/last inference, paired first-inference noise hash. Full line counts and mid-rollout metadata are not reread; video headers sampled once per condition/suite and Full18/V13 samples fully decoded."}
    followup_path=RESULTS/'action_expert_layer_probe/pipeline_status.json'
    audit['action_expert_followup_status']=json.loads(followup_path.read_text()) if followup_path.is_file() else {'stage':'not_started'}
    (output/"round_completion.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n")
    for name, data in (("suite_success.csv",summaries),("missing_chunks.csv",missing)):
        with (output/name).open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]) if data else ["condition","suite","start","end","episodes"])
            writer.writeheader(); writer.writerows(data)
    lines = ["# Experiment 9 结果与完成情况核查", "", f"核查时间（UTC）：{audit['created_at_utc']}。目标 {audit['expected_episodes']:,} 条，已选入 {checked:,} 条唯一 condition/task/init 记录。", "", "按原始 `episode_log.json` 与 manifest 的任务/状态键核查；每任务应覆盖官方初始状态 0–49。重试取最新完整 attempt，与 runner 一致；不将不完整或重复 attempt 加入分母。", "", "## 四套件成功率", "", "除标注外，每格为 500 条（10 任务 × 50 状态）。VLM 变体固定完整 Action Expert / D10；AE-D7/D5/D3 固定完整 VLM / 完整 Action Expert 层，只改变降噪步数。", "", "| 变体 | Object | Spatial | Goal | LIBERO-10 | 总记录数 |", "|---|---:|---:|---:|---:|---:|"]
    for condition in conditions:
        group = [r for r in summaries if r['condition'] == condition]
        cells = [f"{r['successes']}/{r['episodes']} ({100*r['success_rate']:.2f}%)" for r in group]
        lines.append(f"| {condition} | " + " | ".join(cells) + f" | {sum(r['episodes'] for r in group)} |")
    lines.extend(["", "## 缺失与补跑", ""])
    if missing:
        lines.append("发现 V13-Custom 未覆盖初始状态 0–9。原 `run_custom_vlm_variant.py` 只将 10–49 入队，之前的完成通知仅覆盖 16 个已排队分块，不能视为完整 0–49 评测。")
        lines.append("")
        lines.extend(f"- {m['condition']} / {m['suite']} / [{m['start']}, {m['end']})：缺 {m['episodes']} 条。" for m in missing)
        lines.extend(["", "已于 2026-10-05 12:36 UTC 启动四卡补跑；状态见 `backfill_v13_monitor.log`。补跑完成后本核查脚本自动更新统计；未完成批次不会计入结果。"])
    else:
        lines.append("成功率评测矩阵已覆盖所有 14 个条件、四套件、每任务 0–49 状态；无缺失分块。")
    lines.extend(["", "## 原始数据核查", "", f"- 已选中 {len(selected)} 个完整分块；逐条检查视频、trace、infer、telemetry 路径及非空大小，异常 {len(issues)} 条。", f"- 每条轨迹初始/终末步、声明的 N+1 视频/状态对齐、末端 XYZ 有限值及首尾推理配置检查；配置异常 {len(metadata_issues)} 条。", f"- 与 baseline 同任务、同状态的首次推理噪声 hash：不一致 {len(noise_mismatches)} 条。", f"- 逐 condition/suite 抽查视频头部 {len(probes)} 个，H.264 与预期帧数通过 {sum(p['ok'] for p in probes)} 个；baseline/V13 视频全帧解码抽查 {len(decodes)} 个，通过 {sum(p['ok'] for p in decodes)} 个。", "- 此次未重新解析全部轨迹中间行、未解码所有视频；不能据抽查声称每一个 MP4 都已人工观看。", f"- 不完整 attempt {len(partial)} 个，另有完整重复 attempt {len(duplicates)} 组，均不重复计数。"])
    lines.extend(f"  - 未完成（可能正在运行）：{p['run_dir']}，{p['episodes']} 条。" for p in partial)
    lines.extend(f"  - 重复：{d['runs']}；成功标签一致={d['identical_success_labels']}；选中 {d['selected']}。" for d in duplicates)
    lines.extend(["", "## 尚需完成的分析与探针", "", "1. Action Expert 结构探针：本轮 D7/D5/D3 标准闭环评测已完成，但尚无本轮 baseline 的分阶段、全 18 层 hidden 相似度和固定噪声单层删除动作敏感性结果。历史 Experiment 1 有 200 个真实观测的中间层 action-head 分析，但不等价于本轮探针。补充管线将复放本轮 baseline 的 40 条官方 init0 轨迹，按真实 initial/contact/goal-alignment 阶段采集；不可达阶段显式列缺失。", "2. Action Expert 单层闭环筛选是探针之后的条件任务：只有低动作敏感性候选通过事先记录的筛选条件，才追加最多两个候选；不能直接凭 cosine 高认定可删。", "3. 最新全量分阶段失败分析、任务级结果、配对差异与区间尚未生成；原始 telemetry 和终末阶段字段均已保存。", "4. 目前只有 `analysis/pilot`，含 140 张 XYZ 图和 140 个配对视频；全量及新增 V13、后续 AE 的配对图/视频仍待生成。保留用户先跑完记录数据、暂缓正式分析报告的要求。", "5. `experiment9_standard_vlm_pruning_report.md` 是旧初筛报告（各格 100 条），不代表本表全量结果。报告里的 baseline 378/400 是成功数而非完成条数，标题表述需要修正。", "6. 专门的 `action_denoise_ms` 字段在现有日志首条中均为空；`full_fallback_ms` 已保存。代码证据：全推理轮把 staged `decoder_ms` 传入 `_populate_full_round_timing` 并保存为 `full_fallback_ms`。本轮 force-full 协议下，可明确注明来源后从该字段恢复解码时延，不需要仅为字段名重跑 28,000 条。", "7. 旧 21% Control 的具体单一根因仍未严格隔离；任务分布/初始状态/实现差异不能混为因果结论。本轮 Full18 已恢复标准高成功率，但不等于完成旧协议根因消融。", "", "## 当前数据能支持的判断", "", "- 单独跳过 L2/L3/L4/L5：Object 与 Spatial 保持 95% 以上；Goal 和 LIBERO-10 存在不同程度下降，不能只引用简单套件。", "- 联合跳过 L2–L5 明显失效，单层安全性不能直接组合。V12/Object 已完成 500 条且 0 成功，不支持其仍有 90% 成功率的猜测。", "- AE 降噪步数减少目前比批量 VLM 删除更有希望：D5/D3 在四套件的观测成功率接近 baseline；是否能稳定提升需要配对区间与时延分析，不能从点估计宣称显著更好。", "- V13 目前若为 400 条，只能与 baseline 相同 10–49 状态配对；不要直接与 500 条 baseline 相减。配对结果在 CSV 的 `paired_delta_pp`。", "", "## 文件", "", "- 原始根目录：`results/experiment9_standard_vlm_pruning_20261004/`", "- 结构化核查：`audit/round_completion.json`", "- 套件成功率与配对计数：`audit/suite_success.csv`", "- 缺失分块：`audit/missing_chunks.csv`", "- 可复跑核查：`.venv/bin/python scripts/experiment9/audit_round_completion.py`", ""])
    status=audit['action_expert_followup_status']
    lines.extend(["", "## 补充实验实时状态", "", f"Action Expert 补充管线：`{status['stage']}`。状态源：`action_expert_layer_probe/pipeline_status.json`。", ""])
    probe_summary=RESULTS/'action_expert_layer_probe/probe/summary.json'
    if probe_summary.is_file():
        probe=json.loads(probe_summary.read_text())
        if probe.get('complete'):
            lines=[line.replace("但尚无本轮 baseline 的分阶段、全 18 层 hidden 相似度和固定噪声单层删除动作敏感性结果", "并已补齐本轮 baseline 的分阶段、全 18 层 hidden 相似度和固定噪声单层删除动作敏感性结果").replace("补充管线将复放本轮 baseline 的 40 条官方 init0 轨迹", "补充管线已复放本轮 baseline 的 40 条官方 init0 轨迹") for line in lines]
            lines.extend([f"- 探针已完成：{probe['task_count']} 任务、{probe['sample_count']} 个真实观测、三组噪声。", f"- 闭环候选：{probe['closed_loop_candidates']}；判定：{probe['decision']}。", "- 数据：`action_expert_layer_probe/probe/summary.json`、`layer_hidden_similarity.csv`、`layer_action_sensitivity.csv`。", ""])
    (ROOT/"experiment9_completion_audit.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"episodes":checked,"missing":missing,"asset_issues":len(issues),"metadata_issues":len(metadata_issues),"noise_mismatches":len(noise_mismatches),"video_header_passed":sum(p["ok"] for p in probes),"video_header_checked":len(probes),"video_decode_passed":sum(p["ok"] for p in decodes),"video_decode_checked":len(decodes),"timing_fields":dict(timing_fields)},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
