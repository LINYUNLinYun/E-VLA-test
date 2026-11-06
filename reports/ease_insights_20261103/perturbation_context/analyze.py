#!/usr/bin/env python3
"""Descriptive paired V14 vs matching pure-draft analysis; CPU standard library only."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import random
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "reports/idea1_libero_plus_half_final_20261022T2124Z"
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
CATEGORIES = ["Objects Layout", "Camera Viewpoints", "Robot Initial States", "Language Instructions",
              "Light Conditions", "Background Textures", "Sensor Noise"]
LABELS = {"all": "全部扰动", "libero_spatial": "母Spatial", "libero_object": "母Object",
          "libero_goal": "母Goal", "libero_10": "母Long", "Objects Layout": "物体布局",
          "Camera Viewpoints": "相机视角", "Robot Initial States": "机器人初态",
          "Language Instructions": "语言指令", "Light Conditions": "光照",
          "Background Textures": "背景纹理", "Sensor Noise": "传感器噪声"}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path, evidence):
    raw = path.read_bytes()
    evidence.append(dict(path=str(path.resolve()), sha256=digest(raw), bytes=len(raw)))
    return json.loads(raw)


def percentile(values, q):
    ordered = sorted(values)
    h = (len(ordered) - 1) * q
    low = int(h)
    return ordered[low] + (h - low) * (ordered[min(low + 1, len(ordered) - 1)] - ordered[low])


def seed_for(seed, label):
    return int.from_bytes(hashlib.sha256(f"{seed}|{label}".encode()).digest()[:8], "little")


def interval(samples):
    return [percentile(samples, 0.025), percentile(samples, 0.975)]


def paired_statistics(rows, dimension, value, reps, seed):
    deltas = [r["V14_success"] - r["draft_success"] for r in rows]
    counts = Counter((r["V14_success"], r["draft_success"]) for r in rows)
    rng = random.Random(seed_for(seed, f"case|{dimension}|{value}"))
    case_draws = [sum(rng.choices(deltas, k=len(rows))) / len(rows) for _ in range(reps)]
    clusters = defaultdict(list)
    for row, delta in zip(rows, deltas):
        clusters[(row["mother_suite"], row["mother_task"])].append(delta)
    totals = [(sum(values), len(values)) for _, values in sorted(clusters.items())]
    rng = random.Random(seed_for(seed, f"cluster|{dimension}|{value}"))
    cluster_draws = []
    for _ in range(reps):
        chosen = rng.choices(totals, k=len(totals))
        cluster_draws.append(sum(x[0] for x in chosen) / sum(x[1] for x in chosen))
    return dict(dimension=dimension, value=value, n=len(rows), mother_tasks=len(clusters),
                Full_successes=sum(r["Full_success"] for r in rows),
                V14_successes=sum(r["V14_success"] for r in rows), draft_successes=sum(r["draft_success"] for r in rows),
                V14_only=counts[1, 0], draft_only=counts[0, 1], both_success=counts[1, 1], both_failure=counts[0, 0],
                paired_difference=sum(deltas) / len(rows), case_bootstrap_95=interval(case_draws),
                mother_task_cluster_bootstrap_95=interval(cluster_draws),
                cluster_size_distribution=dict(sorted(Counter(len(x) for x in clusters.values()).items())))


def analyze(reps, seed):
    evidence = []
    for name in ("README.md", "success_paired.csv"):
        p = SOURCE / name
        raw = p.read_bytes()
        evidence.append(dict(path=str(p), sha256=digest(raw), bytes=len(raw)))
    ledger = load(SOURCE / "audit_ledger.json", evidence)
    summary = load(SOURCE / "summary.json", evidence)
    acceptance = load(SOURCE / "root_data_acceptance.json", evidence)
    frozen = load(ROOT / "results/idea1_v3_2_e4_supplement/libero_plus_half_20261022T1216Z/manifest.json", evidence)
    assert frozen["manifest_hash"] == ledger["half_manifest_hash"]
    assert evidence[-1]["sha256"] == summary["half_manifest_file_sha256"]
    frozen_cases = {(r["group"], r["source_index"]): r for r in frozen["jobs"]}
    assert ledger["completed"] == summary["completed_identity_valid_conditions"] == 585
    assert summary["audit_ledger_sha256"] == acceptance["ledger_sha256"] == digest((SOURCE / "audit_ledger.json").read_bytes())
    registered = {(r["group"], r["source_index"]): r["case_hash"] for r in summary["paired_case_set"]}
    assert len(registered) == 195
    grouped = defaultdict(dict)
    native_equal_fields = ["initial_vector_sha256", "instruction", "H", "R", "max_steps", "settle_steps", "native_hz", "controller"]
    for row in ledger["selected_completed"]:
        p = ROOT / row["path"]
        obj = load(p, evidence)
        assert evidence[-1]["sha256"] == row["sha256"], f"Changed result: {p}"
        key = row["group"], row["source_index"]
        assert obj["status"] == "complete" and obj["success"] == row["success"]
        assert obj["manifest_hash"] == ledger["parent_manifest_hash"]
        assert obj["case_hash"] == registered[key] and obj["mode"] == row["mode"]
        assert obj["case"] == frozen_cases[key]["case"] and obj["case_hash"] == frozen_cases[key]["case_hash"]
        assert row["mode"] not in grouped[key]
        grouped[key][row["mode"]] = obj
    pairs = []
    for key, conditions in sorted(grouped.items()):
        full = conditions["full_d10"]
        case = full["case"]
        modes = ["full_d10", "v14_d10", case["matching_draft"]]
        assert set(conditions) == set(modes)
        assert all(conditions[m]["case"] == case for m in modes)
        assert all(conditions[m]["case_hash"] == full["case_hash"] for m in modes)
        for field in native_equal_fields:
            assert all(conditions[m]["native_info"][field] == full["native_info"][field] for m in modes), (key, field)
        assert case["mother_suite"] in SUITES and case["mother_task"]
        pairs.append(dict(group=key[0], source_index=key[1], case_hash=full["case_hash"],
                          mother_suite=case["mother_suite"], mother_task=case["mother_task"],
                          category=case["category"], difficulty=case["difficulty_level"],
                          trial=case["trial"], case_seed=case["seed"], matching_draft=case["matching_draft"],
                          initial_vector_sha256=full["native_info"]["initial_vector_sha256"],
                          Full_success=int(conditions["full_d10"]["success"]),
                          V14_success=int(conditions["v14_d10"]["success"]),
                          draft_success=int(conditions[case["matching_draft"]]["success"])))
    # A bounded check of the one known image exception: V14 vs draft is exact.
    case210 = {}
    for mode in ("full_d10", "v14_d10", "draft_object"):
        p = ROOT / f"results/idea1_v3_2_e4_supplement/libero_plus_half_20261022T1216Z/perturbed350/case0210/{mode}/attempt000/initial_native_observation.npz"
        raw = p.read_bytes()
        evidence.append(dict(path=str(p), sha256=digest(raw), bytes=len(raw)))
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            case210[mode] = {name: archive.read(name) for name in archive.namelist()}
    assert set(case210["full_d10"]) == set(case210["v14_d10"]) == set(case210["draft_object"])
    exception_check = dict(group="perturbed350", source_index=210,
                           V14_and_draft_all_members_exact=case210["v14_d10"] == case210["draft_object"],
                           Full_different_members=[name for name in case210["full_d10"] if case210["full_d10"][name] != case210["v14_d10"][name]],
                           members=sorted(case210["full_d10"]),
                           meaning="The prior Full-image exception is not an initial-input mismatch in the primary V14 vs pure-draft contrast.")
    assert exception_check["V14_and_draft_all_members_exact"]
    assert exception_check["Full_different_members"] == ["robot0_eye_in_hand_image.npy"]
    perturbed = [r for r in pairs if r["group"] == "perturbed350"]
    original = [r for r in pairs if r["group"] != "perturbed350"]
    assert len(perturbed) == 175 and len(original) == 20
    assert [sum(r[k] for r in perturbed) for k in ["Full_success", "V14_success", "draft_success"]] == [138, 89, 54]
    assert [sum(r[k] for r in original) for k in ["Full_success", "V14_success", "draft_success"]] == [20, 16, 8]
    assert len({(r["mother_suite"], r["mother_task"]) for r in perturbed}) == 40
    # Cross-check aggregate CSV without inheriting its Full-referenced pair counts.
    csv_rows = list(csv.DictReader(io.StringIO((SOURCE / "success_paired.csv").read_text())))
    for mode, column in [("full_d10", "Full_success"), ("v14_d10", "V14_success"), ("matching_draft", "draft_success")]:
        item = next(r for r in csv_rows if r["group"] == "perturbed350" and r["dimension"] == "overall" and r["condition"] == mode)
        assert int(item["successes"]) == sum(r[column] for r in perturbed)
    stats = [paired_statistics(perturbed, "overall", "all", reps, seed)]
    stats += [paired_statistics([r for r in perturbed if r["mother_suite"] == s], "mother_suite", s, reps, seed) for s in SUITES]
    stats += [paired_statistics([r for r in perturbed if r["category"] == c], "category", c, reps, seed) for c in CATEGORIES]
    original_context = dict(n=20, Full_successes=20, V14_successes=16, draft_successes=8,
                            distinct_mother_tasks=len({(r["mother_suite"], r["mother_task"]) for r in original}),
                            causal_perturbation_difference_computed=False,
                            reason="Original20 and perturbed175 differ in mother-task coverage and initial states; no same-task same-init unperturbed counterpart assignment is established.")
    result = dict(generated_utc=datetime.now(timezone.utc).isoformat(),
                  status="historical_descriptive_exploratory_only", pair_count=175,
                  protocol=dict(contrast="V14 success minus matching pure-draft success", unit="same perturbed instance and trial",
                                case_key=["group", "source_index", "case_hash"], cluster_key=["mother_suite", "mother_task"],
                                interval="Two-sided percentile bootstrap, 2.5th/97.5th percentiles with linear interpolation (type7)",
                                replicates=reps, base_seed=seed,
                                case_bootstrap="Resample n paired case differences with replacement; same sampled indices for both conditions.",
                                mother_task_cluster_bootstrap="Resample m mother-task clusters uniformly with replacement, retain every case within each chosen cluster, and compute sum(paired differences)/sum(case counts). Overall clusters are not suite-stratified.",
                                estimand="Case-weighted success-rate difference; cluster resampling retains unequal observed cluster sizes rather than changing to an equal-task estimand.",
                                scope="Intervals are descriptive sensitivity summaries on known historical data, not preregistered confirmatory inference; no p-values are computed.",
                                multiplicity="Subgroup intervals are unadjusted descriptive intervals; no across-subgroup significance ranking.",
                                draft_acceptance_rate=None, draft_Full_fallback=None),
                  identity_checks=dict(selected_results_read=585, selected_result_hashes_match_existing_ledger=True,
                                       three_condition_case_hash_and_case_object_match=True,
                                       native_metadata_equal_fields=native_equal_fields,
                                       mother_task_ids_confirmed=True, frozen_job_case_identity_matches=True,
                                       case210_initial_NPZ_check=exception_check, prior_full_request_audit_repeated=False,
                                       prior_acceptance_source=str(SOURCE / "root_data_acceptance.json"),
                                       prior_initial_image_differences=acceptance["initial_image_differences"]),
                  original20_context=original_context, statistics=stats, perturbed_pairs=perturbed,
                  limits=["Only this fixed LIBERO-Plus175 subset and these released mother-suite-matched pure-draft heads; not all trained drafts or the 10,030-instance benchmark.",
                          "Pure draft has no verification or Full fallback. Acceptance and Full-fallback are N/A; do not attach 2026-11-03 FLASH metrics.",
                          "V14 is a fixed VLM14 D10/H50/R5 policy with skip[3,4,9,11], not the current VLM12 Full/Fast/Fast system and not a same-compute-budget contrast with pure draft.",
                          "Full138 is a reference on the same perturbed cases. Original20 (20/16/8) is separate context, not a controlled no-perturbation baseline for these175 cases.",
                          "Historical outcomes have already been examined. Both case-level and mother-task cluster intervals are exploratory, and subgroup sample sizes/cluster counts are limited.",
                          "Previous acceptance records one initial wrist-image deviation of Full from V14 and draft for source_index210; V14/draft NPZs are exact matches. Retain all cases and do not assert zero effect for Full comparisons."])
    return result, evidence


def pct(value):
    return f"{100 * value:+.1f}"


def ci(values):
    return f"[{pct(values[0])}, {pct(values[1])}]"


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--replicates", type=int, default=20000)
    p.add_argument("--seed", type=int, default=20261103)
    args = p.parse_args()
    if args.replicates < 1000:
        p.error("Use at least 1000 descriptive bootstrap replicates")
    result, evidence = analyze(args.replicates, args.seed)
    evidence.append(dict(path=str(Path(__file__).resolve()), sha256=digest(Path(__file__).read_bytes()), role="analysis_source"))
    hashes_path = HERE / "input_hashes.json"
    hashes = dict(inputs=evidence, scientific_results_mutated=False, gpu_or_simulation_calls=0)
    hashes_path.write_text(json.dumps(hashes, indent=2) + "\n")
    result["input_hashes"] = dict(path=str(hashes_path), sha256=digest(hashes_path.read_bytes()))
    (HERE / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    first = result["statistics"][0]
    lines = ["# 历史扰动证据：V14 与匹配纯 draft 的配对补充", "",
             f"在已验收的同一175条LIBERO-Plus扰动实例上，V14成功89/175，匹配纯draft成功54/175；V14-only {first['V14_only']}例，draft-only {first['draft_only']}例，配对差+20.0pp。Full为138/175，作为同扰动case的参考。", "",
             f"整体95%实例配对bootstrap区间为{ci(first['case_bootstrap_95'])}pp；按40个母任务聚类重采样为{ci(first['mother_task_cluster_bootstrap_95'])}pp。历史结果已看过，全部为描述性/探索性补充，不新增确认性p值。", "",
             "| 范围 | case/母task | V14-only / draft-only | 差值pp | 实例配对95%区间pp | 母task聚类95%区间pp |",
             "| --- | ---: | ---: | ---: | --- | --- |"]
    for row in result["statistics"]:
        lines.append(f"| {LABELS[row['value']]} | {row['n']}/{row['mother_tasks']} | {row['V14_only']}/{row['draft_only']} | {pct(row['paired_difference'])} | {ci(row['case_bootstrap_95'])} | {ci(row['mother_task_cluster_bootstrap_95'])} |")
    lines += ["", "配对键由已有账本的group/source_index连接原result，三个条件的case_hash、case对象、初始物理状态hash、指令及环境/控制协议一致。母task明确为(mother_suite,mother_task名称)，40个母task各含1–9条扰动实例。复用原42,305请求验收，仅重新读取并核对585个小result JSON及其已有hash，没有重跑模型或逐请求审计。", "",
              f"两类区间均为{args.replicates:,}次重采样、固定seed={args.seed}、2.5/97.5百分位。实例法保留双方配对；cluster法整体在40个母task中有放回抽样，每次保留所抽母task全部实例，使用总差值/总病例数，保持case加权口径。分组在各自母task中重采样；小组区间未作多重比较校正，不据此排序显著性。", "",
              "原始20例单列为Full20/20、V14 16/20、纯draft8/20，只有20个母任务，且初态与扰动组不对应。因此本表描述的是这些扰动case上不同策略的表现差异，不能证明同task同init从无扰动到有扰动的因果降幅。先前验收保留source_index210的一处Full腕部图像差异（3像素/4通道、最大1级uint8）；本次只针对该例读取三份初始NPZ，确认V14与draft的6个成员及整文件hash完全相同。因此这不是主对比双方的初始输入不配对，但涉及Full时仍不能宣称图像差异零影响。所有结果保留。", "",
              "这项历史证据可说明：在本固定扰动子集与母suite匹配的已发布head上，纯draft存在较低成功率，且相对V14的差异随类别变化；它不能外推所有trained draft或完整FLASH。纯draft无verify/Full接管，接受率和Full fallback为N/A，不能借用11/03 FLASH统计。V14是跳[3,4,9,11]、保留VLM14、D10/H50/R5的固定路径，与当前VLM12周期Full/Fast/Fast协议不同，也不是与纯draft同预算的比较。", "",
              "[完整配对与区间](analysis.json) · [全部输入hash](input_hashes.json) · [CPU复验脚本](analyze.py) · [原报告](../../idea1_libero_plus_half_final_20261022T2124Z/README.md)", "",
              "```bash", f"python3 -B {HERE / 'analyze.py'}", "```", ""]
    (HERE / "README_zh.md").write_text("\n".join(lines))
    print(json.dumps(dict(output=str(HERE), overall=first, identity_checks=result["identity_checks"])))


if __name__ == "__main__":
    main()
