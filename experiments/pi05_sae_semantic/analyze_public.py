"""Analyze the authors' concept/layer results without recomputing model activations.

Feature IDs are only compared within a single SAE, never between layers.
The derived tables describe public evidence; they do not measure FFN skipping.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re

import numpy as np


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def success_value(value):
    if value is True or value == "true":
        return 1
    if value is False or value == "false":
        return 0
    return None


def analyze(assets, references, github, output):
    output.mkdir(parents=True, exist_ok=True)
    source = assets / "concepts/concept_features_pi05.json"
    index = json.loads(source.read_text())
    rows, lookup = [], {}
    for key, categories in index.items():
        match = re.fullmatch(r"(paligemma|action_expert)_layer_(\d+)", key)
        if not match:
            continue
        pathway, layer = match.group(1), int(match.group(2))
        for category, concepts in categories.items():
            for concept, entry in concepts.items():
                ids = entry.get("feature_indices", [])
                row = dict(pathway=pathway, layer=layer, category=category, concept=concept,
                           listed_features=len(ids), reported_concept_features=entry.get("concept_features"),
                           reported_strong_features=entry.get("strong_features"),
                           reported_very_strong_features=entry.get("very_strong_features"),
                           max_cohens_d=entry.get("max_cohens_d"), feature_ids=json.dumps(ids))
                rows.append(row)
                lookup[(pathway, layer, category, concept)] = entry
    write_csv(output / "public_concept_layer.csv", rows)

    # The public JSON is a capped concept index, not a ranked per-feature score table.
    concepts = sorted({(r["category"], r["concept"]) for r in rows if r["pathway"] == "paligemma"})
    candidates = []
    for category, concept in concepts:
        a = lookup.get(("paligemma", 5, category, concept))
        b = lookup.get(("paligemma", 16, category, concept))
        if a is None or b is None:
            continue
        candidates.append(dict(category=category, concept=concept,
                               L5_max_cohens_d=a["max_cohens_d"], L16_max_cohens_d=b["max_cohens_d"],
                               min_endpoint_max_d=min(a["max_cohens_d"], b["max_cohens_d"]),
                               L5_feature_ids=a["feature_indices"], L16_feature_ids=b["feature_indices"]))
    candidates.sort(key=lambda r: r["min_endpoint_max_d"], reverse=True)
    (output / "public_L5_L16_candidates.json").write_text(json.dumps(dict(
        selection="Shared concept labels with candidates in both layers; sorted only by endpoint summary d",
        feature_order="Unranked candidate pools; original per-feature d/frequency are not in this index",
        candidates=candidates), indent=2))

    # Label confounds: compare candidate pools within each layer, then summarize over layers.
    pairs = []
    for a, b in itertools.combinations(concepts, 2):
        overlaps = []
        for layer in range(18):
            first = lookup.get(("paligemma", layer, *a))
            second = lookup.get(("paligemma", layer, *b))
            if first is None or second is None:
                continue
            aa, bb = set(first["feature_indices"]), set(second["feature_indices"])
            if aa or bb:
                overlaps.append(len(aa & bb) / len(aa | bb))
        if overlaps and np.mean(overlaps) >= .8:
            pairs.append(dict(first="/".join(a), second="/".join(b), layers_compared=len(overlaps),
                              mean_within_layer_jaccard=float(np.mean(overlaps)),
                              identical_pool_layers=sum(v == 1 for v in overlaps)))
    pairs.sort(key=lambda r: r["mean_within_layer_jaccard"], reverse=True)
    write_csv(output / "public_concept_overlap.csv", pairs)

    statistics_path = references / "action-atlas-pi05-activations/analysis/layer_analysis_results.json"
    statistics_rows = []
    if statistics_path.exists():
        published = json.loads(statistics_path.read_text())
        for suite, paths in published["results"].items():
            for pathway, values in paths.items():
                for layer, entry in values.get("layers", {}).items():
                    statistics_rows.append(dict(suite=suite, pathway=pathway, layer=int(layer), **entry))
        write_csv(output / "public_activation_statistics.csv", statistics_rows)

    ablation_path = assets / "videos/ablation_index.json"
    if not ablation_path.exists():
        ablation_path = github / "pi05_ablation_index.json"
    ablations = json.loads(ablation_path.read_text())
    videos = ablations.get("videos", []) if isinstance(ablations, dict) else ablations
    grouped = defaultdict(list)
    for record in videos:
        kind = record.get("subtype", record.get("experiment_type", "unspecified"))
        pathway = record.get("pathway", "unspecified")
        key = (record.get("suite", "unspecified"), pathway, kind, record.get("layer"), record.get("concept", ""))
        grouped[key].append(record)
    ablation_rows = []
    for (suite, pathway, kind, layer, concept), records in grouped.items():
        scored = [success_value(r.get("success")) for r in records]
        scored = [v for v in scored if v is not None]
        ablation_rows.append(dict(suite=suite, pathway=pathway, experiment=kind, layer=layer, concept=concept,
                                  listed_episodes=len(records), scored_episodes=len(scored), successes=sum(scored),
                                  listed_success_rate=sum(scored) / len(scored) if scored else ""))
    write_csv(output / "public_ablation_counts.csv", ablation_rows)

    # Make standalone research figures, preserving absent entries as missing data.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(concepts) * .25)), constrained_layout=True)
    all_values = [r["max_cohens_d"] for r in rows if r["max_cohens_d"] is not None]
    vmax = max(all_values)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#dddddd")
    for axis, pathway in zip(axes, ("paligemma", "action_expert")):
        matrix = np.full((len(concepts), 18), np.nan)
        for i, (category, concept) in enumerate(concepts):
            for layer in range(18):
                item = lookup.get((pathway, layer, category, concept))
                if item:
                    matrix[i, layer] = item.get("max_cohens_d", np.nan)
        im = axis.imshow(matrix, aspect="auto", interpolation="nearest", norm=PowerNorm(.4, vmin=0, vmax=vmax), cmap=cmap)
        axis.set_xticks(range(18))
        axis.set_yticks(range(len(concepts)), [f"{c}/{n}" for c, n in concepts])
        axis.set_xlabel("Layer (0-based)")
        axis.set_title(f"{pathway}: published max Cohen's d")
        for boundary in (4.5, 5.5, 15.5, 16.5):
            axis.axvline(boundary, color="white", lw=.8, alpha=.8)
    fig.colorbar(im, ax=axes, label="Published max d (power-scaled colors; not skip sensitivity)", shrink=.6)
    fig.savefig(output / "public_concept_layers.png", dpi=180)
    fig.savefig(output / "public_concept_layers.pdf")
    plt.close(fig)

    selected = candidates[:10]
    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    xx = np.arange(len(selected))
    axis.bar(xx - .18, [r["L5_max_cohens_d"] for r in selected], width=.36, label="L5")
    axis.bar(xx + .18, [r["L16_max_cohens_d"] for r in selected], width=.36, label="L16")
    axis.set_xticks(xx, [r["category"] + "/" + r["concept"] for r in selected], rotation=35, ha="right")
    axis.set_ylabel("Published max Cohen's d")
    axis.set_title("Concept candidates present at both L5 and L16")
    axis.legend()
    fig.savefig(output / "public_L5_L16_candidates.png", dpi=180)
    plt.close(fig)

    source_files = [source, ablation_path, assets / "asset_manifest.json"]
    if statistics_path.exists():
        source_files.append(statistics_path)
    provenance = [dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in source_files]
    summary = dict(scope="Reanalysis of published results, not new local VLA/FFN measurements",
                   source_files=provenance, index_metadata=index.get("_metadata"),
                   n_layer_keys=len({(r["pathway"], r["layer"]) for r in rows}),
                   n_concept_layer_entries=len(rows), n_shared_endpoint_concepts=len(candidates),
                   ablation_source=str(ablation_path), n_listed_ablation_records=len(videos),
                   n_ablation_records_with_success=sum(success_value(r.get("success")) is not None for r in videos),
                   ablation_layers=dict(Counter(str(r.get("layer")) for r in videos)),
                   top_endpoint_candidates=[{k: v for k, v in c.items() if "feature_ids" not in k} for c in selected],
                   overlapping_labels=pairs[:12])
    (output / "public_analysis.json").write_text(json.dumps(summary, indent=2))
    report = ["# π0.5 SAE 语义—层关系：公开资产复用首轮结果", "",
              "本轮直接重分析 Action Atlas 发布的概念索引、网站数据和已有层统计；没有重新训练 SAE 或重复采集轨迹。",
              "本地 checkpoint 迁移检查和 FFN 跳过干预仍待实测，以下结果不作为跳过安全性结论。", "",
              f"覆盖 {summary['n_layer_keys']} 个 pathway/layer、{len(rows)} 条概念—层索引，L5/L16 共有 {len(candidates)} 个候选概念。",
              "", "## L5/L16 候选概念", "",
              "按两端 max d 的较小值排序，作为本地复核候选；这不是逐特征排名，也不是对所有候选的语义正确率估计。", "",
              "| 概念 | L5 max d | L16 max d |", "|---|---:|---:|"]
    report += [f"| {r['category']}/{r['concept']} | {r['L5_max_cohens_d']:.3f} | {r['L16_max_cohens_d']:.3f} |" for r in selected]
    report += ["", "![公开概念跨层分布](public_concept_layers.png)", "", "## 对后续实验的直接帮助", "",
               "- L5 与 L16 已有公共概念候选，可直接复用两层各自的 feature ID 池；不需要从零发现概念或训练字典。",
               "- 使用公开层分布选择需要本地检查的概念；ID 仅在相应层的 SAE 内有意义。",
               "- 概念索引只提供候选 ID、数量和 max d，没有逐特征 d/激活频率。网站自动描述可用于检索线索；缺失的逐特征统计需继续查原始结果或在小批激活上计算，不能编造排名。",
               "- 原始激活、统计与本地观测分开记来源；公开激活可先验证官方 SAE 编码和重构，本地观测用于检查迁移后是否仍有语义区分。", "",
               "## 概念重叠与解释", ""]
    for pair in pairs[:5]:
        report.append(f"- `{pair['first']}` 与 `{pair['second']}`：{pair['layers_compared']} 层的平均同层候选集合 Jaccard={pair['mean_within_layer_jaccard']:.3f}，其中 {pair['identical_pool_layers']} 层完全相同。")
    report += ["", "这些重叠可能反映任务标签共现，不能当作发现了彼此独立的语义机制；选少量概念时优先覆盖不同标签组合。",
               "", "## 已有消融结果", "",
               f"已整理 `{ablation_path.name}` 中 {len(videos)} 条记录，其中 {summary['n_ablation_records_with_success']} 条带 success 字段。逐 suite/layer/concept 的覆盖量及可评分成功数见 `public_ablation_counts.csv`；其余成功率留空，不记为失败。",
               "这些是发布表中的记录计数；缺失 pathway 标为 unspecified，不推断归属，也不把视频索引视为完整随机采样实验。",
               "SAE feature 消融与跳过 FFN 是不同干预；这里的结果用于寻找线索，不能替代计划中的 Concept–FFN 实测矩阵。",
               "", "## 数据口径", "",
               "- 灰色格表示索引缺失；未发布概念不记为零。max d 是发布摘要，不保证跨不同 SAE、任务组成和样本池完全可比。",
               "- feature_indices 是截断候选池，数量不是全字典中该概念的神经元总数；strong/very_strong 原样保存，不自行解释为校准置信度。",
               "- 既有层激活方差和时间变化保存于 `public_activation_statistics.csv`；激活幅度不直接代表控制重要性。",
               "- 三个 Hugging Face 仓库均固定 revision；原文件 SHA256 和引用入口保存于 manifests。", ""]
    (output / "public_analysis.md").write_text("\n".join(report))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets", required=True)
    p.add_argument("--references", required=True)
    p.add_argument("--github", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = analyze(*(Path(getattr(args, k)).resolve() for k in ("assets", "references", "github", "output")))
    print(json.dumps({k: v for k, v in result.items() if k not in {"source_files", "index_metadata", "overlapping_labels"}}, indent=2))


if __name__ == "__main__":
    main()
