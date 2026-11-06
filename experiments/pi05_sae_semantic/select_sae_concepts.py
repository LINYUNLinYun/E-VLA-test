"""Discovery-only feature selection and semantic calibration for the frozen pilot.

Public feature IDs are candidates. Local observation-level max activations,
task/episode-balanced descriptive statistics, and three semantic thresholds
implement sections 4-5 of the experiment plan. No action or rollout SR is read.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re

import numpy as np

TARGETS = [("object", "bowl"), ("object", "stove"), ("spatial", "on")]


def concept_label(instruction, category, concept):
    text = instruction.lower().replace("_", " ")
    if (category, concept) == ("spatial", "on"):
        text = re.sub(r"\bturn\s+on\b", "turn", text)
    return bool(re.search(r"\b" + re.escape(concept.replace("_", " ")) + r"\b", text))


def balanced_weights(rows):
    episodes = Counter((r["suite"], r["task"], r["init"]) for r in rows)
    per_task = Counter((suite, task) for suite, task, init in episodes)
    weights = np.array([1 / episodes[(r["suite"], r["task"], r["init"])] / per_task[(r["suite"], r["task"])]
                        for r in rows], dtype=np.float64)
    return weights / weights.sum()


def weighted_quantile(x, weights, q):
    order = np.argsort(x, kind="stable")
    return float(x[order][min(np.searchsorted(np.cumsum(weights[order]) / weights.sum(), q), len(x)-1)])


def weighted_mean(x, weights):
    return float(np.average(x, weights=weights))


def metrics(trigger, labels, rows, weights):
    positive = labels.astype(bool)
    count = int(trigger.sum())
    hits = trigger & positive
    return dict(triggered=count, positive_hits=int(hits.sum()),
                precision=weighted_mean(positive[trigger], weights[trigger]) if count else None,
                recall=weighted_mean(trigger[positive], weights[positive]) if positive.any() else None,
                false_positive_rate=weighted_mean(trigger[~positive], weights[~positive]) if (~positive).any() else None,
                positive_tasks=len({(r["suite"], r["task"]) for r, flag in zip(rows, hits) if flag}),
                positive_episodes=len({(r["suite"], r["task"], r["init"]) for r, flag in zip(rows, hits) if flag}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--collection", required=True)
    p.add_argument("--assets", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--max-features", type=int, default=4)
    p.add_argument("--candidate-pool",choices=("released","full_dictionary"),default="released")
    p.add_argument("--token-pool",choices=("all","image","instruction"),default="all")
    args = p.parse_args()
    collection, output = Path(args.collection).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((collection / "collection_config.json").read_text())
    folders = [collection / "observations" / identity for identity in cfg["observations"]]
    ready = [f for f in folders if (f / "complete.json").exists()]
    if len(ready) != len(folders) and not args.allow_partial:
        raise ValueError(f"Wait for complete frozen pilot: {len(ready)}/{len(folders)}")
    rows = [json.loads((f / "complete.json").read_text()) for f in ready]
    if not rows or any(r["split"] not in {"discovery", "calibration"} for r in rows):
        raise ValueError("Only discovery/calibration observations may enter feature selection")
    weights = balanced_weights(rows)
    discovery = np.array([r["split"] == "discovery" for r in rows])
    calibration = ~discovery
    published = json.loads((Path(args.assets) / "concepts/concept_features_pi05.json").read_text())
    readout_file='readouts.npz'
    analysis_ids=cfg['feature_ids']
    if args.candidate_pool=='full_dictionary':
        from .dictionary_readouts import ensure_dictionary_readouts
        analysis_ids=ensure_dictionary_readouts(collection,ready,Path(args.assets).resolve())
        readout_file='readouts_all_features.npz'
    arrays = {key: [] for key in [f"a{layer}_{kind}" for layer in (5, 16) for kind in ("all", "image", "instruction")]}
    for f in ready:
        with np.load(f / readout_file, allow_pickle=False) as archive:
            for key in arrays:
                arrays[key].append(archive[key])
    arrays = {k: np.stack(v) for k, v in arrays.items()}
    criteria = dict(targets=["/".join(c) for c in TARGETS], max_features=args.max_features,
                    candidate_pool=args.candidate_pool,
                    token_pool=args.token_pool,
                    discovery_feature_precision_reference=.8, minimum_positive_trigger_episodes=2,
                    calibration_precision_reference=.8, calibration_false_positive_rate_reference=.2,
                    feature_count_rule="Choose 1..4 top-R features using discovery precision/coverage at score 0.8; calibration does not select features",
                    concept_threshold_candidates=[.8, .6, .9], quantile_method="task/episode-balanced empirical inverse CDF",
                    scope="Exploratory semantic criteria, not calibrated deployment probabilities or action guarantees")
    (output / "selection_config.json").write_text(json.dumps(criteria, indent=2))
    result, feature_rows, score_arrays = {}, [], {}
    for category, concept in TARGETS:
        name = f"{category}/{concept}"
        labels = np.array([concept_label(r["instruction"], category, concept) for r in rows])
        result[name] = dict(category=category, concept=concept, layers={})
        for layer in (5, 16):
            ids = analysis_ids[str(layer)]
            locations = {int(f): i for i, f in enumerate(ids)}
            pool = (ids if args.candidate_pool=='full_dictionary' else
                    published[f"paligemma_layer_{layer}"].get(category, {}).get(concept, {}).get("feature_indices", []))
            positive, negative = discovery & labels, discovery & ~labels
            pos_tasks = {(r["suite"], r["task"]) for r, flag in zip(rows, positive) if flag}
            neg_tasks = {(r["suite"], r["task"]) for r, flag in zip(rows, negative) if flag}
            layer_info = dict(positive_tasks=len(pos_tasks), negative_tasks=len(neg_tasks), features=[], supported=False)
            result[name]["layers"][str(layer)] = layer_info
            if len(pos_tasks) < 2 or len(neg_tasks) < 2 or not np.any(calibration & labels) or not np.any(calibration & ~labels):
                layer_info["reason"] = "Insufficient positive/negative task support in current batch"
                continue
            eligible = []
            for feature in pool:
                if feature not in locations:
                    continue
                a = arrays[f"a{layer}_{args.token_pool}"][:, locations[feature]].astype(np.float64)
                if not np.any(a[positive]>0):
                    continue
                pos_mu, neg_mu = weighted_mean(a[positive], weights[positive]), weighted_mean(a[negative], weights[negative])
                pos_var = weighted_mean((a[positive]-pos_mu)**2, weights[positive])
                neg_var = weighted_mean((a[negative]-neg_mu)**2, weights[negative])
                d = (pos_mu-neg_mu) / max(np.sqrt((pos_var+neg_var)/2), 1e-8)
                frequency = weighted_mean(a[positive] > 0, weights[positive])
                relevance = max(d, 0) * frequency
                tau = weighted_quantile(a[negative], weights[negative], .95)
                trigger = a > tau
                support = positive & trigger
                stat = metrics(trigger[discovery], labels[discovery], [r for r, flag in zip(rows, discovery) if flag], weights[discovery])
                reference = weighted_quantile(a[support], weights[support], .5) if support.any() else None
                scale = reference-tau if reference is not None else None
                usable = (relevance > 0 and scale is not None and scale > 1e-8 * max(1., abs(reference), abs(tau))
                          and stat["precision"] is not None and stat["precision"] >= .8 and stat["positive_episodes"] >= 2)
                f = dict(feature_id=int(feature), d=float(d), positive_frequency=frequency, R=float(relevance),
                         tau=tau, r=reference, b=scale, discovery_metrics=stat)
                feature_rows.append(dict(concept=name, layer=layer, feature_id=int(feature), d=float(d),
                                         positive_frequency=frequency, R=float(relevance), tau=tau, r=reference, b=scale,
                                         discovery_precision=stat["precision"], discovery_recall=stat["recall"],
                                         positive_trigger_tasks=stat["positive_tasks"], positive_trigger_episodes=stat["positive_episodes"],
                                         eligible=usable))
                if usable:
                    eligible.append(f)
            ranked = sorted(eligible, key=lambda f: f["R"], reverse=True)[:args.max_features]
            count_options = []
            for count in range(1,len(ranked)+1):
                subset=ranked[:count]
                rel=np.array([f['R'] for f in subset]);rel/=rel.sum()
                a=arrays[f'a{layer}_{args.token_pool}'][:,[locations[f['feature_id']] for f in subset]]
                u=(a-np.array([f['tau'] for f in subset]))/np.array([f['b'] for f in subset])
                score=np.clip(u,0,1) @ rel
                stat=metrics(score[discovery]>=.8,labels[discovery],[r for r,flag in zip(rows,discovery) if flag],weights[discovery])
                count_options.append(dict(features=count,**stat))
            usable_counts=[r for r in count_options if r['precision'] is not None and r['precision']>=.8]
            chosen_count=max(usable_counts,key=lambda r:(r['recall'],r['precision'],-r['features']))['features'] if usable_counts else 1
            selected=ranked[:chosen_count]
            layer_info['feature_count_options']=count_options
            layer_info["features"] = selected
            if not selected:
                layer_info["reason"] = "No supported selective features under this pilot/configuration; inspect candidate statistics"
                continue
            relevance = np.array([f["R"] for f in selected]); relevance /= relevance.sum()
            for f, weight in zip(selected, relevance):
                f["weight"] = float(weight)
            for kind in ("all", "image", "instruction"):
                a = arrays[f"a{layer}_{kind}"][:, [locations[f["feature_id"]] for f in selected]]
                u = (a - np.array([f["tau"] for f in selected])) / np.array([f["b"] for f in selected])
                score = np.clip(u, 0, 1) @ relevance
                score_arrays[f"L{layer}_{category}_{concept}_{kind}"] = score
            score = score_arrays[f"L{layer}_{category}_{concept}_{args.token_pool}"]
            cal_rows = [r for r, flag in zip(rows, calibration) if flag]
            options = []
            for theta in criteria["concept_threshold_candidates"]:
                stat = metrics(score[calibration] >= theta, labels[calibration], cal_rows, weights[calibration])
                options.append(dict(theta=theta, **stat))
            valid = [r for r in options if r["precision"] is not None and r["precision"] >= .8
                     and r["false_positive_rate"] <= .2 and r["positive_episodes"] >= 2]
            chosen = valid[0] if valid else options[0]
            layer_info.update(theta=chosen["theta"], calibration_metrics=chosen, threshold_options=options,
                              supported=bool(valid), reason="Semantic calibration supported" if valid else "Candidate only; semantic calibration not supported")
        result[name]["ready_for_intervention"] = all(result[name]["layers"][str(layer)]["supported"] for layer in (5, 16))
    with (output / "feature_discovery.csv").open("w", newline="") as stream:
        if feature_rows:
            writer = csv.DictWriter(stream, fieldnames=list(feature_rows[0])); writer.writeheader(); writer.writerows(feature_rows)
    (output / "concept_features.json").write_text(json.dumps(result, indent=2))
    thresholds = {name: {layer: dict(theta=value.get("theta", .8), supported=value["supported"],
        features=[{k:f[k] for k in ("feature_id", "tau", "r", "b", "weight")} for f in value["features"]])
        for layer,value in item["layers"].items()} for name,item in result.items()}
    (output / "semantic_thresholds.json").write_text(json.dumps(thresholds, indent=2))
    np.savez_compressed(output / "semantic_scores.npz", **score_arrays)
    (output / "score_observations.json").write_text(json.dumps(rows, indent=2))
    summary = dict(completed_observations=len(rows), expected_observations=len(folders), partial=len(rows)!=len(folders),
                   candidate_pool=args.candidate_pool,readout_file=readout_file,
                   token_pool=args.token_pool,
                   ready_concepts=[name for name, value in result.items() if value["ready_for_intervention"]],
                   concepts={name:{layer:{k:v for k,v in value.items() if k not in {"features", "threshold_options"}}
                                   for layer,value in item["layers"].items()} for name,item in result.items()},
                   scope="Semantic calibration only; no rollout/action information used")
    (output / "semantic_selection_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
