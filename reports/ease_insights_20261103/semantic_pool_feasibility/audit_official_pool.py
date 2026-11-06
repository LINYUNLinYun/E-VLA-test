"""Read official task/BDDL/CPU initial-state assets only; never read outcomes.

Run with CUDA_VISIBLE_DEVICES='' and PYTHONDONTWRITEBYTECODE=1 using the
libero-eval-v21 interpreter. This script constructs no benchmark environment.
"""
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
OFFICIAL = ROOT / "LIBERO/libero/libero"
CONFIGURED = ROOT / "policies/realtime-vla-flash/third_party/libero/libero/libero"
TAXONOMY = ROOT / "preparations/ease_insights_20261103/history/semantic_taxonomy/extract_taxonomy.py"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def packed(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def value_hash(value):
    return hashlib.sha256(packed(value).encode()).hexdigest()


def goal_identity(row):
    """A deliberately coarser operation-family sensitivity grouping.

    Retain goal graph, object/fixture types, region roles, preservation flags,
    requested state actions and category; ignore scene layout/distractors and
    instance numbering. Canonicalize identical-type entity role permutations.
    This is not claimed equivalent to full task/BDDL identity.
    """
    entities = {}
    for leaf in row["goal_predicates"]:
        for arg in leaf["resolved_arguments"]:
            kind = arg["kind"]
            if kind == "region":
                kind = "object" if arg["owner"] in row["objects"] else "fixture"
            assert kind in ("object", "fixture") and arg["type"]
            entities[arg["owner"]] = (kind, arg["type"])
    groups = defaultdict(list)
    for owner, kind_type in entities.items():
        groups[kind_type].append(owner)
    choices = []
    for kind_type, owners in sorted(groups.items()):
        choices.append([dict(zip(order, [(*kind_type, i) for i in range(len(order))]))
                        for order in itertools.permutations(sorted(owners))])
    representations = []
    for assignment in itertools.product(*choices):
        identities = {owner: identity for part in assignment for owner, identity in part.items()}
        leaves = []
        for leaf in row["goal_predicates"]:
            args = [(*identities[a["owner"]], a.get("region")) for a in leaf["resolved_arguments"]]
            leaves.append((leaf["predicate"].lower(), args,
                           leaf in row["initially_declared_state_goal_constraints"]))
        representations.append(packed(dict(category=row["category"], goals=sorted(leaves, key=packed),
            explicit_state_actions=sorted(set().union(*map(set, row["explicit_instruction_state_actions"].values()))))))
    return min(representations)


def normalized_bddl(form):
    # Ignore names/language only; retain the complete task's objects, scene
    # regions, initial constraints and goal. Normalize case and set ordering.
    def walk(node):
        if not isinstance(node, list):
            return node.lower()
        out = [walk(x) for x in node]
        if out and out[0] in ("and", ":init", ":regions", ":obj_of_interest"):
            out = [out[0], *sorted(out[1:], key=packed)]
        return out
    parts = [walk(x) for x in form if not (isinstance(x, list) and x and x[0].lower() in ("problem", ":language"))]
    return sorted(parts, key=packed)


def grouping(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    grouped = []
    for identity, members in sorted(groups.items()):
        categories = sorted({r["category"] for r in members})
        assert len(categories) == 1, (key, identity, categories)
        grouped.append(dict(identity=identity, category=categories[0], members=[
            dict(suite=r["suite"], task=r["task"], task_name=r["task_name"]) for r in members]))
    overlap = [g for g in grouped if any(r["suite"] == "libero_90" for r in g["members"])
               and any(r["suite"] != "libero_90" for r in g["members"])]
    return dict(unique_count=len(grouped), counts=dict(Counter(g["category"] for g in grouped)),
        cross_pool_overlap_group_count=len(overlap),
        existing40_members_in_overlap=sum(sum(r["suite"] != "libero_90" for r in g["members"]) for g in overlap),
        libero90_members_in_overlap=sum(sum(r["suite"] == "libero_90" for r in g["members"]) for g in overlap),
        groups=grouped, cross_pool_overlap_groups=overlap)


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Require explicit CPU-only environment"
    import numpy as np
    import torch
    config_path = ROOT/"policies/realtime-vla-flash/.libero/config.yaml"
    settings = {k.strip():v.strip() for line in config_path.read_text().splitlines() if line.strip()
                for k,v in [line.split(":", 1)]}
    assert Path(settings["bddl_files"]).resolve() == CONFIGURED/"bddl_files"
    assert Path(settings["init_states"]).resolve() == CONFIGURED/"init_files"
    spec = importlib.util.spec_from_file_location("official_blind_taxonomy", TAXONOMY)
    taxonomy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(taxonomy)
    task_map = taxonomy.literal_assignment(ast.parse((OFFICIAL/"benchmark/libero_suite_task_map.py").read_text()), "libero_task_map")
    configured_map = taxonomy.literal_assignment(ast.parse((CONFIGURED/"benchmark/libero_suite_task_map.py").read_text()), "libero_task_map")
    assert task_map == configured_map
    rows = []
    for suite in SUITES:
        assert len(task_map[suite]) == (90 if suite == "libero_90" else 10)
        for task, name in enumerate(task_map[suite]):
            bddl = OFFICIAL/"bddl_files"/suite/(name+".bddl")
            configured_bddl = CONFIGURED/"bddl_files"/suite/(name+".bddl")
            assert bddl.read_bytes() == configured_bddl.read_bytes()
            row = taxonomy.extract_task(suite, task, name, bddl, bddl.read_text())
            assert not row["unclassified_reasons"]
            init = CONFIGURED/"init_files"/suite/(name+".pruned_init")
            states = torch.load(init, map_location="cpu", weights_only=False)
            array = np.asarray(states)
            assert len(states) == 50 and array.ndim == 2 and np.isfinite(array).all()
            row.update(full_bddl_identity=value_hash(normalized_bddl(taxonomy.parse_bddl(bddl.read_text()))),
                goal_operation_identity=value_hash(goal_identity(row)),
                goal_operation_definition=json.loads(goal_identity(row)),
                configured_bddl_path=str(configured_bddl), configured_bddl_sha256=sha(configured_bddl),
                init_path=str(init), init_sha256=sha(init), init_shape=list(array.shape), init_dtype=str(array.dtype),
                init_all_finite=True)
            rows.append(row)
    groupings = {key: grouping(rows, key) for key in ("task_name", "bddl_sha256", "full_bddl_identity", "benchmark_instruction", "goal_operation_identity")}
    sources = [TAXONOMY, OFFICIAL/"benchmark/libero_suite_task_map.py", OFFICIAL/"benchmark/__init__.py",
               CONFIGURED/"benchmark/libero_suite_task_map.py", CONFIGURED/"benchmark/__init__.py",
               ROOT/"policies/realtime-vla-flash/.libero/config.yaml"]
    receipt = dict(schema="ease-official-libero90-semantic-pool-feasibility-v1",
        data_access="Official task maps, benchmark source, BDDL, configured official CPU initial-state files, existing taxonomy source and path configuration only; no outcome/episode/result files read",
        environment_constructed=False, gpu_launched=False, current_experiment_or_queue_modified=False,
        assets=dict(task_count=130, libero90_task_count=90, configured_bddl_present_and_same_as_official=130,
            official_initial_state_files_cpu_loaded=130, initial_states_per_task=50, all_initial_values_finite=True,
            completeness_scope="Task-map/BDDL/initial-state assets; no claim of rendered environment, mesh dependency, model prompt-cache or GPU execution readiness"),
        taxonomy=dict(source=str(TAXONOMY), source_sha256=sha(TAXONOMY), unchanged_rule=True,
            classes=list(taxonomy.CLASSES), labels=taxonomy.LABELS),
        counts_by_suite={s:dict(Counter(r["category"] for r in rows if r["suite"] == s)) for s in SUITES},
        total_counts=dict(Counter(r["category"] for r in rows)), groupings=groupings,
        identity_definitions=dict(task_name="Full official task name, including SCENE prefix when present",
            bddl_sha256="Exact BDDL bytes",
            full_bddl_identity="Parsed complete BDDL ignoring problem name and language; lowercases atoms and sorts top-level fields, conjunctions, init lists, regions and object-of-interest lists; preserves scene geometry, init constraints, object identities and all goals",
            benchmark_instruction="Exact official filename-derived instruction after removing scene prefix",
            goal_operation_identity="Coarser sensitivity grouping: typed goal-argument graph, region roles, state-preservation flags and explicit state actions; canonical identical-type entity renaming; ignores scene geometry, initial position, distractors and which same-type instance was named"),
        interpretation=[
            "There are 130 distinct official task instances by name and full BDDL identity; this is not proof of statistical independence or 130 distinct operation meanings.",
            "The coarse goal-operation grouping is a sensitivity analysis, not a replacement for the frozen taxonomy or an assertion that grouped tasks are identical environments.",
            "Task-instance counts reach 10 for state-only, state-plus-placement and single placement; coarse operation-family counts reach 10 only for state-only and single placement.",
            "Multiple placement remains below ten even at the official task-instance level.",
            "Future category/cluster definitions and case selection must be frozen separately; current720/possible600 evaluation is unchanged.",
        ], tasks=rows, sources=[dict(path=str(p), sha256=sha(p)) for p in sources],
        generator=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)))
    (OUTPUT/"official_pool_audit.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n")
    fields = ["suite", "task", "task_name", "category", "benchmark_instruction", "full_bddl_identity", "goal_operation_identity", "bddl_path", "bddl_sha256", "init_path", "init_sha256", "init_shape"]
    with (OUTPUT/"task_inventory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows({key:row[key] for key in fields} for row in rows)
    print(json.dumps(dict(assets=receipt["assets"], counts_by_suite=receipt["counts_by_suite"],
        grouped={k:{n:v[n] for n in ("unique_count", "counts", "cross_pool_overlap_group_count", "existing40_members_in_overlap", "libero90_members_in_overlap")} for k,v in groupings.items()})))


if __name__ == "__main__":
    main()
