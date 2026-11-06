#!/usr/bin/env python3
"""Build a results-blind semantic taxonomy from official LIBERO metadata only.

This script imports no benchmark runtime, simulator, model, torch, or GPU code.
It reads the official task map, benchmark ordering/language source, and 40 BDDL
files. It never reads experimental outcomes, manifests, or path comparisons.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[4]
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
STATE_PREDICATES = {"open", "close", "turnon", "turnoff"}
PLACEMENT_PREDICATES = {"on", "in"}
CLASSES = (
    "state_change_only", "state_change_and_object_placement",
    "multiple_object_placement", "single_object_placement", "unknown",
)
LABELS = {
    "state_change_only": "仅状态改变",
    "state_change_and_object_placement": "状态改变与物体放置",
    "multiple_object_placement": "多物体放置或移位",
    "single_object_placement": "单物体放置或移位",
    "unknown": "无法按当前规则确定",
}
ACTION_PATTERNS = {"open": r"\bopen\b", "close": r"\bclose\b",
                   "turnon": r"\bturn\s+on\b", "turnoff": r"\bturn\s+off\b"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def literal_assignment(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"Official metadata assignment missing: {name}")


def benchmark_language(filename):
    # Exact rule in official grab_language_from_filename; original text retained.
    if filename[0].isupper():
        offset = 8 if "SCENE10" in filename else 7
        language = " ".join(filename[filename.find("SCENE") + offset:].split("_"))
    else:
        language = " ".join(filename.split("_"))
    return language[:language.find(".bddl")]


def parse_bddl(text):
    tokens = re.findall(r"\(|\)|[^\s()]+", re.sub(r";[^\n]*", "", text))
    stack, roots = [], []
    for token in tokens:
        if token == "(":
            value = []
            (stack[-1] if stack else roots).append(value)
            stack.append(value)
        elif token == ")":
            if not stack:
                raise ValueError("Unbalanced BDDL closing parenthesis")
            stack.pop()
        else:
            if not stack:
                raise ValueError("Token outside BDDL expression")
            stack[-1].append(token)
    if stack or len(roots) != 1 or roots[0][0] != "define":
        raise ValueError("Unsupported BDDL root")
    return roots[0]


def goal_leaves(node):
    if not isinstance(node, list) or not node:
        return [], ["Empty or non-expression goal"]
    if node[0].lower() == "and":
        leaves, errors = [], []
        for child in node[1:]:
            a, b = goal_leaves(child)
            leaves.extend(a)
            errors.extend(b)
        return leaves, errors
    if any(isinstance(x, list) for x in node):
        return [], ["Unsupported nested or non-conjunctive goal: " + repr(node)]
    return [node], []


def declarations(tokens):
    result, pending, index = {}, [], 0
    while index < len(tokens):
        token = tokens[index]
        if token == "-":
            if not pending or index + 1 == len(tokens):
                raise ValueError("Malformed typed object declaration")
            for name in pending:
                result[name] = tokens[index + 1]
            pending = []
            index += 2
        else:
            if not isinstance(token, str):
                raise ValueError("Unexpected nested object declaration")
            pending.append(token)
            index += 1
    if pending:
        raise ValueError("Untyped object declaration")
    return result


def classify(state_required, placement_objects, errors):
    if errors or (not state_required and not placement_objects):
        return "unknown"
    if state_required and placement_objects:
        return "state_change_and_object_placement"
    if state_required:
        return "state_change_only"
    return "multiple_object_placement" if len(placement_objects) >= 2 else "single_object_placement"


def extract_task(suite, task, name, path, text):
    form = parse_bddl(text)
    sections = {x[0].lower(): x[1:] for x in form if isinstance(x, list) and x and x[0].startswith(":")}
    if len(sections.get(":goal", [])) != 1:
        raise ValueError("Exactly one BDDL goal expression is required")
    leaves, errors = goal_leaves(sections[":goal"][0])
    objects = declarations(sections.get(":objects", []))
    fixtures = declarations(sections.get(":fixtures", []))
    regions = {}
    for region in sections.get(":regions", []):
        targets = [x[1] for x in region[1:] if isinstance(x, list) and x and x[0] == ":target"]
        if len(targets) != 1:
            raise ValueError("A region must have one explicit target")
        regions[targets[0] + "_" + region[0]] = {"name": region[0], "owner": targets[0]}

    def resolve(symbol):
        if symbol in objects:
            return {"symbol": symbol, "kind": "object", "owner": symbol, "type": objects[symbol]}
        if symbol in fixtures:
            return {"symbol": symbol, "kind": "fixture", "owner": symbol, "type": fixtures[symbol]}
        if symbol in regions:
            owner = regions[symbol]["owner"]
            return {"symbol": symbol, "kind": "region", "owner": owner,
                    "type": objects.get(owner, fixtures.get(owner)), "region": regions[symbol]["name"]}
        return {"symbol": symbol, "kind": "unresolved", "owner": None, "type": None}

    instruction = benchmark_language(name + ".bddl")
    bddl_instruction = " ".join(sections.get(":language", []))
    actions = {source: sorted(action for action, pattern in ACTION_PATTERNS.items()
                              if re.search(pattern, phrase.lower()))
               for source, phrase in (("benchmark_instruction", instruction), ("bddl_instruction", bddl_instruction))}
    language_actions = sorted(set(actions["benchmark_instruction"]) | set(actions["bddl_instruction"]))
    initial = {tuple(str(y).lower() for y in x) for x in sections.get(":init", []) if isinstance(x, list)}
    state, maintained, placement, predicates = [], [], [], []
    for leaf in leaves:
        pred = leaf[0].lower()
        present = tuple(x.lower() for x in leaf) in initial
        item = {"predicate": leaf[0], "arguments": leaf[1:], "literal": leaf,
                "exact_literal_declared_in_init": present,
                "resolved_arguments": [resolve(x) for x in leaf[1:]]}
        predicates.append(item)
        if pred in STATE_PREDICATES and len(leaf) == 2:
            (maintained if present and pred not in language_actions else state).append(item)
        elif pred in PLACEMENT_PREDICATES and len(leaf) == 3:
            placement.append(item)
            if leaf[1] not in objects:
                errors.append("Placement subject is not a declared movable object: " + leaf[1])
            if any(x["kind"] == "unresolved" for x in item["resolved_arguments"]):
                errors.append("Unresolved placement goal argument: " + repr(leaf))
        else:
            errors.append("Unsupported goal predicate or arity: " + repr(leaf))
    placed = sorted({p["arguments"][0] for p in placement})
    language_only = sorted(set(language_actions) - {p["predicate"].lower() for p in state})
    ambiguity = []
    if instruction.lower() != bddl_instruction.lower():
        ambiguity.append("Benchmark filename-derived instruction and BDDL :language differ; both originals are retained.")
    if language_only:
        ambiguity.append("Explicit instruction state action is not a required final-state goal predicate: " + ", ".join(language_only))
    if maintained:
        ambiguity.append("State goal is explicitly present in :init and no corresponding state-change instruction exists; treated as a preservation constraint, not a requested state change.")
    push_pull = sorted({verb for phrase in (instruction, bddl_instruction) for verb in ("push", "pull")
                        if re.search(r"\b" + verb + r"\b", phrase.lower())})
    if push_pull and placement:
        ambiguity.append("Push/pull wording describes object displacement; On/In goal classification does not assume a grasp or an articulated-state change.")
    if re.search(r"\b(?:put|place)\b.*\bin\b", instruction.lower()) and any(p["predicate"].lower() == "on" for p in placement):
        ambiguity.append("Filename-derived instruction uses 'in' while a final placement goal uses On; no finer containment/support label is inferred.")
    possible_instruction_state_targets = []
    if language_only:
        for p in placement:
            target = resolve(p["arguments"][1])
            if target["owner"] in fixtures and any(term in (target["type"] or "") for term in ("cabinet", "drawer", "stove", "microwave")):
                possible_instruction_state_targets.append(target)
    state_required = bool(state or language_actions)
    category = classify(state_required, placed, errors)
    explicit_state_entities = [resolve(p["arguments"][0]) for p in state]
    return {
        "suite": suite, "task": task, "task_order_index": 0, "task_name": name,
        "benchmark_instruction": instruction, "bddl_instruction": bddl_instruction,
        "bddl_path": str(path.resolve()), "bddl_sha256": sha(text.encode()),
        "goal_expression": sections[":goal"][0], "goal_predicates": predicates,
        "objects": objects, "fixtures": fixtures, "obj_of_interest": sections.get(":obj_of_interest", []),
        "placement_subjects": [resolve(obj) for obj in placed], "placement_object_count": len(placed),
        "placement_destinations": [resolve(p["arguments"][1]) for p in placement],
        "required_state_goal_predicates": state, "initially_declared_state_goal_constraints": maintained,
        "explicit_instruction_state_actions": actions,
        "instruction_state_actions_absent_from_final_goal": language_only,
        "state_control_entities_from_goal": explicit_state_entities,
        "possible_instruction_state_targets_from_placement": possible_instruction_state_targets,
        "instruction_target_resolution_note": "Candidate targets are inferred from the explicit placement destination, not added to the BDDL final goal; they are not used to alter the object count.",
        "manipulated_entities": sorted(set(placed) | {x["owner"] for x in explicit_state_entities + possible_instruction_state_targets if x["owner"]}),
        "push_pull_words": push_pull,
        "category": category, "category_label": LABELS[category],
        "classification_reason": {"state_action_required_by_goal_or_explicit_language": state_required,
                                  "distinct_On_In_goal_subjects": placed, "unsupported_or_ambiguous_structure": errors},
        "goal_only_category": classify(bool(state), placed, errors),
        "ambiguity_notes": ambiguity, "unclassified_reasons": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    metadata = args.root.resolve() / "LIBERO/libero/libero"
    source_records = []

    def read_metadata(relative):
        path = (metadata / relative).resolve()
        if not path.is_relative_to(metadata):
            raise ValueError("Metadata read escaped official LIBERO directory")
        raw = path.read_bytes()
        source_records.append({"path": str(path), "sha256": sha(raw), "bytes": len(raw)})
        return path, raw.decode()

    _, map_text = read_metadata("benchmark/libero_suite_task_map.py")
    _, benchmark_text = read_metadata("benchmark/__init__.py")
    task_map = literal_assignment(ast.parse(map_text), "libero_task_map")
    task_orders = literal_assignment(ast.parse(benchmark_text), "task_orders")
    assert task_orders[0] == list(range(10)), "Official task_order_index=0 changed"
    language_node = next(n for n in ast.parse(benchmark_text).body if isinstance(n, ast.FunctionDef) and n.name == "grab_language_from_filename")
    namespace = {}
    exec(compile(ast.Module(body=[language_node], type_ignores=[]), "official-language-function", "exec"), namespace)
    tasks = []
    for suite in SUITES:
        assert len(task_map[suite]) == 10
        for task, idx in enumerate(task_orders[0]):
            name = task_map[suite][idx]
            assert benchmark_language(name + ".bddl") == namespace["grab_language_from_filename"](name + ".bddl")
            path, text = read_metadata(Path("bddl_files") / suite / (name + ".bddl"))
            tasks.append(extract_task(suite, task, name, path, text))
    assert len(tasks) == 40 and len({(t["suite"], t["task"]) for t in tasks}) == 40
    counts = Counter(t["category"] for t in tasks)
    by_suite = {suite: {c: sum(t["suite"] == suite and t["category"] == c for t in tasks) for c in CLASSES} for suite in SUITES}
    rules = {
        "version": "libero40-explicit-command-and-goal-v1",
        "priority": ["unknown if unsupported or unresolved goal structure", "state plus placement", "state only", "multiple placement", "single placement"],
        "state_action": "Open/Close/Turnon/Turnoff goal predicates or literal open/close/turn on/turn off commands in either official instruction source.",
        "state_preservation": "A state goal literal already explicitly listed in :init is preservation when the instruction does not request that action; this is a declaration-based rule, not a simulation claim.",
        "placement_object_count": "Count distinct first arguments of positive On/In goal predicates that refer to declared :objects; destinations/receivers are not counted as moved objects.",
        "category_definitions": {"state_change_only": "State action required; no On/In placement subjects.",
                                 "state_change_and_object_placement": "State action required and at least one placement subject.",
                                 "multiple_object_placement": "No required state action; at least two placement subjects.",
                                 "single_object_placement": "No required state action; exactly one placement subject, including pushing an object to a goal region.",
                                 "unknown": "Unsupported predicate/logic, unresolved placement subject/target, or neither recognized state action nor placement."},
        "not_inferred": ["difficulty", "precision requirement", "contact phase", "grasp necessity", "path preference", "model success", "speedup"],
        "conflicts": "Keep both instruction sources and the complete BDDL goal. Language-only state commands and state-preservation predicates are explicit ambiguity notes; unknown is used when the categorical rule itself cannot be resolved.",
    }
    report = {
        "schema": "ease-libero40-semantic-taxonomy-v1", "status": "metadata_only_exploratory_taxonomy",
        "data_access": "Only official LIBERO benchmark source and 40 BDDL files were read by this extractor; no experiment outcomes, policy outputs, run manifests, or path comparison data are inputs.",
        "rules": rules, "counts": {c: counts[c] for c in CLASSES}, "counts_by_suite": by_suite,
        "class_labels": LABELS, "tasks": tasks, "sources": source_records,
        "extractor": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__).read_bytes())},
        "limitations": [
            "The current frozen primary selection rule remains suite-only; this taxonomy does not amend it.",
            "Any future comparison using these categories is exploratory unless independently preregistered before new evaluation outcomes.",
            "Metadata categories alone cannot establish complementary paths, adaptive-selection value, causal layer specialization, or speedup.",
            "Category sizes are unequal and strongly associated with suite identity; a future pooled category comparison must not silently attribute suite differences to semantics.",
            "Official BDDL :init is a declared initialization constraint, not a fresh simulator measurement after settling.",
        ],
        "reproduce": f"python3 {Path(__file__).resolve()} --root {args.root.resolve()} --output {args.output.resolve()}",
    }
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "taxonomy.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    columns = ["suite", "task", "task_name", "benchmark_instruction", "bddl_instruction", "category", "category_label",
               "placement_object_count", "placement_subjects", "manipulated_entities", "goal_predicates",
               "explicit_instruction_state_actions", "initially_declared_state_goal_constraints", "goal_only_category",
               "ambiguity_notes", "unclassified_reasons", "bddl_path", "bddl_sha256"]
    with (out / "taxonomy.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for task in tasks:
            writer.writerow({key: json.dumps(task[key], ensure_ascii=False, sort_keys=True) if isinstance(task[key], (list, dict)) else task[key] for key in columns})
    print(json.dumps({"tasks": len(tasks), "counts": report["counts"], "by_suite": by_suite,
                      "output": str(out), "sources_read": len(source_records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
