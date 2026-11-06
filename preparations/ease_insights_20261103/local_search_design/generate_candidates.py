"""Freeze proposals from archived init14 calibration only; never launch an experiment.

This module does not select a closed-loop pair or set its qualification rule.
"""
import argparse
from collections import defaultdict
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
METHODS = {"jacobian": "predicted_rms", "action_mse": "actual_rms", "cosine": "layer_cosine"}
GROUPS = {"translation", "rotation", "gripper"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fraction(value):
    return dict(numerator=value.numerator, denominator=value.denominator, decimal=float(value))


def legal(mask):
    return len(mask) == 6 and set(mask) <= set(range(1, 18)) and all(
        not {i, i + 1, i + 2} <= set(mask) for i in range(1, 16)
    )


def build(root=ROOT):
    root = Path(root).resolve()
    inputs = {
        "archived_layer_metrics": root / "results/idea1_v3_4_E2_E4_single/E2_requests.csv",
        "archived_case_split": root / "results/idea1_v3_4_E2_E4_single/selection.json",
        "archived_metrics_manifest": root / "results/idea1_v3_4_E2_E4_single/manifest.json",
        "archived_v12_candidates": root / "results/idea1_v3_4_E2_v12_candidates/selection.json",
    }
    split = json.loads(inputs["archived_case_split"].read_text())
    previous = json.loads(inputs["archived_v12_candidates"].read_text())
    source_manifest = json.loads(inputs["archived_metrics_manifest"].read_text())
    assert source_manifest["selection_sha256"] == sha(inputs["archived_case_split"])
    calibration = {int(row["id"]): row["case"] for row in split if row["e2"] and row["split"] == "calibration"}
    assert len(calibration) == 20
    assert {(c["suite"], c["task"], c["init"]) for c in calibration.values()} == {
        (s, t, 14) for s in SUITES for t in (0, 2, 4, 6, 8)
    }
    with inputs["archived_layer_metrics"].open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if int(r["id"]) in calibration and float(r["gate"]) == 0.0]
    assert len(rows) == 20 * 17 * 3
    cells = defaultdict(list)
    for row in rows:
        assert {"suite": row["suite"], "task": int(row["task"]), "init": int(row["init"])} == calibration[int(row["id"])]
        cells[(int(row["id"]), row["group"])].append(row)
    assert set(cells) == {(i, g) for i in calibration for g in GROUPS}
    ranked = defaultdict(lambda: defaultdict(list))
    tie_cells = []
    for key, cell in sorted(cells.items()):
        assert len(cell) == 17 and {int(r["layer"]) for r in cell} == set(range(1, 18))
        for method, metric in METHODS.items():
            if len({float(r[metric]) for r in cell}) != 17:
                tie_cells.append(dict(id=key[0], group=key[1], method=method))
            sign = -1 if method == "cosine" else 1
            ordered = sorted(cell, key=lambda r: (sign * float(r[metric]), int(r["layer"])))
            for rank, row in enumerate(ordered):
                layer = int(row["layer"])
                ranked[(row["suite"], method)][layer].append(rank)
                ranked[("all", method)][layer].append(rank)
    scores = {key: {layer: Fraction(sum(values), len(values)) for layer, values in layers.items()}
              for key, layers in ranked.items()}
    reproductions = []
    for method, old in previous["candidates"].items():
        score = sum(scores[("all", method)][layer] for layer in old["skip_layers_1based"])
        assert abs(float(score) - old["score_sum"]) < 1e-12
        reproductions.append(dict(method=method, skip=old["skip_layers_1based"],
                                  published_score=old["score_sum"], reconstructed_score=fraction(score), passed=True))
    anchor = set(previous["candidates"]["jacobian"]["skip_layers_1based"])
    excluded = {tuple(c["skip_layers_1based"]) for c in previous["candidates"].values()}
    all_neighbors, legal_neighbors, neighbors = [], [], []
    for restore in sorted(anchor):
        for newly_skip in sorted(set(range(1, 18)) - anchor):
            mask = sorted(anchor - {restore} | {newly_skip})
            row = dict(restore_layer=restore, newly_skipped_layer=newly_skip, skip=mask)
            all_neighbors.append(row)
            if legal(mask):
                legal_neighbors.append(row)
                if tuple(mask) not in excluded:
                    neighbors.append(row)
    assert (len(all_neighbors), len(legal_neighbors), len(neighbors)) == (66, 52, 51)

    def proxy(suite, mask):
        # Equal weights for both rank metrics, three action groups, and five tasks.
        return sum(scores[(suite, method)][layer] for method in ("jacobian", "action_mse") for layer in mask) / 2

    proposals = []
    rankings = {}
    for suite in SUITES:
        order = sorted(neighbors, key=lambda r: (proxy(suite, r["skip"]), tuple(r["skip"])))
        rankings[suite] = [dict(row, combined_rank_score=fraction(proxy(suite, row["skip"]))) for row in order]
        chosen = order[0]
        proposals.append(dict(id=f"fixed_v12_restore{chosen['restore_layer']}_skip{chosen['newly_skipped_layer']}",
            proposal_suite=suite, **chosen, combined_rank_score=fraction(proxy(suite, chosen["skip"])),
            anchor_combined_rank_score=fraction(proxy(suite, anchor)),
            global_jacobian_rank_score=fraction(sum(scores[("all", "jacobian")][l] for l in chosen["skip"])),
            global_action_mse_rank_score=fraction(sum(scores[("all", "action_mse")][l] for l in chosen["skip"]))))
    assert len({tuple(row["skip"]) for row in proposals}) == 4
    assert [row["skip"] for row in proposals] == [
        [2, 3, 5, 13, 16, 17], [2, 3, 5, 14, 15, 17],
        [2, 3, 11, 14, 16, 17], [2, 3, 5, 12, 16, 17],
    ]
    receipt = dict(
        schema="ease-vlm-local-search-historical-proposal-v1",
        proposal_only=True, gpu_launched=False, new_initial_state_outcomes_read=False,
        inputs={name: dict(path=str(path.relative_to(root)), sha256=sha(path)) for name, path in inputs.items()},
        generator=dict(path=str(Path(__file__).resolve().relative_to(root)), sha256=sha(__file__)),
        historical_scope=dict(init=14, suites=list(SUITES), tasks=[0, 2, 4, 6, 8],
            selected_observations=20, gate=0, action_groups=sorted(GROUPS), metric_rows=1020,
            case_ids=sorted(calibration)),
        rank_definition=dict(rank_origin=0, rank_unit="Within each historical observation and action group, among layers 1..17",
            directions={"jacobian": "predicted_rms ascending", "action_mse": "actual_rms ascending", "cosine": "layer_cosine descending"},
            metric_ties="ascending original one-based layer number", observed_tie_cells=tie_cells,
            aggregation="Mean layer rank over five task observations and three action groups within each suite; mask score is sum over six skipped layers",
            proposal_objective="Equal mean of jacobian and action_mse mask rank scores within each suite",
            neighbor_ties="lexicographically ascending sorted skip tuple", arithmetic="exact fractions after ordinal ranking"),
        prior_score_reproduction=reproductions,
        constraints=dict(anchor=sorted(anchor), original_one_based_layers=list(range(1, 18)), skip_count=6,
            retained_layers=12, depth=18, max_consecutive_skips=2, mutation="Restore exactly one anchor-skipped layer and skip exactly one previously retained layer",
            all_one_swap_count=66, legal_one_swap_count=52,
            excluded_prior_masks=[list(mask) for mask in sorted(excluded)], remaining_neighbor_count=51),
        proposal_rule="Select the lowest-scoring nonprevious neighbor for each suite using archived calibration only; no closed-loop result or new initial state is used",
        proposals=proposals, complete_neighbor_rankings=rankings,
        proposed_experiment_scope=dict(conditions="Four proposed neighbors plus Jac anchor plus Full", discovery_inits=[36, 37, 38],
            discovery_episodes=720, confirmation_inits=[39, 40, 41, 42, 43], confirmation_episodes_if_qualified=600,
            confirmation_qualification_rule="Not specified here; root must freeze a separate rule before discovery",
            rollout_manifest_created=False),
        limitations=[
            "Layer metrics come from historical Full-trajectory single-layer perturbations, not six-layer compensated-path outcomes.",
            "proposal_suite labels the calibration source of the proposal; it is not an observed task preference or a confirmatory direction.",
            "Reproducing three old score sums identifies a compatible ranking calculation; the original candidate-generation script was not located.",
            "The old candidate file's source_manifest_sha256 does not equal the current archived E2/E4 manifest file hash; this receipt preserves both files and does not assert that historical link is resolved.",
            "No pair selection, success threshold, stopping rule, or confirmatory hypothesis is set by this proposal generator.",
        ],
    )
    receipt["receipt_hash"] = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return receipt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("candidate_proposal_receipt.json"))
    parser.add_argument("--check", action="store_true", help="Verify existing receipt without writing")
    args = parser.parse_args()
    receipt = build()
    if args.check:
        assert json.loads(args.output.read_text()) == receipt, "Existing proposal receipt differs"
    else:
        if args.output.exists():
            assert json.loads(args.output.read_text()) == receipt, "Refusing to replace a different frozen receipt"
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as stream:
                json.dump(receipt, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
    print(json.dumps(dict(output=str(args.output), receipt_hash=receipt["receipt_hash"],
        candidates=[dict(id=r["id"], skip=r["skip"], proposal_suite=r["proposal_suite"]) for r in receipt["proposals"]],
        checked=args.check, gpu_launched=False), ensure_ascii=False))


if __name__ == "__main__":
    main()
