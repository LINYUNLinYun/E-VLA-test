"""Export the frozen EASE local-search heldout confirmation figures."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ACCEPTANCE = ROOT / "reports/ease_insights_20261103/vlm_local_search_runtime_acceptance/full_confirmation/audit.json"
ANALYSIS = ROOT / "results/ease_insights_20261103/vlm_local_search_v1/confirmation/analysis/analysis.json"
STATISTICS = ROOT / "reports/ease_insights_20261103/vlm_local_confirmation_statistics/audit.json"
CASE_ROOT = ROOT / "results/ease_insights_20261103/vlm_local_search_v1/confirmation"
MANIFEST_PATH = CASE_ROOT / "manifest.json"
MANIFEST_HASH = "80176d9f4b01e68abb022345bd6dc1a7bfe4e5aef5d69d13257801ca1eb6bdc0"
MANIFEST_SHA = "3c4b8bf35ecf768f129c00721ce0fd0e49ef2c209f7b9995ed0a1a128796365a"
CONDITIONS = ("full_d10", "jac_anchor", "local_long")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
DIRECTIONS = (("libero_object", "jac_anchor", "local_long"),
              ("libero_10", "local_long", "jac_anchor"))
SUITE_LABELS = {"libero_spatial": "Spatial", "libero_object": "Object",
                "libero_goal": "Goal", "libero_10": "Long"}
CONDITION_LABELS = {"full_d10": "Full", "jac_anchor": "Jac", "local_long": "local_long"}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bound(path, expected):
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    need(actual == expected, f"SHA256 mismatch: {path}")
    return json.loads(raw), {"path": str(Path(path).resolve()), "sha256": actual}


def gate(acceptance, analysis, statistics, acceptance_sha, analysis_sha, statistics_sha):
    stage = acceptance.get("stage", {})
    need(acceptance.get("revision") == "full-local-search-confirmation-read-only-acceptance-v1" and
         acceptance.get("status") == "pass" and acceptance.get("root") == str(CASE_ROOT) and
         acceptance.get("failed_checks") == [],
         "Independent 600-case acceptance must pass with zero issues")
    need(stage.get("passed") is True and stage.get("complete") is True and
         stage.get("expected") == stage.get("valid") == 600 and stage.get("scenes") == 200 and
         stage.get("planned_conditions") == list(CONDITIONS) and
         stage.get("per_condition") == {condition: 200 for condition in CONDITIONS},
         "Acceptance does not certify the frozen 600-case matrix")
    need(all(stage.get(key) == [] and stage.get(key + "_count") == 0
             for key in ("missing", "invalid", "unexpected")) and stage.get("reasons") == [],
         "Acceptance reports issues")
    need(acceptance.get("manifest_hash") == MANIFEST_HASH and
         acceptance.get("manifest_file_sha256") == MANIFEST_SHA,
         "Acceptance manifest identity differs")
    need(analysis.get("revision") == "ease-vlm-local-search-confirmation-analysis-v1" and
         analysis.get("complete") is True and analysis.get("expected_episodes") == 600 and
         analysis.get("valid_episodes") == 600 and analysis.get("manifest_hash") == MANIFEST_HASH and
         all(analysis.get(key) == [] for key in ("missing", "invalid", "unexpected")),
         "Supervisor analysis is incomplete or differs from the frozen manifest")
    evidence = analysis.get("evidence", {})
    need(Path(evidence.get("manifest", {}).get("path", "")).resolve() == MANIFEST_PATH.resolve() and
         evidence.get("manifest", {}).get("sha256") == MANIFEST_SHA,
         "Analysis does not link to the frozen manifest SHA")
    need(statistics.get("revision") == "independent-local-confirmation-statistics-v1" and
         statistics.get("status") == "pass" and statistics.get("differences") == [],
         "Independent statistics audit must pass with zero differences")
    sources = statistics.get("input_evidence", {})
    for label, expected_path, expected_sha in (
            ("acceptance", ACCEPTANCE, acceptance_sha),
            ("supervisor", ANALYSIS, analysis_sha)):
        item = sources.get(label, {})
        need(Path(item.get("path", "")).resolve() == expected_path.resolve() and
             item.get("sha256") == expected_sha,
             f"Statistics audit source link differs: {label}")
    need((statistics.get("independent_statistics") or {}).get("manifest_hash") == MANIFEST_HASH and
         (statistics.get("independent_statistics") or {}).get("complete") is True,
         "Independent statistics do not bind the complete frozen matrix")
    conclusion = analysis.get("conclusion")
    independent_conclusion = statistics["independent_statistics"].get("conclusion")
    need(conclusion in ("not_reproduced", "complementarity_reproduced") and
         independent_conclusion == conclusion,
         "Analysis and independent statistics conclusions differ")
    need(statistics.get("input_evidence", {}).get("acceptance", {}).get("sha256") == acceptance_sha and
         acceptance_sha == sha(ACCEPTANCE) and analysis_sha == sha(ANALYSIS) and
         statistics_sha == sha(STATISTICS), "A final source report changed after SHA pinning")
    return matrix_rows(analysis), direction_rows(analysis), conclusion


def matrix_rows(analysis):
    found = {}
    for row in analysis.get("condition_summaries", []):
        if row.get("level") == "suite":
            key = (row.get("condition"), row.get("suite"))
            need(key not in found, "Duplicate suite summary")
            found[key] = row
    expected = {(condition, suite) for condition in CONDITIONS for suite in SUITES}
    need(set(found) == expected, "Analysis does not contain the complete 3 by 4 suite matrix")
    rows = []
    for condition in CONDITIONS:
        for suite in SUITES:
            item = found[condition, suite]
            need(item.get("n") == item.get("expected_n") == 50 and
                 type(item.get("successes")) is int and 0 <= item["successes"] <= 50,
                 f"Invalid 50-case suite summary: {condition}/{suite}")
            rows.append(dict(condition=condition, suite=suite, successes=item["successes"],
                             n=50, success_rate=item["successes"] / 50))
    return rows


def direction_rows(analysis):
    rows = analysis.get("primary_directions", [])
    need(len(rows) == 2, "Analysis must contain both frozen directions")
    result = []
    for item, expected in zip(rows, DIRECTIONS):
        direction = item.get("frozen_direction", {})
        suite, better, worse = expected
        need((direction.get("suite"), direction.get("better"), direction.get("worse")) == expected and
             item.get("n_paired") == 50, "Primary contrast differs from the frozen paired direction")
        inference = item.get("inference", {})
        ci = inference.get("confidence_intervals", {}).get("task_cluster", {})
        estimate = item.get("success_difference")
        low, high = ci.get("low"), ci.get("high")
        p = inference.get("primary_holm2_p")
        need(all(isinstance(x, (int, float)) and math.isfinite(x) for x in (estimate, low, high, p)) and
             -1 <= low <= high <= 1 and -1 <= estimate <= 1 and 0 <= p <= 1 and
             type(inference.get("direction_reproduced")) is bool,
             f"Invalid direction estimate/inference: {suite}")
        result.append(dict(suite=suite, better=better, worse=worse, n_paired=50,
                           difference_pp=estimate * 100, ci_low_pp=low * 100, ci_high_pp=high * 100,
                           holm2_exact_signflip_p=p,
                           reproduced=inference["direction_reproduced"]))
    return result


def export(matrix, directions, sources, conclusion):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/ease_local_confirmation_mpl")
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42,
                                "font.family": "DejaVu Sans", "axes.spines.top": False,
                                "axes.spines.right": False})
    import matplotlib.pyplot as plt
    import numpy as np

    out = HERE / "exports"
    out.mkdir(parents=True, exist_ok=True)
    suites = list(SUITES)
    conditions = list(CONDITIONS)
    values = np.array([[next(r["successes"] for r in matrix
                              if r["condition"] == c and r["suite"] == s)
                        for s in suites] for c in conditions])
    fig, ax = plt.subplots(figsize=(10.2, 4.6), layout="constrained")
    image = ax.imshow(values, cmap="cividis", vmin=0, vmax=50, aspect="auto")
    ax.set_xticks(range(4), [SUITE_LABELS[s] for s in suites])
    ax.set_yticks(range(3), [CONDITION_LABELS[c] for c in conditions])
    ax.set_xlabel("Heldout suite (10 tasks × 5 initial states each)")
    ax.set_title(f"Heldout confirmation success matrix · init39–43\nOverall analysis conclusion: {conclusion}")
    for i in range(3):
        for j in range(4):
            ax.text(j, i, f"{values[i, j]} / 50", ha="center", va="center",
                    color="white" if values[i, j] < 27 else "black", fontsize=11, fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax, pad=.025, shrink=.9)
    colorbar.set_label("Successful episodes out of 50")
    fig.text(.01, -.02, f"Heldout init39–43; 10 task clusters per suite. Overall analysis conclusion: {conclusion}.", fontsize=9)
    save_figure(fig, out / "success_matrix")
    plt.close(fig)

    fig = plt.figure(figsize=(11.5, 4.2), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.35, 1))
    ax = fig.add_subplot(grid[0, 0])
    meta = fig.add_subplot(grid[0, 1]); meta.axis("off")
    y = np.arange(2)[::-1]
    for row, ypos in zip(directions, y):
        color = "#168c79" if row["reproduced"] else "#737b83"
        ax.hlines(ypos, row["ci_low_pp"], row["ci_high_pp"], color=color, linewidth=1.8)
        ax.vlines([row["ci_low_pp"], row["ci_high_pp"]], ypos - .08, ypos + .08,
                  color=color, linewidth=1.8)
        ax.plot(row["difference_pp"], ypos, "o", color=color, markersize=7)
    ax.axvline(0, color="#333333", linewidth=1, linestyle="--")
    bounds = [0.0] + [value for row in directions for value in
                      (row["difference_pp"], row["ci_low_pp"], row["ci_high_pp"])]
    low, high = min(bounds), max(bounds)
    padding = max(3.0, .15 * (high - low))
    ax.set_xlim(low - padding, high + padding)
    ax.set_ylim(-.6, 1.6)
    ax.set_yticks(y, [f"{SUITE_LABELS[r['suite']]}: {CONDITION_LABELS[r['better']]} > {CONDITION_LABELS[r['worse']]}"
                      for r in directions])
    ax.set_xlabel("Success difference (percentage points)")
    ax.set_title("Prespecified paired directions")
    meta.text(.02, .77, "Each row: 10 task clusters, 50 paired cases", fontsize=10, fontweight="bold")
    for ypos, row in zip((.55, .30), directions):
        status = "YES" if row["reproduced"] else "NO"
        meta.text(.02, ypos,
                  f"{SUITE_LABELS[row['suite']]}\n"
                  f"95% task-cluster CI: [{row['ci_low_pp']:.1f}, {row['ci_high_pp']:.1f}] pp\n"
                  f"Exact task sign-flip Holm-2 p: {row['holm2_exact_signflip_p']:.4g}\n"
                  f"Direction reproduced: {status}", fontsize=10, va="center",
                  color="#126d5d" if row["reproduced"] else "#4f555b")
    fig.suptitle(f"Heldout confirmation · init39–43 · overall analysis conclusion: {conclusion}")
    fig.text(.01, -.02, f"Existing task-cluster 95% bootstrap CI (pp); no new inference. Overall conclusion: {conclusion}.", fontsize=9)
    save_figure(fig, out / "direction_differences")
    plt.close(fig)

    write_csv(out / "success_matrix.csv", matrix,
              ("condition", "suite", "successes", "n", "success_rate"))
    write_csv(out / "direction_differences.csv", directions,
              ("suite", "better", "worse", "difference_pp", "ci_low_pp", "ci_high_pp",
               "n_paired", "holm2_exact_signflip_p", "reproduced"))
    output_names = sorted(p.name for p in out.iterdir() if p.is_file() and p.name != "sha256.json")
    report = dict(revision="ease-local-confirmation-figures-v1", manifest_hash=MANIFEST_HASH,
                  overall_conclusion=conclusion,
                  scope=dict(heldout_inits=list(range(39, 44)), task_clusters=10,
                             episodes_per_condition=200, matrix_cell_n=50), sources=sources,
                  plot_source=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
                  outputs={name: sha(out / name) for name in output_names},
                  output_sha_json_excludes_its_own_digest=True,
                  claims_excluded=["quality noninferiority", "exclusive-GPU acceleration", "runtime-adaptive benefit"])
    (out / "sha256.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(export_dir=str(out), output_files=len(output_names),
                          sha256_json_sha256=sha(out / "sha256.json")), sort_keys=True))


def save_figure(fig, stem):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix("." + ext), dpi=300 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance-sha256", required=True, help="SHA256 of final independent 600-case acceptance report")
    parser.add_argument("--analysis-sha256", required=True, help="SHA256 of final 600-case supervisor analysis.json")
    parser.add_argument("--statistics-sha256", required=True, help="SHA256 of final independent statistics audit")
    args = parser.parse_args(argv)
    for value in (args.acceptance_sha256, args.analysis_sha256, args.statistics_sha256):
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            parser.error("Report SHA arguments must be lowercase SHA256")
    acceptance, acceptance_source = load_bound(ACCEPTANCE, args.acceptance_sha256)
    analysis, analysis_source = load_bound(ANALYSIS, args.analysis_sha256)
    statistics, statistics_source = load_bound(STATISTICS, args.statistics_sha256)
    matrix, directions, conclusion = gate(acceptance, analysis, statistics, args.acceptance_sha256,
                                          args.analysis_sha256, args.statistics_sha256)
    export(matrix, directions, dict(acceptance=acceptance_source, analysis=analysis_source,
                                    independent_statistics=statistics_source), conclusion)


if __name__ == "__main__":
    main()
