"""Plot accepted discovery counts only; never read partial confirmation cases."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
AUDIT = ROOT / "reports/ease_insights_20261103/vlm_local_search_runtime_acceptance/full_discovery/audit.json"
AUDIT_SHA = "c560c53b188229144ea8da0719fead97f40e6c2c07166c6f5231adfec3eab6d6"
CONDITIONS = ("full_d10", "jac_anchor", "local_spatial", "local_object", "local_goal", "local_long")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def main():
    raw = AUDIT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == AUDIT_SHA
    audit = json.loads(raw)
    assert audit["status"] == "pass" and audit["stage"]["valid"] == 720
    assert audit["stage"]["missing_count"] == audit["stage"]["invalid_count"] == audit["stage"]["unexpected_count"] == 0
    # The accepted audit enumerates all policy failures; its strict census
    # certifies exactly 30 boolean outcomes per suite for every condition.
    rows, matrix = [], []
    for condition in CONDITIONS:
        assert audit["stage"]["per_condition"][condition] == 120
        failed = {(c["suite"], c["task"], c["init"]) for c in audit["episode_success_failures"][condition]}
        assert len(failed) == len(audit["episode_success_failures"][condition])
        assert all(s in SUITES and t in range(10) and i in (36, 37, 38) for s, t, i in failed)
        values = []
        for suite in SUITES:
            successes = 30 - sum(s == suite for s, _, _ in failed)
            values.append(successes)
            rows.append(dict(condition=condition, suite=suite, successes=successes, cases=30,
                             success_rate=successes / 30, stage="discovery_selected_not_replication"))
        matrix.append(values)
    matrix = np.asarray(matrix)
    assert matrix.sum() == 627
    assert matrix[1, 1] == 30 and matrix[5, 1] == 27
    assert matrix[5, 3] == 24 and matrix[1, 3] == 19
    csv_path = OUT / "discovery_success_counts.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"font.size": 11, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    im = ax.imshow(matrix / 30, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4), ["Spatial", "Object", "Goal", "Long"])
    ax.set_yticks(range(6), ["Full (D10)", "Jac (selected)", "local_spatial", "local_object", "local_goal", "local_long (selected)"])
    for i in range(6):
        for j in range(4):
            n = int(matrix[i, j])
            ax.text(j, i, f"{n}/30", ha="center", va="center",
                    color="white" if n / 30 > .64 else "#162b43", fontweight="bold")
    ax.set_title("Local VLM path search: accepted discovery outcomes", pad=15)
    ax.tick_params(axis="both", length=0, pad=9)
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=.045, pad=.04)
    cb.set_label("Success rate (30 cases per cell)")
    fig.text(.5, .025, "Exploratory selection data (init 36–38); independent init 39–43 confirmation is pending.\nEach suite uses the same tasks and initial states across conditions. No significance claim.",
             ha="center", va="bottom", fontsize=9, color="#39495b")
    fig.tight_layout(rect=(0, .12, 1, 1))
    paths = []
    for extension in ("png", "pdf", "svg"):
        path = OUT / f"discovery_success_counts.{extension}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    paths.append(csv_path)
    evidence = dict(scope="Accepted discovery only; no confirmation inputs read",
                    source_audit=str(AUDIT), source_audit_sha256=AUDIT_SHA,
                    success_counts=matrix.tolist(), conditions=CONDITIONS, suites=SUITES,
                    total_cases=720, total_successes=627,
                    source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    artifacts={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (OUT / "figure_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"success_counts": matrix.tolist(), "artifacts": [str(p) for p in paths]}))


if __name__ == "__main__":
    main()
