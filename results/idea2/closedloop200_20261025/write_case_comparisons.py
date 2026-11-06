"""Export casewise paired outcomes after the fixed study is complete (CPU only)."""
import csv
import json
from pathlib import Path

R = Path(__file__).resolve().parent
P = R.parents[2] / "reports/idea2_closedloop200_20261025"


def main():
    statistics = json.loads((P / "statistics.json").read_text())
    assert statistics["status"] == "complete" and statistics["received_rows"] == 800
    data = json.loads((R / "normalized_cases.json").read_text())
    rows = data["cases"] if isinstance(data, dict) else data
    indexed = {(r["configuration"], r["suite"], int(r["task"]), int(r["init"])): r for r in rows}
    assert len(indexed) == len(rows) == 800
    comparisons = [r for r in statistics["comparisons"] if r["scope"] == "overall"]
    outcomes = []
    for comparison in comparisons:
        reference, candidate = comparison["reference"], comparison["candidate"]
        group = []
        keys = sorted(k[1:] for k in indexed if k[0] == reference)
        for key in keys:
            a, b = indexed[(reference, *key)], indexed[(candidate, *key)]
            assert type(a["success"]) is bool and type(b["success"]) is bool
            outcome = ("both_success" if a["success"] else "candidate_only_success") if b["success"] else ("reference_only_success" if a["success"] else "both_failure")
            group.append(dict(reference=reference, candidate=candidate, suite=key[0], task=key[1], init=key[2],
                              reference_success=a["success"], candidate_success=b["success"], outcome=outcome,
                              reference_steps=a["steps"], candidate_steps=b["steps"],
                              reference_directory=a["directory"], candidate_directory=b["directory"]))
        assert len(group) == 200
        assert sum(r["outcome"] == "candidate_only_success" for r in group) == comparison["candidate_only_success_b"]
        assert sum(r["outcome"] == "reference_only_success" for r in group) == comparison["reference_only_success_c"]
        outcomes.extend(group)
    for name, selected in [("paired_case_outcomes.csv", outcomes),
                           ("discordant_cases.csv", [r for r in outcomes if r["reference_success"] != r["candidate_success"]])]:
        with (P / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(outcomes[0]))
            writer.writeheader()
            writer.writerows(selected)
    print(json.dumps(dict(status="passed", comparisons=len(comparisons), paired_rows=len(outcomes),
                          discordant_rows=sum(r["reference_success"] != r["candidate_success"] for r in outcomes))))


if __name__ == "__main__":
    main()
