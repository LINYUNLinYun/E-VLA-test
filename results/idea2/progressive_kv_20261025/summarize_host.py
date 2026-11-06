"""Export Host diagnostics from the passed formal runs, without GPU access."""
import csv
import json
import statistics
from pathlib import Path

R = Path(__file__).resolve().parent
P = R.parents[2] / "reports/idea2_progressive_kv_20261025/host_diagnostics.csv"
CASES = [
    "formal_same_task_gpu2", "repeat_same_task_gpu2",
    "gpu0_full_control_formal", "gpu0_full_control_formal_repeat",
    "gpu0_sm16_lead3_formal", "gpu0_sm16_lead3_repeat",
]


def main():
    output = []
    for case in CASES:
        data = json.loads((R / case / "result.json").read_text())
        if data["status"] != "passed" or not data["source_unchanged"]:
            raise ValueError(f"Invalid formal artifact: {case}")
        groups = {}
        for row in data["rows"]:
            if row["phase"] == "paired":
                groups.setdefault((row["lead"], row["variant"]), []).append(row)
        for (lead, variant), rows in groups.items():
            value = dict(
                case=case, lead=lead, variant=variant, count=len(rows),
                graph_recaptures_max=max(x["graph_recaptures"] for x in rows),
                prefix_done_at_response_fraction=statistics.mean(
                    x["prefix_done_at_response"] for x in rows),
            )
            for key in ["ae_submit_ms", "lookahead_prepare_ms", "input_prepare_ms", "decode_host_ms"]:
                samples = [x[key] for x in rows if x.get(key) is not None]
                value[key + "_mean"] = statistics.mean(samples) if samples else None
            output.append(value)
    with P.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    main()
