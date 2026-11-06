"""Export one completed nsys capture and prove complete runtime stage ranges."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess

from results.idea2.batch_study_20261031.trace_analysis import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    receipt = json.loads((root / (args.name + ".execution.json")).read_text())
    if receipt.get("exit_code") != 0 or receipt.get("status") != "passed":
        raise RuntimeError("Only a successfully exited profile may be adopted")
    trace = root / (args.name + "_trace.nsys-rep")
    sql = root / (args.name + "_trace.sqlite")
    directory = root / (args.name + "_analysis")
    directory.mkdir(exist_ok=False)
    if not sql.is_file():
        with (directory / "export.log").open("x") as log:
            subprocess.run(["nsys", "export", "--type", "sqlite", "--output", str(sql), str(trace)],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
    prefix = directory / "projection"
    with (directory / "projection.log").open("x") as log:
        subprocess.run(["nsys", "stats", "--report", "nvtx_gpu_proj_trace", "--format", "csv",
                        "--output", str(prefix), str(sql)], check=True, stdout=log, stderr=subprocess.STDOUT)
    result = analyze(sql)
    projection = list(csv.DictReader(next(directory.glob("projection*.csv")).open()))
    counts = Counter(row["Name"].rsplit(":", 1)[-1] for row in projection)
    proofs = {(p["stage"], p["cpu_start_ns"], p["cpu_end_ns"]): p for p in result["replay_envelopes"]}
    verified = 0
    for row in projection:
        label = row["Name"].rsplit(":", 1)[-1]
        if label not in ("encoder", "vlm", "AE"): continue
        start = int(row["Orig Start (ns)"])
        end = start + int(row["Orig Duration (ns)"])
        proof = proofs[label, start, end]
        first = int(row["Projected Start (ns)"])
        last = first + int(row["Projected Duration (ns)"])
        if (first, last) != (proof["gpu_first_ns"], proof["gpu_last_ns"]):
            raise RuntimeError("Official projection endpoints do not match real GPU operations")
        verified += 1
    if any(counts[stage] != 8 for stage in ("encoder", "vlm", "AE")) or verified != 24:
        raise RuntimeError(f"Expected eight complete replay envelopes per stage: {counts}")
    result.update(full_replay_ranges_verified=verified, official_projected_counts=dict(counts),
                  trace=str(trace), trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
    (directory / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    with (directory / "kernel_groups.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["kernel_groups"][0]))
        writer.writeheader();writer.writerows(result["kernel_groups"])
    print(json.dumps({key: result[key] for key in ("counts", "real_kernels_total", "full_replay_ranges_verified",
        "ae_vlm_real_kernel_overlap_ms", "ae_encoder_real_kernel_overlap_ms", "actual_resources")}, indent=2))


if __name__ == "__main__": main()
