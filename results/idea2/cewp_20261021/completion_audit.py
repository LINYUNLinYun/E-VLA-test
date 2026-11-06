"""Read measured evidence, not queue intent, before declaring this study done."""
import hashlib
import csv
import json
from pathlib import Path
import time


root = Path(__file__).resolve().parent
workspace = root.parents[2]
checks = []


def check(name, paths, function):
    paths = [root/p for p in paths]
    missing = [str(p.relative_to(root)) for p in paths if not p.is_file()]
    row = dict(requirement=name, evidence=[str(p.relative_to(root)) for p in paths])
    if missing:
        row.update(passed=False, missing=missing)
    else:
        try:
            value = function(paths)
            row.update(passed=bool(value), detail=value)
        except Exception as error:
            row.update(passed=False, error=repr(error))
    checks.append(row)


def read(path):
    return json.loads(path.read_text())


def request_result(paths):
    data = read(paths[0])
    args = data["args"]
    assert args["warmup"] >= 10 and args["count"] >= 100 and args["continuous_count"] >= 100
    assert len(data["groups"]) == len(args["pairs"])+len(args["continuous"])
    assert all(g["correct"] is True for g in data["groups"])
    expected = 2*len(args["pairs"])*args["count"]+len(args["continuous"])*args["continuous_count"]
    assert len(data["rows"]) == expected
    for row in data["rows"]:
        assert row["correct"] is True and row["steady_recapture"] == 0
        assert row["pool"]["live_snapshots"] <= 2
        for error in ("normalized_error", "decoded_error"):
            assert row[error]["max_abs"] <= 1e-5 and row[error]["mae"] <= 1e-6
    assert data["tolerance"]
    assert "pi05=True" in data["identity"]["config"]
    return dict(outputs=len(data["rows"]), groups=len(data["groups"]), steps=args["steps"])


for name in ("d10_main", "d5_main", "d1_main"):
    check("Original eager/dual and Graph scheduling at "+name, [name+"/result.json"], request_result)
for steps in (10, 5, 1):
    for suffix in ("selective_all", "selective_vision_only", "selective_vlm_only", "hoist",
                   "wait_full", "vision_wait", "modulation", "vision_modulation_wait",
                   "vision_modulation_wait_repeat"):
        name = f"d{steps}_{suffix}"
        check("Independent/combined scheduling and repeated evidence: "+name, [name+"/result.json"], request_result)

    def isolated(paths):
        data = read(paths[0])
        assert data["args"]["warmup"] >= 10 and data["args"]["count"] >= 100
        assert len(data["groups"]) == 2 and all(g["correct"] is True for g in data["groups"])
        assert len(data["rows"]) == 400
        assert {r["phase"] for r in data["rows"]} == {"isolated_ae", "isolated_prefill"}
        assert all(r["correct"] is True for r in data["rows"])
        return dict(outputs=len(data["rows"]), groups=2)
    check(f"Isolated AE/Prefill, eager/Graph D{steps}", [f"d{steps}_isolated/result.json"], isolated)

    def preparation(paths):
        data = read(paths[0])
        assert data["passed"] is True and len(data["rows"]) == 700 and len(data["groups"]) == 5
        assert all(g["correct"] is True for g in data["groups"])
        assert data["kv_python_objects_alive_after_close"] == 0 and data["total_engines"] == 1
        assert data["graph_audit"]["kv_snapshots"] == 2
        assert data["graph_audit"]["total_private_pools"] == 10
        assert all(r["correct"] is True and r["steady_recapture"] == 0 for r in data["rows"])
        return dict(outputs=700, paired_ablation_groups=2, continuous_groups=3)
    check(f"True preparation off/on ABBA D{steps}", [f"d{steps}_preparation_pair_gpu3/result.json"], preparation)

    def native(paths):
        data = read(paths[0])
        assert data["mode"] == "steady" and len(data["rows"]) == 100
        assert data["args"]["warmup"] >= 10 and data["repeatability"]["passed"]
        assert data["native_provenance"]["compression_disabled"] is True
        assert data["native_provenance"]["vision_compression"] == {"mode":"off","keep_tokens":256}
        assert data["gpu_uuid"].removeprefix("GPU-") == "83fdf473-8633-29d3-4a42-e482cfdf0c25"
        assert data["selected_bank_indices"] == read(root/"d10_main/result.json")["selected_bank_indices"]
        assert data["bank_sha256"] == read(root/"d10_main/result.json")["bank_sha256"]
        return dict(outputs=100, boundary=data["boundary"], comparison=data["comparison"])
    check(f"Original JAX baseline on same primary GPU D{steps}", [f"native_d{steps}_gpu2/result.json"], native)

for name in ("lifecycle_gpu0", "lifecycle_wait_modulation_gpu0"):
    def lifecycle(paths):
        data = read(paths[0])
        assert data["passed"] is True
        assert all(all(r["checks"].values()) for r in data["results"])
        return dict(configs=len(data["results"]), checks=[r["checks"] for r in data["results"]])
    check("In-place updates, snapshots, reset and exception ownership: "+name, [name+"/result.json"], lifecycle)

def traces(paths):
    rows = read(paths[0])
    assert len(rows) >= 27
    assert all(r["matched"] == r["total_kernels"] for r in rows)
    assert all(r["device_sync"] == 0 and r["capture_begin"] == 0 for r in rows)
    assert all("global_measured_window" in r for r in rows)
    plans = {r["plan"] for r in rows}
    assert {"profile_d10_full", "profile_d10_vision_wait_modulation", "profile_d1_vision_wait", "profile_d5_vision_wait"} <= plans
    return dict(traces=len(rows), plans=sorted(plans), overlap_scope="Kernel interval intersection, including cross-request overlap")
check("Real kernel overlap and runtime sync/capture audit", ["request_trace_summary.json"], traces)
check("Stage pairing and independent repeat", ["stage_delivery.json"],
      lambda paths: read(paths[0])["passed"] and all(read(paths[0])["checks"].values()))

def kv_ready_tail(paths):
    data = read(paths[0])
    rows = list(csv.DictReader(paths[1].open()))
    assert data["status"] == "complete" and data["measurements"] == len(rows) == 100
    assert data["warmup"] >= 10 and data["semantics"]["passed"]
    assert data["static_addresses_unchanged"] and data["steady_recaptures"] == 0
    assert data["capture"]["last_update_output_addresses_equal_final_cache"]
    assert data["capture"]["update_restored"] and data["max_complete_kv_snapshots"] <= 2
    assert len(data["validation"]) == data["dynamic_inputs"] == 13
    assert all(v["kv_digest_equal"] for v in data["validation"])
    assert hashlib.sha256(paths[2].read_bytes()).hexdigest() == data["source_sha256"]
    return dict(measurements=100, tail=data["summary"]["kv_ready_to_end_ms"], scope=data["scope"])
check("Full old-KV ready tail quantified without changing publication", ["kv_ready_tail/result.json", "kv_ready_tail/measurements.csv", "kv_ready_tail/loaded_source.py"], kv_ready_tail)
check("Production RPC lifecycle", ["server_rpc_gpu0/receipt.json"],
      lambda paths: read(paths[0])["passed"] and read(paths[0])["request_count"] == 13
      and read(paths[0])["steady_recaptures"] == 0 and read(paths[0])["process_reaped"])

def code_review(paths):
    data=read(paths[0])
    assert data["cpu_tests"]["passed"] and data["cpu_tests"]["count"] == 41
    assert all(i["passed"] for i in data["invariants"])
    for name, expected in data["source_sha256"].items():
        assert hashlib.sha256((workspace/name).read_bytes()).hexdigest()==expected, name
    assert hashlib.sha256(paths[1].read_bytes()).hexdigest()==data["cpu_tests"]["log_sha256"]
    return dict(tests=41, invariant_count=len(data["invariants"]), reviewed_source_matches=True)
check("Final production ownership/error recovery and actual CPU regressions", ["final_code_review.json","final_review_cpu_tests.log"], code_review)

def reference(paths):
    data=read(paths[0])
    assert set(data)=={"joint.py","graphs.py","preparation.py","blocks.py"}
    for name, item in data.items():
        assert "/2c4d69e1dea643fbd162e612254967ba1eb27784/" in item["url"]
        assert hashlib.sha256((root/"reference"/name).read_bytes()).hexdigest()==item["sha256"]
    return dict(fixed_commit="2c4d69e1dea643fbd162e612254967ba1eb27784", files=sorted(data))
check("Requested pinned RealtimeWAM source inspection", ["reference/sources.json","reference/source_review.txt"],reference)
check("Native baseline does not enable pre-existing model experiments", ["native_source_audit.json","native_provenance_cpu_receipt.json"],
      lambda paths: read(paths[0])["passed"] and read(paths[0])["pi0_ast_equals_head"]
      and read(paths[0])["siglip_off_jaxpr_text_equals_head"] and read(paths[1])["passed"] and read(paths[1])["nonmutating_check"])

for name in ("compile_smoke", "compile_graph_smoke", "compile_d1_only_gate", "compile_d1_graph_gate",
             "compile_d5_only_gate", "compile_d5_graph_gate"):
    def compile_gate(paths):
        data=read(paths[0])
        assert data["args"]["compile"]
        assert any(g["correct"] is False for g in data["groups"])
        assert any(r.get("normalized_error",{}).get("max_abs",0)>1e-5 for r in data["rows"])
        return dict(rejected=True, steps=data["args"]["steps"])
    check("Numerically invalid compile candidate retained, excluded: "+name,[name+"/result.json"],compile_gate)

def final_source(paths):
    files=("policy.py","runtime_graph.py","graph_sampler.py","cewp_segmented.py","cewp_preparation.py",
           "scheduler.py","submission.py","openpi_backend.py")
    data=read(paths[0])
    for name in files:
        path=workspace/"preparations/idea2"/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==data["source_sha256"][name], name
    return dict(final_production_files=list(files), source_matches_final_repeat=True)
check("Final implementation actually exercised by critical repeat", ["d10_vision_modulation_wait_repeat/result.json"], final_source)

def report(paths):
    data=read(paths[0])
    assert not data["read_errors"] and not data["incomplete_expected"]
    assert all(value=="complete" for value in data["native_expected_status"].values())
    assert data["best_candidates"]
    primary=[r for r in data["best_candidates"] if r["gpu_uuid"].removeprefix("GPU-")=="83fdf473-8633-29d3-4a42-e482cfdf0c25"]
    assert {r["steps"] for r in primary}=={1,5,10}
    assert all(r["lowest_response_preserving_freshness"]["freshness_gate"] for r in primary)
    return dict(primary_best=primary)
check("All requested cohorts and conservative freshness-gated ranking",["summary.json","comparison.csv","paired_comparison.csv","native_fresh_comparison.csv"],report)

result=dict(updated_unix=time.time(),passed=all(c["passed"] for c in checks),checks=checks,
            note="Automated evidence audit is necessary, not sufficient: inspect final report, commands, visualizations and causal claims manually.")
(root/"completion_audit.json").write_text(json.dumps(result,indent=2))
print(json.dumps(dict(passed=result["passed"],checks=len(checks),failed=[c["requirement"] for c in checks if not c["passed"]]),indent=2))
