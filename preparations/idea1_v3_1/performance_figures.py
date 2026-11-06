"""Plot authenticated saved timing findings; never rerun a raw audit/experiment."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .common import ROOT,atomic_json,read_manifest
from .same_process_parity import sha

MODELS = ("pi0","pi05","xiaomi-robotic0","db-cogact")
ARMS = ("full_io","lean_io","reuse_env","two_workers")
TIMERS = ("request_roundtrip_ms","server_request_ms","vision_ms")
DISPLAY = dict(pi0="π0",pi05="π0.5",**{"xiaomi-robotic0":"Xiaomi","db-cogact":"CogACT"})
SCOPE = "Saved audit/receipt/benchmark JSON hashes and arithmetic checked; NOT a new raw audit. Timing only: no completed-200 quality requirement or same-quality gain claim."


def equal(a,b):
    assert math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12),f"Source/ratio mismatch: {a} vs {b}"


def load(path,inputs,expected=None):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT/path
    actual = sha(path)
    if expected is not None:
        assert actual==expected,f"Referenced saved artifact SHA mismatch: {path}"
    inputs[str(path.absolute())] = actual
    return json.loads(path.read_text())


def e2_values(findings,manifest,inputs):
    assert findings.get("manifest_hash",manifest["manifest_hash"])==manifest["manifest_hash"]
    assert {r["model"] for r in findings["models"]}==set(MODELS) and len(findings["models"])==4
    rows,increments,proofs = [],[],[]
    for model in MODELS:
        record = next(r for r in findings["models"] if r["model"]==model)
        audit = load(record["audit_path"],inputs,record["audit_sha256"])
        assert audit["passed"] and audit["video_verified"] and audit["manifest_hash"]==manifest["manifest_hash"]
        assert audit["counts"]=={arm:8 for arm in ARMS} and audit["summary"]["complete_episodes"]==32
        assert audit["summary"]["fixed_controls_pass"] and audit["validation_receipt_sha256"]==record["validation_sha256"]
        equal(audit["fixed_control_max_abs"],record["fixed_control_max_abs"])
        assert 0<=audit["fixed_control_max_abs"]<=1e-5
        validation = load(audit["validation_receipt"],inputs,record["validation_sha256"])
        assert validation["manifest_hash"]==manifest["manifest_hash"] and validation["condition"]==model+"_full"
        assert validation["status"]=="completed" and validation["server_exit_code"]==0 and validation["warmup"]==20
        assert validation["protocol"]["server_pid"]==validation["server_pid"] and validation["protocol"]["condition_id"]==model+"_full"
        assert [r["variant"] for r in validation["arms"]]==list(ARMS)
        walls = {r["variant"]:r["wall_s"] for r in validation["arms"]}
        assert set(record["eight_episode_wall_s"])==set(record["speedup_vs_full_io"])==set(ARMS)
        evidence = {r["variant"]:r for r in audit["adoption_evidence"]}
        for arm in ARMS:
            assert math.isfinite(walls[arm]) and walls[arm]>0
            equal(walls[arm],record["eight_episode_wall_s"][arm])
            speedup = walls["full_io"]/walls[arm]
            equal(speedup,record["speedup_vs_full_io"][arm])
            equal(speedup,audit["speedups"][arm])
            if arm!="full_io":
                assert 0<=evidence[arm]["matched_scenes"]<=8
                assert evidence[arm]["matched_requests_total"]==record["paired_requests"][arm]>=0
            rows.append(dict(model=model,arm=arm,episodes=8,eight_episode_wall_s=walls[arm],
                aggregate_episodes_per_hour=8*3600/walls[arm],speedup_vs_full_io=speedup,
                audit_path=record["audit_path"],audit_sha256=record["audit_sha256"],validation_sha256=record["validation_sha256"]))
        lean,reuse = walls["lean_io"]/walls["two_workers"],walls["reuse_env"]/walls["two_workers"]
        equal(lean,record["two_workers_speedup_vs_lean_io"])
        equal(reuse,record["two_workers_speedup_vs_reuse_env"])
        equal(1-walls["lean_io"]/walls["full_io"],record["lean_io_time_reduction_fraction"])
        increments.append(dict(model=model,two_workers_vs_lean_io=lean,two_workers_vs_reuse_env=reuse,
            lean_io_wall_s=walls["lean_io"],reuse_env_wall_s=walls["reuse_env"],two_workers_wall_s=walls["two_workers"]))
        proofs.append(dict(model=model,audit_path=record["audit_path"],audit_sha256=record["audit_sha256"],
            inherited_passed_definition=audit.get("passed_definition"),paired_evidence_supports_consideration={a:evidence[a]["evidence_supports_consideration"] for a in ARMS[1:]},
            gpu=validation.get("gpu_identity",{"assigned":validation.get("gpu"),"uuid":"not retained in legacy receipt"})))
    return rows,increments,proofs


def vision_values(findings,manifest,inputs):
    assert findings["manifest_hash"]==manifest["manifest_hash"]
    assert {r["model"] for r in findings["models"]}=={"pi0","pi05"} and len(findings["models"])==2
    definitions = {c["id"]:c for c in manifest["conditions"]+manifest["benchmark_references"]}
    rows,proofs = [],[]
    for record in findings["models"]:
        model = record["model"]
        audit = load(record["source"],inputs,record["source_sha256"])
        assert audit["passed"] and isinstance(record["stable"],bool)
        assert audit["stable_for_primary_comparison"]==record["stable"]
        assert audit["manifest_hash"]==manifest["manifest_hash"] and audit["gpu_identity"]==record["gpu"]
        assert record["gpu"]["uuid"].startswith("GPU-")
        assert audit["final_vision_cohort"] and audit["vision_budget_coverage"]==dict(observed=[3,6,9,12],required=[3,6,9,12],complete=True)
        assert record["comparison"]==audit["comparison"] and record["comparison"]["unstable"] is (not record["stable"])
        comparison = record["comparison"]
        assert comparison["primary_statistic"]=="P50" and comparison["primary_drift_metrics"]==["request_roundtrip_ms","vision_ms"]
        cohort = load(Path(audit["cohort_path"])/"cohort.json",inputs,audit["cohort_receipt_sha256"])
        assert cohort["manifest_hash"]==manifest["manifest_hash"] and cohort["model"]==model
        assert cohort["status"]=="completed" and cohort["require_four_budgets"] and cohort["gpu_identity"]==record["gpu"]
        phases = audit["phases"]
        assert [p["role"] for p in phases]==["native_full_before","matched_all27_full"]+["candidate"]*4+["native_full_after"]
        assert phases[0]["condition"]==phases[-1]["condition"]==model+"_full"
        assert phases[1]["condition"]==model+"_full_vision_scan_reference"
        assert len(cohort["phases"])==len(phases)
        summaries = {}
        for phase,execution in zip(phases,cohort["phases"],strict=True):
            assert phase["condition"]==execution["condition"] and phase["role"]==execution["role"]
            assert execution["artifact_hashes"]["benchmark.json"]==phase["artifacts"]["benchmark.json"]
            assert execution["gpu_before"]["uuid"]==execution["gpu_after"]["uuid"]==record["gpu"]["uuid"]
            benchmark = load(phase["benchmark_path"],inputs,phase["artifacts"]["benchmark.json"])
            assert benchmark["manifest_hash"]==manifest["manifest_hash"] and benchmark["condition"]==definitions[phase["condition"]]
            assert benchmark["banks"]==audit["banks"] and benchmark["warmup"]==20 and benchmark["samples"]==100 and len(benchmark["rows"])==100
            assert benchmark["summary"]==phase["summary"]
            for metric in TIMERS:
                equal(np.percentile([r[metric] for r in benchmark["rows"]],50),phase["summary"][metric]["p50"])
            key = phase["condition"] if phase["role"]=="candidate" else phase["role"]
            assert key not in summaries
            summaries[key] = phase["summary"]
        assert record["native_before"]==summaries["native_full_before"] and record["native_after"]==summaries["native_full_after"]
        candidates = comparison["candidates"]
        assert len(candidates)==4 and {c["condition"] for c in candidates}=={p["condition"] for p in phases if p["role"]=="candidate"}
        assert sorted(len(definitions[c["condition"]]["skip"]) for c in candidates)==[3,6,9,12]
        for metric in TIMERS:
            before,after = record["native_before"][metric]["p50"],record["native_after"][metric]["p50"]
            drift = abs(after/before-1)
            equal(drift,comparison["native_full_after_p50_drift_fraction"][metric])
            assert comparison["drift_exceeds_threshold_by_metric"][metric]==(drift>comparison["drift_threshold"])
        assert record["stable"]==(not any(comparison["drift_exceeds_threshold_by_metric"][m] for m in comparison["primary_drift_metrics"]))
        for candidate in candidates:
            cid = candidate["condition"]
            condition = definitions[cid]
            assert condition["model"]==model and condition["module"]=="vision" and not condition["reference"]
            for metric in TIMERS:
                before = record["native_before"][metric]["p50"]
                after = record["native_after"][metric]["p50"]
                matched = summaries["matched_all27_full"][metric]["p50"]
                actual = summaries[cid][metric]["p50"]
                assert min(before,after,matched,actual)>0
                equal(before/actual,candidate["native_before_p50_speedup"][metric])
                equal(matched/actual,candidate["matched_full_p50_speedup"][metric])
                rows.append(dict(model=model,condition=cid,skip=len(condition["skip"]),keep=27-len(condition["skip"]),metric=metric,
                    native_before_p50_ms=before,native_after_p50_ms=after,matched_full_p50_ms=matched,candidate_p50_ms=actual,
                    native_before_p50_speedup=before/actual,matched_full_p50_speedup=matched/actual,
                    gpu_uuid=record["gpu"]["uuid"],stable=record["stable"],audit_path=record["source"],audit_sha256=record["source_sha256"]))
        proofs.append(dict(model=model,audit_path=record["source"],audit_sha256=record["source_sha256"],gpu=record["gpu"],stable=record["stable"],
            inherited_passed_definition=audit.get("passed_definition")))
    return rows,proofs


def csv_file(path,rows):
    with path.open("x",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grouped(ax,series,labels,title,ylabel):
    positions = np.arange(len(labels))
    width = .72/len(series)
    for index,(name,values) in enumerate(series):
        bars = ax.bar(positions+(index-(len(series)-1)/2)*width,values,width,label=name)
        ax.bar_label(bars,labels=[f"{v:.4f}×" for v in values],fontsize=8,padding=3,rotation=60)
    ax.axhline(1,color="black",linewidth=.8,linestyle="--")
    ax.set_xticks(positions,labels,fontsize=9)
    ax.set_title(title,fontsize=10,pad=33)
    ax.set_ylabel(ylabel,fontsize=9)
    ax.set_ylim(0,max(v for _,values in series for v in values)*1.22)
    ax.legend(fontsize=8,loc="lower center",bbox_to_anchor=(.5,1.015),ncol=min(3,len(series)),frameon=False)


def build(e2_path,vision_path,manifest,output,manifest_path=None):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Choose a NEW immutable performance-figure directory")
    inputs = {}
    if manifest_path:
        inputs[str(Path(manifest_path).absolute())] = sha(manifest_path)
    e2 = load(e2_path,inputs)
    vision = load(vision_path,inputs)
    e2_rows,increments,e2_proofs = e2_values(e2,manifest,inputs)
    vision_rows,vision_proofs = vision_values(vision,manifest,inputs)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True)
    csv_file(output/"e2_raw_and_ratios.csv",e2_rows)
    csv_file(output/"e2_two_worker_increment.csv",increments)
    csv_file(output/"vision_native_and_matched_raw_ratios.csv",vision_rows)
    fig,axes = plt.subplots(1,2,figsize=(15,5.5))
    series = [(arm,[next(r["speedup_vs_full_io"] for r in e2_rows if r["model"]==m and r["arm"]==arm) for m in MODELS]) for arm in ARMS[1:]]
    grouped(axes[0],series,[DISPLAY[m] for m in MODELS],"Eight-episode aggregate throughput / full_io","Aggregate episode-throughput ratio")
    grouped(axes[1],[("two_workers / lean_io",[r["two_workers_vs_lean_io"] for r in increments]),
        ("two_workers / reuse_env",[r["two_workers_vs_reuse_env"] for r in increments])],
        [DISPLAY[m] for m in MODELS],"Additional aggregate throughput with two workers","Aggregate episode-throughput ratio")
    fig.suptitle("E2: fixed eight official episodes per arm, one persistent native model service",fontsize=12)
    fig.text(.02,.025,"Aggregate environment/RPC/I/O throughput; NOT single-request acceleration or model batching. Fixed arm order; common capture overhead.\nTiming profiles only; no same-quality gain claim. Labels rounded to 4 decimals; CSV preserves full precision.",fontsize=8)
    fig.tight_layout(rect=(0,.1,1,.93))
    fig.savefig(output/"e2_throughput.png",dpi=180)
    plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(13,5.5))
    for ax,metric,title in zip(axes,("request_roundtrip_ms","vision_ms"),("Completed RPC P50 ratio","Standalone all-camera vision P50 ratio")):
        series = [(DISPLAY[m]+(" [UNSTABLE]" if not next(r["stable"] for r in vision_rows if r["model"]==m) else ""),[next(r["native_before_p50_speedup"] for r in vision_rows if r["model"]==m and r["skip"]==budget and r["metric"]==metric) for budget in (3,6,9,12)]) for m in ("pi0","pi05")]
        grouped(ax,series,[f"keep {27-b}\n(skip {b})" for b in (3,6,9,12)],title,"Native Full-before P50 / candidate P50")
    unstable = sorted({r["model"] for r in vision_rows if not r["stable"]})
    fig.suptitle("Vision timing cohorts: same-GPU native Full-before baseline"+(" — UNSTABLE; descriptive ratios only" if unstable else " (stable)"),fontsize=11)
    fig.text(.02,.025,"Whole RPC and standalone vision have INDEPENDENT timers: never add/subtract them or infer component attribution.\nMatched all-27 reference ratios are in CSV only. Timing only; quality confirmation is separate. Labels rounded to 4 decimals; full precision in CSV.",fontsize=8)
    fig.tight_layout(rect=(0,.1,1,.93))
    fig.savefig(output/"vision_native_p50.png",dpi=180)
    plt.close(fig)
    result = dict(manifest_hash=manifest["manifest_hash"],source_sha256=sha(__file__),input_sources_sha256=inputs,
        audit_references=e2_proofs+vision_proofs,validation_scope=SCOPE,
        e2_scope=e2.get("scope"),inherited_vision_scope=vision.get("scientific_scope"),
        unstable_cohorts=unstable,normal_gain_conclusion_authorized=False,
        interpretation="E2 aggregate episode throughput is not single-request speed or batching; vision/RPC independent P50 ratios are not additive; unstable cohorts retained and flagged, never normal gain evidence; no same-quality gain claim.",
        artifacts_sha256={p.name:sha(p) for p in sorted(output.iterdir())})
    atomic_json(output/"plot_index.json",result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--e2-findings",required=True)
    parser.add_argument("--performance-findings",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    result = build(args.e2_findings,args.performance_findings,read_manifest(args.manifest),args.output,args.manifest)
    print(json.dumps(dict(output=args.output,artifacts=len(result["artifacts_sha256"]),validation_scope=SCOPE)))


if __name__=="__main__":
    main()
