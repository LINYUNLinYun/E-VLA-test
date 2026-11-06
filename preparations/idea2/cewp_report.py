"""Read-only CEWP evidence aggregation; generated tables never alter experiments."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np


METRICS = ("cpu_cpu_ms", "ae_output_ms", "prefill_completion_event_ms",
           "both_completion_event_ms", "ae_gpu_event_ms", "prefill_gpu_event_ms",
           "ae_cpu_submit_ms", "prefill_cpu_submit_ms", "ae_wait_after_submit_ms",
           "observation_age", "source_age_at_response_ms", "previous_prefix_wait_ms")


def variant_tags(args):
    tags=[]
    if args.get("compile"):
        tags.append("compile")
    if args.get("hoist_modulation"):
        tags.append("hoist_modulation_with_schedule")
    elif args.get("hoist_schedule"):
        tags.append("hoist_schedule")
    if args.get("selective") and args["selective"]!="base":
        tags.append("selective_"+args["selective"])
    if args.get("await_previous_prefix"):
        tags.append("await_previous_prefix")
    return tags or ["base"]


def stats(values):
    values=[x for x in values if x is not None]
    if not values:
        return dict(n=0,mean=None,p50=None,p95=None,max=None)
    return dict(n=len(values),mean=float(np.mean(values)),p50=float(np.percentile(values,50)),
                p95=float(np.percentile(values,95)),max=float(max(values)))


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def csv_write(path, rows):
    fields=sorted(set().union(*(r.keys() for r in rows))) if rows else ["status"]
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})


def grouped_rows(result):
    """Old snapshots lacked row.group; their complete groups were appended in order."""
    offset=0
    args=result["args"]
    for number,group in enumerate(result.get("groups",[])):
        if result["rows"] and "group" in result["rows"][0]:
            rows=[r for r in result["rows"] if r["group"]==number]
        else:
            count=(2*args["count"] if group["phase"] in ("paired","isolated") else args["continuous_count"])
            rows=result["rows"][offset:offset+count]
            offset+=count
        yield number,group,rows


def request_run(path, result):
    args=result["args"]
    groups=list(grouped_rows(result))
    expected=(len(args.get("continuous",[])) or 2) if args.get("kind")=="isolated" else len(args.get("pairs",[]))+len(args.get("continuous",[]))
    numeric_failed=any(g.get("correct") is False for _,g,_ in groups)
    complete=len(groups)==expected and all(g.get("correct") is True for _,g,_ in groups)
    category=("compile_failed" if args.get("compile") and numeric_failed else
              "profile" if args.get("profile") else
              "smoke" if "smoke" in path.parent.name or args.get("warmup",0)<10 or args.get("count",0)<100 else
              "isolated" if args.get("kind")=="isolated" else "main")
    status="numerical_failure" if numeric_failed else "complete" if complete else "incomplete"
    run=dict(run=path.parent.name,path=str(path),category=category,status=status,
             completed_groups=len(groups),expected_groups=expected,steps=args.get("steps"),
             source_sha256=result.get("source_sha256"),gpu_uuid=result.get("gpu_uuid"))
    output=[]
    paired=[]
    for group_index,group,rows in groups:
        for phase,mode in sorted({(r["phase"],r["mode"]) for r in rows}):
            selected=[r for r in rows if (r["phase"],r["mode"])==(phase,mode)]
            arm_args=dict(args)
            preparation_values={r.get("preparation_variant") for r in selected}
            preparation_variant=next(iter(preparation_values)) if len(preparation_values)==1 else None
            if preparation_variant in ("baseline","schedule","modulation"):
                arm_args["hoist_schedule"]=preparation_variant=="schedule"
                arm_args["hoist_modulation"]=preparation_variant=="modulation"
            tags=variant_tags(arm_args)
            variant="+".join(tags)
            sequence=sorted((r.get("pair") if phase=="paired" else i,r["index"],r["seed"]) for i,r in enumerate(selected))
            seeds_ok=all(r["seed"]==1701+(r["pair"] if phase=="paired" else i) for i,r in enumerate(selected))
            index_ok=all(r["index"]==((2*r["pair"]+1 if phase=="paired" else i)%len(result["selected_bank_indices"])) for i,r in enumerate(selected))
            tolerance=result.get("tolerance",dict(max_abs=1e-5,mae=1e-6))
            numerical=all(r.get("correct") is True and
                (phase=="isolated_prefill" or all(
                    error in r and r[error].get("max_abs",float("inf"))<=tolerance["max_abs"] and
                    r[error].get("mae",float("inf"))<=tolerance["mae"]
                    for error in ("normalized_error","decoded_error"))) for r in selected)
            recaptures=sum(r.get("steady_recapture",0) for r in selected)
            expected_count=args["count"] if phase!="continuous" else args["continuous_count"]
            row=dict(run=path.parent.name,group=group_index,category=category,run_status=status,
                steps=args["steps"],variant=variant,mode=mode,phase=phase,n=len(selected),
                preparation_variant=preparation_variant,experiment=result.get("experiment","scheduling"),
                variant_tags=tags,await_previous_prefix=bool(args.get("await_previous_prefix")),
                compile=bool(args.get("compile")),hoist_schedule=bool(arm_args.get("hoist_schedule")),
                hoist_modulation=bool(arm_args.get("hoist_modulation")),
                schedule_hoist_effective=bool(arm_args.get("hoist_schedule") or arm_args.get("hoist_modulation")),
                preparation_ablation="modulation_plus_schedule" if arm_args.get("hoist_modulation") else
                                     "schedule_only" if arm_args.get("hoist_schedule") else "none",
                selective=args.get("selective"),
                execution="graph" if mode.startswith("graph") else "eager",
                priority=-1 if "high" in mode else 0,
                order="prefill_first" if "prefill_first" in mode or "pre_high" in mode else "ae_first",
                semantics="fresh_KV" if "fresh" in mode else "complete_old_KV",
                gpu_uuid=result.get("gpu_uuid"),bank_sha256=result.get("bank_sha256"),
                selected_bank_indices=result.get("selected_bank_indices"),sequence_sha256=digest(sequence),
                model_config=result.get("identity",{}).get("config"),checkpoint=args.get("checkpoint"),
                reset_every=args.get("reset_every"),numeric_pass=numerical,
                input_noise_rule_pass=seeds_ok and index_ok,steady_recaptures=recaptures,
                captured_engines=group.get("capture_engines"),cold_capture_ms=group.get("cold_capture_ms"),
                tolerance=result.get("tolerance"),
                peak_allocated_bytes=max(r.get("peak_allocated",0) for r in selected),
                peak_reserved_bytes=max(r.get("peak_reserved",0) for r in selected),
                eligible=category in ("main","isolated") and complete and numerical and
                         recaptures==0 and seeds_ok and index_ok and len(selected)==expected_count and len(selected)>=100)
            for metric in METRICS:
                default=0 if metric=="previous_prefix_wait_ms" and not args.get("await_previous_prefix") else None
                row.update({metric+"_"+key:value for key,value in stats([r.get(metric,default) for r in selected]).items() if key!="n"})
            for error in ("normalized_error","decoded_error"):
                for measure in ("mae","max_abs"):
                    vals=[r[error][measure] for r in selected if error in r]
                    row[error+"_"+measure+"_max"]=max(vals) if vals else None
            row["prefill_submissions"]=sum("prefill_gpu_event_ms" in r for r in selected)
            row["both_branches_each_request"]=row["prefill_submissions"]==len(selected) and all("ae_gpu_event_ms" in r for r in selected)
            row["completion_scope"]=("AE and Prefill complete on every measured request" if row["both_branches_each_request"] else
                                     "Only actually submitted branches; some requests may lack AE or Prefill, not an equal-work two-branch comparison")
            row["consumed_snapshots"]=len({(r["epoch"],r["snapshot"]["slot"],r["snapshot"]["generation"]) for r in selected})
            row["prefill_done_at_response_fraction"]=float(np.mean([r.get("prefill_done_at_response",False) for r in selected]))
            if phase=="continuous" and "host_end" in selected[-1]:
                duration=selected[-1]["host_end"]-selected[0]["host_begin"]
                row["trajectory_wall_seconds"]=duration
                row["prefill_submissions_hz"]=row["prefill_submissions"]/duration
                row["consumed_snapshot_fraction"]=row["consumed_snapshots"]/len(selected)
                adjacent=[(a,b) for a,b in zip(selected,selected[1:]) if a["epoch"]==b["epoch"]]
                deltas=[b["pool"]["publications"]-a["pool"]["publications"] for a,b in adjacent
                        if "publications" in a["pool"] and "publications" in b["pool"]]
                row["same_epoch_response_intervals"]=len(adjacent)
                row["cross_epoch_response_intervals_excluded"]=len(selected)-1-len(adjacent)
                row["epoch_first_responses_excluded"]=len(selected)-len(adjacent)
                row["publication_counter_complete"]=len(deltas)==len(adjacent) and all(d>=0 for d in deltas)
                row["published_snapshot_updates"]=sum(deltas) if row["publication_counter_complete"] else None
                row["published_snapshot_updates_hz"]=row["published_snapshot_updates"]/duration if row["publication_counter_complete"] else None
                row["consumed_snapshot_transitions"]=sum(a["snapshot"]!=b["snapshot"] for a,b in adjacent)
                row["consumed_snapshot_transitions_hz"]=row["consumed_snapshot_transitions"]/duration
                row["update_frequency_scope"]="Host-observed Pool publication deltas and consumed identity changes between same-epoch adjacent responses; first response and cross-epoch bootstrap excluded; denominator includes reset/bootstrap wall time, conservatively undercounting boundaries"
            else:
                row["prefill_submissions_hz"]=None
            output.append(row)
        if group["phase"]=="paired":
            first,second=group["modes"]
            pairs=defaultdict(dict)
            errors=[]
            for row in rows:
                if row["mode"] in pairs[row["pair"]]:
                    errors.append(f"pair {row['pair']}: duplicate {row['mode']} arm")
                pairs[row["pair"]][row["mode"]]=row
            valid=[]
            if len(rows)!=2*args["count"]:
                errors.append(f"raw rows {len(rows)} != {2*args['count']} expected")
            for index,arms in sorted(pairs.items()):
                if set(arms)!={first,second}:
                    errors.append(f"pair {index}: missing or duplicate arm")
                    continue
                a,b=arms[first],arms[second]
                if (a["index"],a["source_index"],a["seed"])!=(b["index"],b["source_index"],b["seed"]):
                    errors.append(f"pair {index}: unequal observation/KV source/noise")
                    continue
                if (a["arm_order"],b["arm_order"])!=((0,1) if index%2==0 else (1,0)):
                    errors.append(f"pair {index}: AB/BA order mismatch")
                    continue
                valid.append(dict(pair=index,cpu_speedup=a["cpu_cpu_ms"]/b["cpu_cpu_ms"],
                    ae_speedup=a["ae_output_ms"]/b["ae_output_ms"],
                    both_event_speedup=a["both_completion_event_ms"]/b["both_completion_event_ms"]))
            paired.append(dict(run=path.parent.name,group=group_index,category=category,steps=args["steps"],
                variant="+".join(variant_tags(args)),await_previous_prefix=bool(args.get("await_previous_prefix")),
                comparison_type="preparation" if result.get("experiment")=="preparation_variant_ABBA" else "scheduling",
                raw_rows=len(rows),expected_raw_rows=2*args["count"],
                preparation_variants=group.get("preparation_variants"),
                independent_change=("schedule_conditioning" if group.get("preparation_variants")==["baseline","schedule"] else
                                    "layer_modulation_given_schedule" if group.get("preparation_variants")==["schedule","modulation"] else None),
                baseline=first,candidate=second,valid_pairs=len(valid),errors=errors,
                cpu_speedup=stats([r["cpu_speedup"] for r in valid]),
                ae_speedup=stats([r["ae_speedup"] for r in valid]),
                both_event_speedup=stats([r["both_event_speedup"] for r in valid]),
                eligible=category=="main" and complete and not errors and len(valid)>=100 and group.get("correct") is True
                         and all(r["eligible"] for r in output if r["group"]==group_index),
                ratios=valid))
    return run,output,paired


def fairness(rows):
    cohorts=defaultdict(list)
    for row in rows:
        if row["eligible"] and row["phase"]=="continuous" and row["semantics"]=="complete_old_KV":
            key=tuple(str(row[k]) for k in ("gpu_uuid","steps","bank_sha256","selected_bank_indices",
                                          "sequence_sha256","model_config","checkpoint","reset_every"))
            cohorts[key].append(row)
    best=[]
    for cohort in cohorts.values():
        base_serial=[r for r in cohort if r["variant"]=="base" and r["mode"]=="graph_serial"]
        baselines=base_serial or [r for r in cohort if r["mode"] in ("graph_serial","graph_serial_high") or r.get("preparation_variant")=="baseline"]
        if not baselines:
            continue
        baseline=min(baselines,key=lambda r:r["cpu_cpu_ms_p50"])
        allowed=[]
        for row in cohort:
            preparation_baseline=[r for r in cohort if r["run"]==row["run"] and r.get("preparation_variant")=="baseline"]
            own_serial=[r for r in baselines if r["run"]==row["run"]]
            reference=preparation_baseline[0] if preparation_baseline else baseline if base_serial or not own_serial else own_serial[0]
            row["freshness_baseline"]=reference["run"]+":"+reference["mode"]
            row["age_degraded"]=row["observation_age_mean"]>reference["observation_age_mean"] or row["observation_age_p95"]>reference["observation_age_p95"]
            row["receipt_age_degraded"]=row["source_age_at_response_ms_mean"]>reference["source_age_at_response_ms_mean"] or row["source_age_at_response_ms_p95"]>reference["source_age_at_response_ms_p95"]
            hz,base_hz=row.get("prefill_submissions_hz"),reference.get("prefill_submissions_hz")
            row["update_hz_ratio"]=hz/base_hz if hz is not None and base_hz else None
            row["update_hz_degraded"]=row["update_hz_ratio"]<1 if row["update_hz_ratio"] is not None else None
            for metric,label in (("published_snapshot_updates_hz","publication_hz"),("consumed_snapshot_transitions_hz","consumed_hz")):
                value,reference_value=row.get(metric),reference.get(metric)
                row[label+"_ratio"]=value/reference_value if value is not None and reference_value else None
                row[label+"_degraded"]=row[label+"_ratio"]<1 if row[label+"_ratio"] is not None else None
            row["freshness_gate"]=not row["age_degraded"] and not row["receipt_age_degraded"] and all(
                row[key] is False for key in ("update_hz_degraded","publication_hz_degraded","consumed_hz_degraded"))
            if row["freshness_gate"]:
                allowed.append(row)
        identify=lambda r:dict(run=r["run"],mode=r["mode"],variant=r["variant"],cpu_p50_ms=r["cpu_cpu_ms_p50"],
                              ae_p50_ms=r["ae_output_ms_p50"],age_degraded=r["age_degraded"],
                              receipt_age_degraded=r["receipt_age_degraded"],
                              update_hz_ratio=r["update_hz_ratio"],publication_hz_ratio=r["publication_hz_ratio"],
                              consumed_hz_ratio=r["consumed_hz_ratio"],freshness_gate=r["freshness_gate"])
        best.append(dict(gpu_uuid=baseline["gpu_uuid"],steps=baseline["steps"],
            ae_minimum=identify(min(cohort,key=lambda r:r["ae_output_ms_p50"])),
            lowest_response_preserving_freshness=identify(min(allowed,key=lambda r:r["cpu_cpu_ms_p50"])) if allowed else None))
    return best


def plot(path,rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gpus=sorted({r["gpu_uuid"] for r in rows if r["eligible"] and r["phase"]=="continuous"}) or [None]
    largest=max((sum(r["eligible"] and r["phase"]=="continuous" and r["steps"]==d and r["gpu_uuid"]==gpu for r in rows) for d in (1,5,10) for gpu in gpus),default=0)
    fig,axes=plt.subplots(len(gpus),3,squeeze=False,figsize=(18,max(6,.35*largest)*len(gpus)),constrained_layout=True)
    for ax,(gpu,steps) in zip(axes.flat,((gpu,d) for gpu in gpus for d in (1,5,10))):
        selected=[r for r in rows if r["eligible"] and r["phase"]=="continuous" and r["steps"]==steps and r["gpu_uuid"]==gpu]
        selected.sort(key=lambda r:r["cpu_cpu_ms_p50"])
        if not selected:
            ax.text(.5,.5,"Incomplete: no qualified continuous cohort",ha="center",va="center",wrap=True,transform=ax.transAxes)
        else:
            y=np.arange(len(selected))
            ax.barh(y-.18,[r["cpu_cpu_ms_p50"] for r in selected],height=.35,label="CPU-to-CPU P50",color="#3565a3")
            ax.barh(y+.18,[r["ae_output_ms_p50"] for r in selected],height=.35,label="AE output P50",color="#db8d32")
            ax.set_yticks(y,[r["variant"]+" / "+r["mode"] for r in selected],fontsize=7)
            ax.invert_yaxis()
            ax.legend(fontsize=8)
        ax.set_title(f"Flow matching D={steps}"+(f" / GPU {gpu[:8]}" if gpu else ""))
        ax.set_xlabel("Latency (ms), warmup excluded")
        ax.grid(axis="x",alpha=.2)
    fig.suptitle("CEWP qualified continuous requests; fresh-KV bars have different semantics")
    fig.savefig(path,dpi=180)
    plt.close(fig)


def plot_preparation(path,paired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for ax,steps in zip(axes,(1,5,10)):
        groups=[p for p in paired if p["comparison_type"]=="preparation" and p["steps"]==steps and p["eligible"]]
        if groups:
            data=[[r["cpu_speedup"] for r in p["ratios"]] for p in groups]
            labels=["Schedule vs baseline" if p["independent_change"]=="schedule_conditioning" else "Modulation vs schedule" for p in groups]
            ax.boxplot(data,tick_labels=labels,whis=(5,95),showfliers=False)
            ax.tick_params(axis="x",labelsize=8)
            for i,values in enumerate(data,1):
                ax.text(i,float(np.median(values)),f" {np.median(values):.3f}x",fontsize=8,va="bottom")
        else:
            ax.text(.5,.5,"Incomplete",ha="center",va="center",transform=ax.transAxes)
        ax.axhline(1,color="#a33b36",linestyle="--",linewidth=1)
        ax.set_title(f"D={steps}")
        ax.set_ylabel("Matched CPU-to-CPU speedup")
        ax.grid(axis="y",alpha=.2)
    fig.suptitle("Preparation AB/BA: 100 pairs per comparison; whiskers=P5/P95, not confidence intervals")
    fig.savefig(path,dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",default="results/idea2/cewp_20261021")
    args=parser.parse_args()
    root=Path(args.root)
    runs,rows,paired,native,stages,lifecycle,unrecognized,errors=[],[],[],[],[],[],[],[]
    for path in sorted(root.glob("*/result.json")):
        try:
            result=json.loads(path.read_text())
            if "old_layer_error" in result:
                stages.append(dict(run=path.parent.name,path=str(path),args=result["args"],
                    numerical_error=result.get("old_layer_error"),steady_recaptures=result.get("steady_recaptures"),
                    limitation=result.get("limitation")))
            elif str(result.get("backend","")).startswith("original OpenPI JAX"):
                data=result.get("rows",[])
                entry=dict(run=path.parent.name,path=str(path),steps=result["args"]["steps"],n=len(data),
                    mode=result.get("mode"),gpu_uuid=result.get("gpu_uuid"),semantics="fresh_KV_native_JAX",
                    bank_sha256=result.get("bank_sha256"),selected_bank_indices=result.get("selected_bank_indices"),
                    sequence=result["args"].get("sequence"),boundary=result.get("boundary"),
                    repeatability=result.get("repeatability"),sampled_peak_device_used_mib=result.get("sampled_peak_device_used_mib"),
                    jax_allocator_stats_after_warmup=result.get("jax_allocator_stats_after_warmup"),
                    jax_allocator_stats_after_steady=result.get("jax_allocator_stats_after_steady"),
                    jax_allocator_stats_scope=result.get("jax_allocator_stats_scope"),
                    comparison=result.get("comparison"),eligible=False,
                    status="complete" if len(data)==result["args"]["count"] and result.get("repeatability",{}).get("passed") else "incomplete")
                entry.update({"cpu_cpu_ms_"+k:v for k,v in stats([r["cpu_cpu_ms"] for r in data]).items()})
                native.append(entry)
            elif "results" in result and "passed" in result and "rows" not in result:
                lifecycle.append(dict(run=path.parent.name,path=str(path),passed=result["passed"],
                    args=result.get("args"),tolerance=result.get("tolerance"),scope=result.get("scope")))
            elif isinstance(result.get("groups"),list) and isinstance(result.get("rows"),list):
                run,measurements,pairs=request_run(path,result)
                runs.append(run);rows.extend(measurements);paired.extend(pairs)
            else:
                unrecognized.append(dict(run=path.parent.name,path=str(path),keys=sorted(result),
                                         status="separate_schema_not_ranked"))
        except (json.JSONDecodeError,KeyError,ValueError,TypeError) as error:
            errors.append(dict(path=str(path),error=repr(error),status="incomplete_or_unrecognized"))
    best=fairness(rows)
    expected=[f"d{d}_main" for d in (10,5,1)]+[f"d{d}_isolated" for d in (10,5,1)]
    expected += [f"d{d}_selective_{kind}" for d in (10,5,1) for kind in ("all","vision_only","vlm_only")]
    expected += [f"d{d}_hoist" for d in (10,5,1)]
    expected += [f"d{d}_{variant}" for d in (10,5,1) for variant in ("wait_full","vision_wait","modulation","vision_modulation_wait")]
    expected += [f"d{d}_vision_modulation_wait_repeat" for d in (10,5,1)]
    expected += [f"d{d}_preparation_pair_gpu3" for d in (10,5,1)]
    status={r["run"]:r["status"] for r in runs}
    incomplete={name:status.get(name,"missing") for name in expected if status.get(name)!="complete"}
    native_status={r["run"]:r["status"] for r in native}
    native_expected={f"native_d{d}_gpu2":native_status.get(f"native_d{d}_gpu2","missing") for d in (10,5,1)}
    formal_pair_audit=[]
    for run in runs:
        if run["category"]!="main":
            continue
        result=json.loads(Path(run["path"]).read_text())
        formal=[p for p in paired if p["run"]==run["run"]]
        formal_pair_audit.append(dict(run=run["run"],run_status=run["status"],
            expected_groups=len(result["args"].get("pairs",[])),observed_groups=len(formal),
            expected_raw_rows=2*result["args"]["count"]*len(result["args"].get("pairs",[])),
            observed_raw_rows=sum(p["raw_rows"] for p in formal),
            expected_total_rows=2*result["args"]["count"]*len(result["args"].get("pairs",[]))+
                                result["args"]["continuous_count"]*len(result["args"].get("continuous",[])),
            observed_total_rows=len(result.get("rows",[])),
            groups=[dict(group=p["group"],raw_rows=p["raw_rows"],valid_pairs=p["valid_pairs"],errors=p["errors"]) for p in formal]))
    summary=dict(generated_unix=time.time(),runs=runs,measurements=rows,paired=paired,native_fresh=native,
        stages=stages,lifecycle=lifecycle,unrecognized=unrecognized,read_errors=errors,
        incomplete_expected=incomplete,native_expected_status=native_expected,
        formal_pair_audit=formal_pair_audit,best_candidates=best,
        interpretation=["Event spans include host submission gaps and resource waits; they are not kernel busy time.",
          "Preparation ablations are schedule_only versus modulation_plus_schedule. Full modulation includes schedule as a prerequisite; enabling both flags is one combined variant, not two independent gains.",
          "Paired speedups use matched group/pair observation, old-KV source, seed, and alternating AB/BA order.",
          "Native JAX uses fresh KV and separate timing boundary; no numerical equivalence or cross-GPU ranking is asserted.",
          "Prefill submission Hz is a submission frequency, not completed KV update frequency.",
          "Publication Hz sums Pool.publications deltas between same-epoch adjacent responses; consumption Hz counts same-epoch adjacent consumed snapshot identity changes. First observations and cross-epoch bootstrap boundaries are excluded, while elapsed trajectory includes reset/bootstrap, so these are conservative host-observed rates, not GPU-ready timestamps.",
          "Best candidates require completed runs, >=100 samples, numerical success, zero steady captures, common GPU/model/input/noise, and no age or update-rate regression."])
    csv_write(root/"comparison.csv",[r for r in rows if r["category"] in ("main","isolated")])
    for category,filename in (("smoke","smoke_comparison.csv"),("compile_failed","compile_failures.csv"),("profile","profile_comparison.csv")):
        csv_write(root/filename,[r for r in rows if r["category"]==category])
    csv_write(root/"native_fresh_comparison.csv",native)
    csv_write(root/"paired_comparison.csv",[{k:v for k,v in p.items() if k!="ratios"} for p in paired])
    (root/"summary.json").write_text(json.dumps(summary,indent=2))
    audit_errors=[]
    for audit in formal_pair_audit:
        if audit["run_status"]=="complete":
            if any(audit[a]!=audit[b] for a,b in (("expected_groups","observed_groups"),("expected_raw_rows","observed_raw_rows"),("expected_total_rows","observed_total_rows"))):
                audit_errors.append(audit["run"]+": completed run has missing groups/rows")
            if any(g["errors"] or g["raw_rows"]!=2*g["valid_pairs"] for g in audit["groups"]):
                audit_errors.append(audit["run"]+": completed run has unmatched pairs")
    flags=[]
    for row in rows:
        variant=row.get("preparation_variant")
        if variant not in ("baseline","schedule","modulation"):
            continue
        expected_flags=(variant=="schedule",variant=="modulation",variant!="baseline")
        actual=(row["hoist_schedule"],row["hoist_modulation"],row["schedule_hoist_effective"])
        flags.append(dict(run=row["run"],group=row["group"],mode=row["mode"],variant=variant,
                          hoist_schedule=actual[0],hoist_modulation=actual[1],schedule_hoist_effective=actual[2],passed=actual==expected_flags))
        if actual!=expected_flags:
            audit_errors.append(row["run"]+":"+row["mode"]+": wrong preparation flags")
    old_receipt=root/"report_selfcheck.json"
    prior=json.loads(old_receipt.read_text()) if old_receipt.exists() else {}
    prior.update(generated_unix=time.time(),passed=not audit_errors,errors=audit_errors,
                 formal_pair_audit=formal_pair_audit,preparation_variant_flags=flags,
                 full_experiment_complete=not incomplete and all(v=="complete" for v in native_expected.values()))
    old_receipt.write_text(json.dumps(prior,indent=2))
    plot(root/"comparison.png",rows)
    plot_preparation(root/"preparation_comparison.png",paired)
    print(json.dumps(dict(runs=len(runs),measurement_cohorts=len(rows),incomplete_expected=incomplete,
                         errors=errors,best_candidates=best),indent=2))


if __name__=="__main__":
    main()
