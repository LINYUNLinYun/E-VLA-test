"""Independent confirmation statistics for the frozen pair and suite directions.

Read-only with respect to scientific results and runtime manifests. Bootstrap
methods and stability interpretation are fixed here, not selected from outcomes.
"""
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import re
import numpy as np

from .common import SUITES, atomic_json, digest, read_manifest, sha
from .confirm_common import PHASE, validate_output
from .analyze import cost_records, exact_paired, key, load_episodes, paired, summary

BOOTSTRAP_REPLICATES = 10000


def analysis_protocol():
    return dict(bootstrap_replicates=BOOTSTRAP_REPLICATES,confidence=.95,
        case_bootstrap="Paired per-case success differences; sample 50 cases with replacement within each prespecified suite; percentile 2.5/97.5 intervals",
        task_cluster_bootstrap="Sample ten tasks with replacement, retaining each selected task's five paired initial states; percentile 2.5/97.5 intervals",
        seed="SHA256(manifest hash, frozen suite/better/worse, method); independent of observed outcomes",
        primary_tests="One-sided exact conditional McNemar for each frozen better>worse direction; Holm-adjusted across exactly two hypotheses",
        stability="Positive paired difference, both bootstrap lower bounds >0, and positive leave-one-task-out and leave-one-init-out differences",
        interpretation="Support complementarity only when both frozen directions meet stability; otherwise distinguish positive directions with uncertainty from directions not reproduced. No single p threshold determines the conclusion.",
        caveats="Ten task clusters and five initial states limit precision; bootstrap intervals are approximate. Non-significance is not equivalence. Other categories are descriptive; no path or suite reselection.")


def read_selection(path):
    obj=json.loads(Path(path).read_text())
    payload=copy.deepcopy(obj); identity=payload.pop("selection_hash",None)
    if identity is None or digest(payload)!=identity:
        raise ValueError("Frozen selection hash is missing or invalid")
    return obj


def holm(pvalues):
    order=sorted(range(len(pvalues)),key=lambda i:(pvalues[i],i))
    adjusted=[None]*len(pvalues); previous=0.
    for rank,index in enumerate(order):
        previous=max(previous,min(1.,(len(pvalues)-rank)*pvalues[index]))
        adjusted[index]=previous
    return adjusted


def bootstrap_seed(manifest_hash,direction,method):
    value=["ease-confirmation-bootstrap-v1",manifest_hash,direction["suite"],direction["better"],direction["worse"],method]
    return int(hashlib.sha256(json.dumps(value,separators=(",",":")).encode()).hexdigest()[:16],16)


def bootstrap(values,task_ids,seed_case,seed_task,replicates=BOOTSTRAP_REPLICATES):
    values=np.asarray(values,dtype=np.float64)
    tasks=sorted(set(task_ids))
    if len(values)!=50 or len(tasks)!=10 or any(task_ids.count(t)!=5 for t in tasks):
        raise ValueError("Primary confirmation bootstrap requires ten tasks × five paired initial states")
    case_rng=np.random.default_rng(seed_case)
    estimates=case_rng.choice(values,size=(replicates,len(values)),replace=True).mean(axis=1)
    task_means=np.asarray([values[[i for i,t in enumerate(task_ids) if t==task]].mean() for task in tasks])
    task_rng=np.random.default_rng(seed_task)
    cluster_estimates=task_rng.choice(task_means,size=(replicates,len(tasks)),replace=True).mean(axis=1)
    return dict(case=dict(low=float(np.quantile(estimates,.025)),high=float(np.quantile(estimates,.975)),
                    seed=seed_case,replicates=replicates,unit="paired case",method="percentile"),
        task_cluster=dict(low=float(np.quantile(cluster_estimates,.025)),high=float(np.quantile(cluster_estimates,.975)),
                    seed=seed_task,replicates=replicates,clusters=len(tasks),unit="whole task with its five initial states",method="percentile"))


def direction_analysis(direction,manifest,records,scenes,complete,replicates=BOOTSTRAP_REPLICATES):
    suite_cases=[c for c in scenes if c["suite"]==direction["suite"]]
    a,b=direction["better"],direction["worse"]
    result=dict(frozen_direction=direction,**paired(a,b,records,suite_cases))
    result["interpretation"]="Prespecified independent confirmation direction" if complete else "Incomplete confirmation; counts only, no conclusion"
    by_task=[dict(task=t,**paired(a,b,records,[c for c in suite_cases if c["task"]==t])) for t in manifest["tasks"]]
    by_init=[dict(init=i,**paired(a,b,records,[c for c in suite_cases if c["init"]==i])) for i in manifest["confirmation_inits"]]
    def consistency(rows):
        differences=[r["success_difference"] for r in rows if r["n_paired"]==r["expected_n"]]
        return dict(complete_groups=len(differences),positive=sum(d>0 for d in differences),
            tied=sum(d==0 for d in differences),negative=sum(d<0 for d in differences))
    result.update(by_task=by_task,by_init=by_init,task_direction_consistency=consistency(by_task),
        init_direction_consistency=consistency(by_init),inference=None)
    if not complete:
        result["exact_mcnemar_two_sided"]=None
        for row in by_task+by_init:row["exact_mcnemar_two_sided"]=None
        return result
    values=[int(records[a,key(c)]["success"])-int(records[b,key(c)]["success"]) for c in suite_cases]
    task_ids=[c["task"] for c in suite_cases]
    intervals=bootstrap(values,task_ids,bootstrap_seed(manifest["manifest_hash"],direction,"case"),
        bootstrap_seed(manifest["manifest_hash"],direction,"task_cluster"),replicates)
    loo_tasks=[dict(task=t,difference=float(np.mean([v for v,c in zip(values,suite_cases) if c["task"]!=t]))) for t in manifest["tasks"]]
    loo_inits=[dict(init=i,difference=float(np.mean([v for v,c in zip(values,suite_cases) if c["init"]!=i]))) for i in manifest["confirmation_inits"]]
    positive=result["success_difference"]>0
    resampling=intervals["case"]["low"]>0 and intervals["task_cluster"]["low"]>0
    loo=all(x["difference"]>0 for x in loo_tasks+loo_inits)
    result["inference"]=dict(exact_mcnemar_one_sided=exact_paired(result["a_only"],result["b_only"],one_sided=True),
        holm_adjusted_p=None,confidence_intervals=intervals,positive_direction=positive,
        robust_to_case_and_task_resampling=resampling,positive_after_any_one_task_or_init_removed=loo,
        stable_positive_direction=positive and resampling and loo,leave_one_task_out=loo_tasks,leave_one_init_out=loo_inits)
    return result


def contention_records(root,records=None):
    root=Path(root).resolve(); evidence=[]; malformed=[]; gpu_sets={}; times=[]
    files=sorted({*root.rglob("*.json"),*root.rglob("*.jsonl")}) if root.exists() else []
    for path in files:
        try:
            raw=path.read_text()
            values=[json.loads(line) for line in raw.splitlines() if line.strip()] if path.suffix==".jsonl" else [json.loads(raw)]
            evidence.append(dict(path=str(path),sha256=sha(path),records=values))
            for item in values:
                if isinstance(item.get("observed_at"),(int,float)):times.append(item["observed_at"])
                for pid,gpus in item.get("multi_gpu_processes",{}).items():
                    gpu_sets.setdefault(pid,set()).update(gpus)
        except (ValueError,TypeError,AttributeError,OSError) as exc:
            malformed.append(dict(path=str(path),error=str(exc)))
    known_gpus=set().union(*gpu_sets.values()) if gpu_sets else set()
    conditions=sorted({c for c,_ in records}) if records else []
    overlap={cid:dict(episode_gpu_uuids=sorted({r["gpu_uuid"] for (c,_),r in records.items() if c==cid}),
        shares_any_recorded_contention_gpu=any(r["gpu_uuid"] in known_gpus for (c,_),r in records.items() if c==cid),
        exact_temporal_overlap_established=False) for cid in conditions}
    return dict(directory=str(root),all_known_records=evidence,unreadable_records=malformed,
        known_multi_gpu_processes={pid:sorted(gpus) for pid,gpus in gpu_sets.items()},
        observed_timestamp_range=None if not times else [min(times),max(times)],per_condition_gpu_overlap=overlap,
        model_binding="Each study model process is pinned to one GPU UUID; a recorded multi-GPU process is external to these jobs",
        timing_limit="These snapshots do not establish exclusive access or complete overlap intervals. Policy/RPC/wall time and even same-service fixed-input ratios can be affected by contention; do not report them as exclusive-GPU speed.",
        quality_handling="All valid scientific outcomes retained unchanged. Contention does not reclassify success/failure or authorize retries; technical failures remain separate.")


def unexpected_cases(root,manifest,scenes):
    planned={(cid,key(c)) for cid in manifest["planned_conditions"] for c in scenes}
    unexpected=[]
    for path in root.glob("*/episodes/*/task*/init*/complete.json"):
        directory=path.parent
        # Retained technical duplicates are not independent scientific cases.
        if not re.fullmatch(r"init\d+",directory.name):continue
        try:
            result=json.loads((directory/"result.json").read_text())
            if (result["condition"],key(result["case"])) not in planned:
                unexpected.append(dict(path=str(path),condition=result["condition"],case=result["case"]))
        except (OSError,ValueError,KeyError,TypeError) as exc:
            unexpected.append(dict(path=str(path),error=str(exc)))
    return unexpected


def audit_identities(root,manifest,records):
    invalid=[]
    expected_inits={(r["suite"],r["task"]):r["init_sha256"] for r in manifest["official_init_availability"]["cases"]}
    configs={c["id"]:c for c in manifest["conditions"]}
    for (cid,case_key),record in list(records.items()):
        case=record["case"]
        directory=root/cid/"episodes"/case["suite"]/f"task{case['task']:02d}"/f"init{case['init']:02d}"
        try:
            identity=json.loads((directory/"identity.json").read_text())
            done=json.loads((directory/"complete.json").read_text())
            protocol=identity["protocol"]; initial=record["initial_gate"]
            if (identity["manifest_hash"]!=manifest["manifest_hash"] or identity["case"]!=case or identity["config"]!=configs[cid] or
                not done["valid"] or done["manifest_hash"]!=manifest["manifest_hash"] or
                protocol["manifest_hash"]!=manifest["manifest_hash"] or protocol["condition_id"]!=cid or
                not protocol["real_input_parity_passed"] or protocol["gpu_uuid"]!=record["gpu_uuid"] or
                identity["init_vector_sha256"]!=expected_inits[case["suite"],case["task"]][str(case["init"])] or
                not initial["passed"] or initial["manifest_hash"]!=manifest["manifest_hash"] or initial["case"]!=case):
                raise ValueError("Original identity, completion, initialization, or service differs from confirmation")
        except (OSError,ValueError,KeyError,TypeError) as exc:
            invalid.append(dict(condition=cid,case=case,error=str(exc)))
            del records[cid,case_key]
    return invalid


def check_manifest(root,manifest):
    validate_output(root,manifest)
    selection=read_selection(manifest["selection_evidence"]["path"])
    if (selection["selection_hash"]!=manifest["selection_evidence"]["selection_hash"] or
        selection["selection"]!=manifest["selection_evidence"]["frozen_selection"]):
        raise ValueError("Confirmation selection differs from its frozen evidence")
    if selection["selection"]["status"]!="selected":
        raise ValueError("A confirmation manifest cannot launch an unselected pair")
    chosen=selection["selection"]["selected"]
    if (manifest["phase"]!=PHASE or manifest["planned_conditions"]!=["full_d10",*chosen["pair"]] or
        manifest["preregistered_hypotheses"]!=chosen["directions"] or len(chosen["directions"])!=2 or
        manifest["suites"]!=list(SUITES) or manifest["tasks"]!=list(range(10)) or
        manifest["confirmation_inits"]!=list(range(31,36)) or manifest["total_confirmation_episodes"]!=600):
        raise ValueError("Confirmation case universe or frozen directions differ")
    return selection


def analyze(root,selection_path=None,contention_root=None,replicates=BOOTSTRAP_REPLICATES):
    root=Path(root).resolve()
    contention_root=Path(contention_root).resolve() if contention_root else root.parent/"resource_contention"
    base=dict(analysis_protocol=analysis_protocol(),analysis_source=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
        no_reselection=True,quality_results_modified=False)
    if not (root/"manifest.json").exists():
        selection=read_selection(selection_path) if selection_path else None
        if selection and (selection.get("complete_discovery_episodes")!=160 or selection.get("independent_validation_launched") is not False):
            raise ValueError("Selection is not a frozen decision from complete discovery")
        status="no_selection" if selection and selection["selection"]["status"]=="none" else "incomplete"
        return dict(base,status=status,complete=False,valid_episodes=0,expected_episodes=0 if status=="no_selection" else 600,
            selection_status=None if selection is None else selection["selection"]["status"],
            selection_hash=None if selection is None else selection["selection_hash"],
            reason="Frozen discovery found no eligible pair; no independent confirmation is planned" if status=="no_selection" else "Independent confirmation has not been frozen",
            primary_directions=[],conclusion="no_selection" if status=="no_selection" else "incomplete",
            resource_contention=contention_records(contention_root))
    manifest=read_manifest(root/"manifest.json",verify_sources=True)
    selection=check_manifest(root,manifest)
    # An analysis-only view adapts the proven receipt validator to the declared
    # three conditions and init31-35. The scientific manifest/hash is unchanged.
    view=copy.deepcopy(manifest)
    view["conditions"]=[next(c for c in manifest["conditions"] if c["id"]==cid) for cid in manifest["planned_conditions"]]
    view["discovery_inits"]=manifest["confirmation_inits"]
    scenes=[dict(suite=s,task=t,init=i) for s in manifest["suites"] for t in manifest["tasks"] for i in manifest["confirmation_inits"]]
    records,missing,invalid,episode_evidence=load_episodes(root,view)
    invalid.extend(audit_identities(root,manifest,records))
    unexpected=unexpected_cases(root,manifest,scenes)
    complete=len(records)==600 and not missing and not invalid and not unexpected
    primary=[direction_analysis(d,manifest,records,scenes,complete,replicates) for d in manifest["preregistered_hypotheses"]]
    conclusion="incomplete"
    if complete:
        corrected=holm([d["inference"]["exact_mcnemar_one_sided"] for d in primary])
        for d,p in zip(primary,corrected):d["inference"]["holm_adjusted_p"]=p
        if all(d["inference"]["stable_positive_direction"] for d in primary):
            conclusion="supports_complementary_fixed_paths_on_the_two_prespecified_suites"
        elif all(d["inference"]["positive_direction"] for d in primary):
            conclusion="both_directions_positive_but_stability_uncertain"
        else:
            conclusion="both_prespecified_directions_not_reproduced"
    groups=[("all",None,None,scenes)]
    groups += [("suite",s,None,[c for c in scenes if c["suite"]==s]) for s in manifest["suites"]]
    groups += [("task",s,t,[c for c in scenes if c["suite"]==s and c["task"]==t]) for s in manifest["suites"] for t in manifest["tasks"]]
    condition_summaries=[]; descriptive=[]
    for level,suite,task,group in groups:
        for cid in manifest["planned_conditions"]:
            condition_summaries.append(dict(level=level,suite=suite,task=task,condition=cid,expected_n=len(group),
                **summary([records[cid,key(c)] for c in group if (cid,key(c)) in records])))
        for a,b in itertools.combinations(manifest["planned_conditions"],2):
            row=dict(level=level,suite=suite,task=task,**paired(a,b,records,group),
                role="descriptive_only; does not replace or reselect the two primary hypotheses")
            if not complete:row["exact_mcnemar_two_sided"]=None
            descriptive.append(row)
    costs=cost_records(root,manifest)
    costs["timing_limit"]="Report separately from free-running rollout costs; known contention may affect fixed-input ratios too"
    return dict(base,status="complete" if complete else "incomplete",complete=complete,conclusion=conclusion,
        manifest_hash=manifest["manifest_hash"],manifest_path=str(root/"manifest.json"),selection_hash=selection["selection_hash"],
        planned_conditions=manifest["planned_conditions"],valid_episodes=len(records),expected_episodes=600,
        missing=missing,invalid=invalid,unexpected_completed_cases=unexpected,primary_directions=primary,
        condition_summaries=condition_summaries,descriptive_comparisons=descriptive,
        fixed_input_cost=costs,resource_contention=contention_records(contention_root,records),
        uncertainty="Non-significant tests are not evidence of equivalence. Positive point estimates with unstable intervals remain uncertain. This tests fixed-path suite preferences, not runtime selection or phase switching.",
        evidence=dict(episode_receipts=episode_evidence,selection=dict(path=manifest["selection_evidence"]["path"],sha256=manifest["selection_evidence"]["sha256"])))


def markdown(report):
    lines=[f"Independent VLM confirmation: {report['valid_episodes']}/{report['expected_episodes']} valid episodes; status={report['status']}.",
        "Conclusion: "+report["conclusion"]+".","",
        "| Frozen direction | n | Difference | A-only / B-only | One-sided exact p | Holm p | Case 95% CI | Task-cluster 95% CI |",
        "|---|---:|---:|---:|---:|---:|---|---|"]
    for d in report["primary_directions"]:
        f=d["frozen_direction"]; inference=d["inference"]
        fmt=lambda x:"pending" if x is None else f"{x:.4f}"
        interval=lambda name:"pending" if inference is None else f"[{inference['confidence_intervals'][name]['low']:+.3f}, {inference['confidence_intervals'][name]['high']:+.3f}]"
        lines.append(f"| {f['suite']}: {f['better']} > {f['worse']} | {d['n_paired']} | {fmt(d['success_difference'])} | {d['a_only']} / {d['b_only']} | {fmt(None if inference is None else inference['exact_mcnemar_one_sided'])} | {fmt(None if inference is None else inference['holm_adjusted_p'])} | {interval('case')} | {interval('task_cluster')} |")
    lines += ["","Per-task/per-init consistency, leave-one-group-out checks, Full references, complete policy/RPC costs, fixed-input probes, and every known contention record are in analysis.json.",
        "Quality outcomes are retained unchanged. Timings do not establish exclusive-GPU speed. Non-significance does not establish equivalence; no path or category is reselected."]
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--root",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--selection");parser.add_argument("--contention-root")
    args=parser.parse_args(); output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    report=analyze(args.root,args.selection,args.contention_root)
    atomic_json(output/"analysis.json",report);(output/"analysis.md").write_text(markdown(report))
    atomic_json(output/"analysis_protocol.json",dict(source=report["analysis_source"],protocol=analysis_protocol()))
    print(json.dumps(dict(status=report["status"],valid_episodes=report["valid_episodes"],expected_episodes=report["expected_episodes"],
        conclusion=report["conclusion"],output=str(output))))


if __name__=="__main__":main()
