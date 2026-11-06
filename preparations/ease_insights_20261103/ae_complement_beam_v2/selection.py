"""Complete-only depth/time selection with original Full and AE16 from distinct roots."""
import argparse
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import statistics
from preparations.ease_insights_20261103.pipeline import file_lock
from preparations.ease_insights_20261103.vlm.analyze import key,load_episodes,paired,summary
from preparations.ease_insights_20261103.vlm.analyze_confirmation import audit_identities,unexpected_cases
from .cost_records import cost_records
from .common import ROOT,CONDITIONS,NEW_CONDITIONS,DIRECTIONS,REFERENCES,sha,atomic_json,digest,read_manifest,cases,stage_for,conditions,reference

CONDITION_IDS=list(CONDITIONS)

def require(value,message):
    if not value:raise ValueError(message)


def read_selection(path):
    v=json.loads(Path(path).read_text());payload=copy.deepcopy(v);identity=payload.pop('selection_hash',None)
    require(identity is not None and digest(payload)==identity,'Depth/time selection digest differs')
    require(v.get('revision')=='ease-AE-joint-mask-selection-v2' and v.get('complete_discovery_cells')==320
        and v.get('new_discovery_episodes')==160 and v.get('preserved_reference_episodes')==160
        and v.get('independent_validation_launched') is False,'Not a complete frozen depth/time decision')
    return v


def load_stage(root,stage):
    root=Path(root).resolve();m=read_manifest(root/'manifest.json',verify_sources=True)
    require(stage_for(m)==stage and stage in ('discovery','confirm'),'Incorrect depth/time analysis stage')
    scenes=cases(m,stage);view=copy.deepcopy(m)
    view['conditions']=[c for c in m['conditions'] if c['id'] in m['planned_conditions']]
    view['discovery_inits']=m['discovery_inits' if stage=='discovery' else 'confirmation_inits']
    records,missing,invalid,evidence=load_episodes(root,view);invalid.extend(audit_identities(root,m,records))
    from . import common,worker
    tool=ROOT/'reports/ease_insights_20261103/common_case_audit_v1/audit.py'
    require(sha(tool)=='2fec7afe6e80cca6286e52a467d42be9e525f6125da76d0b63873985b4a6d113','Accepted case validator changed')
    spec=importlib.util.spec_from_file_location('depth_time_readonly_validator',tool);validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
    for (cid,k),record in list(records.items()):
        try:validator.case(common,worker,root,m,cid,record['case'])
        except (ValueError,RuntimeError,KeyError,OSError) as error:
            invalid.append(dict(condition=cid,case=record['case'],error=repr(error)));del records[cid,k]
    new_valid=len(records);reference_valid=0;reference_evidence=[]
    if stage=='discovery':
        for alias in REFERENCES:
            origin,old,old_common,old_worker,source_cid=reference(alias,verify_sources=True)
            require(old['checkpoints']==m['checkpoints'] and old['noise']==m['noise'],'Control checkpoint/noise differs')
            for field in ('H','R','refresh_D','settling_actions','environment_seed','policy_action_limits','training'):
                require(old['science'][field]==m['science'][field],'Control active physical/science contract differs: '+field)
            prior_view=copy.deepcopy(old);prior_view['conditions']=[c for c in old['conditions'] if c['id']==source_cid]
            prior,old_missing,old_invalid,old_evidence=load_episodes(origin,prior_view)
            old_invalid.extend(audit_identities(origin,old,prior))
            missing.extend(dict(v,source=alias) for v in old_missing);invalid.extend(dict(v,source=alias) for v in old_invalid)
            for scene in scenes:
                item=prior.get((source_cid,key(scene)))
                if item is not None:
                    # Validate and retain the source record; the alias exists
                    # only as an explicitly derived analysis key.
                    try:validator.case(old_common,old_worker,origin,old,source_cid,scene)
                    except (ValueError,RuntimeError,KeyError,OSError) as error:
                        invalid.append(dict(condition=alias,case=scene,source='original_control',error=repr(error)));continue
                    records[alias,key(scene)]=dict(item,analysis_condition=alias,data_origin=dict(root=str(origin),condition=source_cid,
                        manifest_hash=old['manifest_hash'],new_replication=False));reference_valid+=1
            reference_evidence.extend(old_evidence);evidence.extend(old_evidence)
    for scene in scenes:
        present=[records[c,key(scene)] for c in m['analytical_conditions'] if (c,key(scene)) in records]
        if present:
            first=present[0]['initial_gate']
            if any(any(r['initial_gate'].get(k)!=first.get(k) for k in ('input_hash','seed','physical_assets')) for r in present):
                invalid.append(dict(case=scene,error='Original/new physical first input or noise differs'))
    unexpected=unexpected_cases(root,m,scenes);expected=len(scenes)*len(m['analytical_conditions'])
    return dict(root=root,manifest=m,scenes=scenes,records=records,missing=missing,invalid=invalid,unexpected=unexpected,
        episode_evidence=evidence,reference_evidence=reference_evidence,new_valid=new_valid,reference_valid=reference_valid,
        expected=expected,complete=len(records)==expected and not missing and not invalid and not unexpected)


def completed_service_costs(root, manifest):
    """Equal service weighting; only completed runner services enter eligibility."""
    result = cost_records(Path(root), manifest)
    accepted, excluded = [], list(result["excluded_or_pending"])
    for service in result["services"]:
        path = Path(service["path"])
        try:
            receipt = json.loads((path.parent / "result.json").read_text())
            protocol = json.loads((path.parent / "protocol.json").read_text())
            if (receipt.get("complete") is not True or receipt.get("manifest_hash") != manifest["manifest_hash"] or
                    receipt.get("condition") != protocol["condition_id"] or
                    protocol.get("qk_reference_passed") is not True or
                    protocol.get("config") != next(c for c in manifest["conditions"] if c["id"] == receipt["condition"]) or
                    protocol.get("checkpoint_sha256") != manifest["checkpoints"]["pi0"]["sha256"] or
                    path.parent.parent.parent.name != receipt["condition"]):
                raise ValueError("Cost service has no matching completed runner receipt")
            accepted.append(dict(service, artifact_evidence=[dict(path=str(path.parent / name), sha256=sha(path.parent / name))
                for name in ("fixed_input_cost.json", "protocol.json", "gate.json", "cross_service.json", "result.json")]))
        except (OSError, ValueError, KeyError, TypeError) as error:
            excluded.append(dict(path=str(path), reason=str(error)))
    ratios = {cid: statistics.median(s["full_over_policy_cycle"][cid] for s in accepted)
        for cid in CONDITION_IDS} if accepted else {}
    return dict(valid_services=len(accepted), services=accepted, excluded_or_pending=excluded,
        median_full_over_policy_cycle=ratios,
        pooling="Median of per-completed-service same-input mean Full / candidate three-request native-policy cycle costs",
        scope="Native policy inference only; excludes RPC and environment. Shared GPU load may affect ratios; no closed-loop acceleration claim.")


def usable(count, full_count, n):
    # Exact integer comparisons implement >=.75 and >=Full-.15 without rounding.
    return count * 4 >= n * 3 and (full_count - count) * 20 <= n * 3


def direction_summary(a, b, records, scenes, tasks):
    result = paired(a, b, records, scenes)
    rows = [dict(task=t, **paired(a, b, records, [c for c in scenes if c["task"] == t])) for t in tasks]
    total = result["success_difference_count"]
    loo = [dict(task=r["task"], difference_count=total - r["success_difference_count"],
        difference=(total - r["success_difference_count"]) / (len(scenes) - r["n_paired"])) for r in rows]
    result.update(by_task=rows, positive_tasks=sum(r["success_difference_count"] > 0 for r in rows),
        leave_one_task_out=loo, every_leave_one_task_out_positive=all(r["difference_count"] > 0 for r in loo),
        interpretation="Discovery-selected comparison; all p-values are descriptive only")
    return result


def select_pair(manifest, records, costs, complete):
    if not complete:
        return dict(status="incomplete_discovery", selected=None, eligible_crossings=[])
    if not costs.get("valid_services") or any(cid not in costs.get("median_full_over_policy_cycle", {}) for cid in CONDITION_IDS):
        return dict(status="missing_valid_cost_probe", selected=None, eligible_crossings=[])
    scenes = [dict(suite=s, task=t, init=i) for s in manifest["suites"] for t in manifest["tasks"] for i in manifest["discovery_inits"]]
    candidates = sorted(cid for cid in manifest["analytical_conditions"] if cid != "full_d10")
    counts = {cid: sum(records[cid, key(c)]["success"] for c in scenes) for cid in manifest["analytical_conditions"]}
    n = len(scenes)
    gates = {}
    for cid in candidates:
        ratio = costs["median_full_over_policy_cycle"][cid]
        passed = usable(counts[cid], counts["full_d10"], n)
        gates[cid] = dict(successes=counts[cid], n=n, success_rate=counts[cid] / n,
            full_success_rate=counts["full_d10"] / n, usability_pass=passed,
            median_fixed_input_speed_ratio=ratio, cost_pass=ratio >= 1.15,
            qualified=passed and ratio >= 1.15)
    comparisons, crossings, eligible = [], [], []
    for a, b in itertools.combinations(NEW_CONDITIONS, 2):
        directions = {}
        for suite in manifest["suites"]:
            cases = [c for c in scenes if c["suite"] == suite]
            for better, worse in ((a, b), (b, a)):
                d = direction_summary(better, worse, records, cases, manifest["tasks"])
                d.update(suite=suite, better=better, worse=worse)
                d["direction_eligible"] = (10 * d["success_difference_count"] >= d["n_paired"] and
                    d["positive_tasks"] >= 2 and d["every_leave_one_task_out_positive"])
                directions[better, suite] = d
            comparisons.append(dict(pair=[a, b], suite=suite, a_over_b=directions[a, suite], b_over_a=directions[b, suite]))
        for x, y in itertools.permutations(manifest["suites"], 2):
            da, db = directions[a, x], directions[b, y]
            gap_a, gap_b = da["success_difference"], db["success_difference"]
            entry = dict(pair=[a, b], paths=[a, b], directions=[
                dict(suite=x, better=a, worse=b, discovery_difference=gap_a),
                dict(suite=y, better=b, worse=a, discovery_difference=gap_b)],
                minimum_suite_gap=min(gap_a, gap_b), sum_suite_gaps=gap_a + gap_b,
                minimum_overall_success=min(counts[a], counts[b]) / n,
                ranking_counts=[min(da["success_difference_count"], db["success_difference_count"]),
                    da["success_difference_count"] + db["success_difference_count"], min(counts[a], counts[b])],
                lexical_tie_key=[a, b, x, y],
                qualified_paths=gates[a]["qualified"] and gates[b]["qualified"],
                directions_eligible=da["direction_eligible"] and db["direction_eligible"])
            entry["cost_matched"] = max(costs["median_full_over_policy_cycle"][a], costs["median_full_over_policy_cycle"][b]) <= 1.10 * min(costs["median_full_over_policy_cycle"][a], costs["median_full_over_policy_cycle"][b])
            entry["fixed_proxy_directions"] = [{k:d[k] for k in ('suite','better','worse')} for d in entry['directions']] == DIRECTIONS
            entry["eligible"] = entry["qualified_paths"] and entry["directions_eligible"] and entry["cost_matched"] and entry['fixed_proxy_directions']
            crossings.append(entry)
            if entry["eligible"]:
                eligible.append(entry)
    # All suites have exactly20 cases and all paths80, so integer counts are
    # exactly the prescribed three rate criteria, without floating tie drift.
    eligible.sort(key=lambda e: (*(-v for v in e["ranking_counts"]), *e["lexical_tie_key"]))
    return dict(status="selected" if eligible else "no_candidate", selected=eligible[0] if eligible else None,
        condition_eligibility=gates, all_pair_suite_comparisons=comparisons, all_ordered_suite_crossings=crossings,
        eligible_crossings=eligible, candidate_order=candidates, suite_order=manifest["suites"],
        rule="Only the frozen proxy pair and Object/Spatial directions; >=.75 and >=Full-.15 overall; median completed-service cost ratio>=1.15; opposite gaps>=.10, >=2 positive tasks and every task LOO positive; pair costs within10%",
        interpretation="Exploratory selection only. Names denote proposal provenance, not assumed task preference. Pair costs must also match within10%. No candidate ends this bounded search.")


def condition_summaries(manifest, records, scenes):
    groups = [("all", None, None, scenes)]
    groups += [("suite", s, None, [c for c in scenes if c["suite"] == s]) for s in manifest["suites"]]
    groups += [("task", s, t, [c for c in scenes if c["suite"] == s and c["task"] == t]) for s in manifest["suites"] for t in manifest["tasks"]]
    rows = []
    for level, suite, task, cases in groups:
        for cid in manifest["analytical_conditions"]:
            values = [records[cid, key(c)] for c in cases if (cid, key(c)) in records]
            aggregate = summary(values)
            actions = aggregate["actions"]
            aggregate["native_policy_ms_per_executed_action"] = aggregate["policy"]["total_ms"] / actions if actions else None
            aggregate["full_client_rpc_ms_per_executed_action"] = aggregate["rpc"]["total_ms"] / actions if actions else None
            rows.append(dict(level=level, suite=suite, task=task, condition=cid, expected_n=len(cases), **aggregate))
    return rows


def analyze_discovery(root):
    stage=load_stage(root,'discovery');m=stage['manifest'];costs=completed_service_costs(stage['root'],m)
    complete=stage['complete'] and costs['valid_services']==4
    decision=select_pair(m,stage['records'],costs,complete)
    return dict(revision='ease-AE-joint-mask-discovery-analysis-v2',manifest_hash=m['manifest_hash'],complete=complete,
        expected_episodes=320,valid_episodes=len(stage['records']),analytical_cells=320,
        new_discovery_episodes=stage['new_valid'],preserved_reference_episodes=stage['reference_valid'],
        missing=stage['missing'],invalid=stage['invalid'],unexpected=stage['unexpected'],
        protocol_evidence=m['protocol_evidence'],design_evidence=m['design_evidence'],selection=decision,
        condition_summaries=condition_summaries(m,stage['records'],stage['scenes']),fixed_input_cost=costs,
        evidence=dict(episodes=stage['episode_evidence'],preserved_reference=stage['reference_evidence'],
            manifest=dict(path=str(stage['root']/'manifest.json'),sha256=sha(stage['root']/'manifest.json')),
            analysis_source=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__))),
        reference_scope='160 original Full/AE16D2 records, source identities unchanged; not160 newly replicated episodes',
        no_confirmation_outcomes_used=True)


def freeze(output,report):
    require(report['complete'] and report['valid_episodes']==320 and report['new_discovery_episodes']==160
        and report['preserved_reference_episodes']==160 and report['selection']['status'] in ('selected','no_candidate'),
        'Joint-mask selection requires all160 new and160 preserved accepted cases')
    value=dict(revision='ease-AE-joint-mask-selection-v2',discovery_manifest_hash=report['manifest_hash'],
        complete_discovery_cells=320,new_discovery_episodes=160,preserved_reference_episodes=160,
        protocol_evidence=report['protocol_evidence'],design_evidence=report['design_evidence'],
        selection=report['selection'],evidence=report['evidence'],independent_validation_launched=False)
    value['selection_hash']=digest(value);output=Path(output).resolve();path=output/'selection.json'
    with file_lock(output/'selection.lock'):
        if path.exists():require(read_selection(path)==value,'Do not replace a frozen module selection')
        else:atomic_json(path,value)
    return path


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--freeze-selection',action='store_true');args=parser.parse_args()
    report=analyze_discovery(args.root);output=Path(args.output).resolve();atomic_json(output/'analysis.json',report)
    selected=freeze(output,report) if args.freeze_selection else None
    print(json.dumps(dict(complete=report['complete'],new_valid_episodes=report['new_discovery_episodes'],
        preserved_reference_episodes=report['preserved_reference_episodes'],selection_status=report['selection']['status'],selection=str(selected) if selected else None)))

