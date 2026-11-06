#!/usr/bin/env python3
"""Small CPU-only quantification of the preserved Spatial task1/init20 mismatch."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(path):
    raw = path.read_bytes()
    return json.loads(raw), dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())


def sign(x):
    return (x > 0) - (x < 0)


def summarize(rows):
    accepted = sum(r["accepted_actions"] for r in rows)
    verified = sum(r["verified_rounds"] for r in rows)
    return dict(episodes=len(rows), successes=sum(r["success"] for r in rows),
                accepted_actions=accepted, verify_requests=verified,
                acceptance_denominator=12 * verified,
                acceptance_rate=accepted / (12 * verified))


def main():
    snapshot, snapshot_evidence = load(HERE / "first_requests_20261103T132858Z.json")
    records = {r["mode"]: r for r in snapshot if r["source_suite"] == "libero_spatial"
               and r["target_suite"] == "libero_spatial" and r["task_id"] == 1 and r["init_id"] == 20}
    assert set(records) == {"full", "flash"}
    b, c = records["full"]["request"], records["flash"]["request"]
    dimensions = ["x", "y", "z", "rotation_0", "rotation_1", "rotation_2", "gripper"]
    numerical = {}
    for field in ("raw_prefix", "decoded_prefix"):
        maxima = [max(abs(c[field][i][j] - b[field][i][j]) for i in range(12)) for j in range(7)]
        numerical[field] = dict(per_dimension_max_abs=dict(zip(dimensions, maxima)),
                               group_max_abs=dict(xyz=max(maxima[:3]), rotation=max(maxima[3:6]), gripper=maxima[6]),
                               gripper_Full=[row[6] for row in b[field]],
                               gripper_FLASH=[row[6] for row in c[field]],
                               gripper_Full_signs=[sign(row[6]) for row in b[field]],
                               gripper_FLASH_signs=[sign(row[6]) for row in c[field]],
                               gripper_sign_changed_steps=[i for i in range(12) if sign(c[field][i][6]) != sign(b[field][i][6])])
    contexts = {}
    for mode, record in records.items():
        episode_path = Path(record["result_evidence"]["path"])
        run = episode_path.parents[2]
        result, result_evidence = load(episode_path)
        manifest, manifest_evidence = load(run / "manifest.json")
        run_result, run_evidence = load(run / "result.json")
        environment, environment_evidence = load(episode_path.parent / "environment_identity.json")
        request = record["request"]
        contexts[mode] = dict(
            request_evidence=record["request_evidence"], result_evidence=result_evidence,
            manifest_evidence=manifest_evidence, run_evidence=run_evidence,
            environment_evidence=environment_evidence, environment_identity=environment,
            protocol=manifest["protocol"], base_checkpoint=manifest["base_checkpoint"],
            run_started_utc=datetime.fromtimestamp(run_result["started_unix"], timezone.utc).isoformat(),
            gpu_uuid=run_result["gpu"], preparation_seconds=run_result["preparation_seconds"],
            first_planned_episode=manifest["cases"][0], process_first_request=record["request_evidence"]["line"] == 1,
            first_replan_in_compared_episode=request["replan_id"] == 0,
            policy_wall_ms=request["policy_wall_ms"],
            timing_ms={k: request["timing"][k] for k in
                       ["sample_actions_ms", "encoder_ms", "vlm_prefill_ms", "full_fallback_ms", "total_ms"]},
            first_request_execution={k: request["timing"][k] for k in
                                     ["num_action_layers", "num_flow_steps", "denoise_steps", "denoise_times",
                                      "action_layer_indices_1based", "vlm_layer_indices_1based"]},
            route_calls=request["route_calls"],
            episode_config={k: result["episode"][k] for k in ["task_suite_name", "task_id", "episode_idx", "max_steps", "num_steps_wait", "seed"]},
            episode_success=result["success"])
    configuration_equal = {k: contexts["full"][k] == contexts["flash"][k] for k in
                           ["environment_identity", "protocol", "base_checkpoint", "first_request_execution", "route_calls", "episode_config"]}
    assert all(configuration_equal.values())
    assert all(not numerical[f]["gripper_sign_changed_steps"] for f in numerical)
    episodes, analysis_evidence = load(ROOT / "reports/ease_insights_20261103/interim/draft_analysis_snapshot/episodes.json")
    sensitivity = []
    expected = {(t, 20) for t in range(10)}
    for source in ("libero_spatial", "libero_object", "libero_goal"):
        rows = [r for r in episodes if r["mode"] == "flash" and r["source_suite"] == source and r["target_suite"] == "libero_spatial"]
        assert len(rows) == 10 and {(r["task_id"], r["init_id"]) for r in rows} == expected
        reduced = [r for r in rows if (r["task_id"], r["init_id"]) != (1, 20)]
        sensitivity.append(dict(source_suite=source, primary=summarize(rows),
                                same_case_removed_sensitivity=summarize(reduced),
                                removed_case=[r for r in rows if r["task_id"] == 1]))
    output = dict(generated_utc=datetime.now(timezone.utc).isoformat(),
                  audit_snapshot_utc="2026-11-03T13:28:58Z", case=dict(source_suite="libero_spatial", target_suite="libero_spatial", task_id=1, init_id=20),
                  snapshot_evidence=snapshot_evidence, numerical=numerical,
                  contexts=contexts, configuration_equal=configuration_equal,
                  sensitivity=dict(source=analysis_evidence, target_suite="libero_spatial",
                                   rule="Primary includes all 10 tasks. Sensitivity removes target Spatial task1/init20 from every compared source, leaving the same 9 cases.",
                                   formula="sum(accepted_actions) / (12 * sum(verified_rounds))",
                                   scope="Three source conditions complete in the existing fixed interim summary; Long-source partial rows are not pooled.",
                                   rows=sensitivity),
                  limits=["Primary science result includes the mismatching case; no rerun, exclusion, threshold change, or runtime modification.",
                          "The FLASH observation is the process-first cold-start request, but no explicit capture/compile marker establishes the cause of the numerical difference.",
                          "Full's compared episode also begins at replan0 but is request line9; its process started with task0, while this FLASH discovery run starts at task1 because task0 was covered by smoke.",
                          "XYZ/rotation values are decoded policy command components, not measured meter/radian trajectory deviations.",
                          "Unchanged gripper signs and the observed small component differences do not establish that success is unaffected; no causal success-impact claim is made.",
                          "Acceptance sensitivity is descriptive and cannot identify the cause or effect of the first-request mismatch."])
    (HERE / "materiality.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(dict(numerical={f: numerical[f]["group_max_abs"] for f in numerical},
                         configuration_equal=configuration_equal,
                         gripper_sign_changes={f: numerical[f]["gripper_sign_changed_steps"] for f in numerical},
                         sensitivity=[{k: r[k] for k in ["source_suite", "primary", "same_case_removed_sensitivity"]} for r in sensitivity])))


if __name__ == "__main__":
    main()
