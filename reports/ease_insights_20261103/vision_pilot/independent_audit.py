#!/usr/bin/env python3
"""Independent CPU-only audit of the frozen init20 vision pilot."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "results/ease_insights_20261103/fastpath"
OUT = Path(__file__).resolve().parent
CONDITIONS = ("pi0_full", "pi0_keep24_2_front_middle", "pi0_keep24_1_uniform")
SKIPS = ([], [8, 9, 10], [5, 14, 23])
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def main():
    manifest = json.loads((RUN / "manifest.json").read_text())
    mh = manifest["manifest_hash"]
    assert sha_json({k: v for k, v in manifest.items() if k != "manifest_hash"}) == mh
    assert manifest["screen_inits"] == [20]
    assert manifest["preregistered_hypotheses"] == [
        {"suite": "libero_goal", "better": CONDITIONS[1], "worse": CONDITIONS[2]},
        {"suite": "libero_10", "better": CONDITIONS[2], "worse": CONDITIONS[1]},
    ]
    for cid in CONDITIONS:
        markers = list((RUN / cid).glob("pilot*_complete.json"))
        assert markers, f"Pilot complete marker absent: {cid}"
        assert all(json.loads(p.read_text())["manifest_hash"] == mh for p in markers)
    outcomes = {}
    initial_pairs = {}
    sources = []
    request_counts = {}
    reference_protocol = None
    compare_protocol = ("model", "checkpoint", "backend", "H", "R", "D", "vision_depth",
                        "vlm_depth", "action_expert", "vision_compression", "vision_tokens", "openpi_sources")
    for cid, skip in zip(CONDITIONS, SKIPS):
        outcomes[cid] = {}
        request_counts[cid] = 0
        for suite in SUITES:
            for task in range(10):
                scene = {"suite": suite, "task": task, "init": 20}
                d = RUN / cid / "episodes" / suite / f"task{task:02d}" / "init20"
                receipt = json.loads((d / "episode_receipt.json").read_text())
                result = json.loads((d / "result.json").read_text())
                identity = json.loads((d / "identity.json").read_text())
                complete = json.loads((d / "complete.json").read_text())
                assert receipt["valid"] and result["valid"] and complete["valid"]
                assert receipt["case"] == scene and receipt["condition"] == cid
                assert all(x["manifest_hash"] == mh for x in (receipt, result, identity, complete))
                assert result["success"] == receipt["success"]
                assert all(result[k] == v and identity[k] == v for k, v in scene.items())
                assert result["condition"] == cid and identity["condition"]["id"] == cid
                for name, expected in receipt["artifacts"].items():
                    assert sha(d / name) == expected, d / name
                p = identity["protocol"]
                assert (p["D"], p["H"], p["R"], p["vision_depth"], p["vlm_depth"]) == (10, 50, 5, 27, 18)
                assert p["skip"] == skip
                numeric = {key: p[key] for key in compare_protocol}
                if reference_protocol is None:
                    reference_protocol = numeric
                assert numeric == reference_protocol
                assert identity["env_seed"] == 7 and identity["control_hz"] == 20
                requests = [json.loads(line) for line in (d / "infer.jsonl").read_text().splitlines()]
                assert len(requests) == result["predictions"]
                for r in requests:
                    expected_seed = int(sha_json(["idea1-v3.1", 1701, suite, task, 20, r["request"]])[:8], 16)
                    assert r["seed"] == expected_seed
                    assert r["audit"]["NFE"] == 10 and r["audit"]["vision_skip"] == skip
                    assert r["audit"]["original_scanned_ae"]
                    assert len(r["actions"]) == 50 and r["execute_length"] == 5
                    assert 1 <= r["executed_length"] <= 5
                first = requests[0]
                gate = receipt["initial_gate"]
                assert gate["input_hash"] == first["input_hash"] and gate["seed"] == first["seed"]
                paired_identity = {"input_hash": first["input_hash"], "noise_hash": first["noise_hash"],
                                   "seed": first["seed"], "init_vector_sha256": identity["init_vector_sha256"],
                                   "bddl_sha256": identity["bddl_sha256"], "controller": identity["controller"]}
                key = (suite, task)
                if key in initial_pairs:
                    assert initial_pairs[key] == paired_identity
                else:
                    initial_pairs[key] = paired_identity
                outcomes[cid][key] = bool(result["success"])
                request_counts[cid] += len(requests)
                sources.append({"receipt": str(d / "episode_receipt.json"), "sha256": sha(d / "episode_receipt.json")})
    suite_counts = {suite: {cid: sum(outcomes[cid][(suite, task)] for task in range(10)) for cid in CONDITIONS}
                    for suite in SUITES}
    a, b = CONDITIONS[1:]
    discordant = [{"suite": suite, "task": task, "init": 20, "A_success": outcomes[a][(suite, task)],
                   "B_success": outcomes[b][(suite, task)], "Full_success": outcomes[CONDITIONS[0]][(suite, task)]}
                  for suite in SUITES for task in range(10) if outcomes[a][(suite, task)] != outcomes[b][(suite, task)]]
    report = {"valid": True, "manifest_hash": mh, "init": 20, "paired_cases": 40,
              "suite_successes_out_of_10": suite_counts,
              "totals_out_of_40": {cid: sum(rows.values()) for cid, rows in outcomes.items()},
              "predeclared_directions": {"Goal_A_gt_B": suite_counts["libero_goal"][a] > suite_counts["libero_goal"][b],
                                         "Long_B_gt_A": suite_counts["libero_10"][b] > suite_counts["libero_10"][a]},
              "discordant_cases": discordant, "requests_checked": request_counts,
              "common_numeric_protocol": reference_protocol, "receipt_sources": sources,
              "initial_input_noise_init_bddl_controller_matched_cases": len(initial_pairs),
              "limitations": ["One new official initial state per task: preliminary independent replication only.",
                              "Candidates and Goal/Long preference directions are held fixed from historical init0 discovery.",
                              "Closed-loop request latencies do not establish same-input computation speedup."],
              "audit_source_sha256": sha(__file__)}
    (OUT / "independent_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("valid", "totals_out_of_40", "suite_successes_out_of_10", "predeclared_directions", "discordant_cases")}, indent=2))


def sha_json(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


if __name__ == "__main__":
    main()
