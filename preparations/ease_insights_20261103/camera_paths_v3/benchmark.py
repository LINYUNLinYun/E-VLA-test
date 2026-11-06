"""Bounded fixed-observation cost probe; no closed-loop quality inference."""
import statistics
import time
import numpy as np
from .common import SUITES, atomic_json
from .policy import Policy

WARMUP_ROUNDS = 10
MEASURED_ROUNDS = 20


def summarize(rows):
    def costs(group):
        values = [r["policy_wall_ms"] for r in group]
        return dict(n=len(values), mean_ms=statistics.mean(values), p50_ms=statistics.median(values),
                    total_ms=sum(values)) if values else dict(n=0, mean_ms=None, p50_ms=None, total_ms=0.)
    cycles = [sum(r["policy_wall_ms"] for r in rows[i:i+5]) for i in range(0, len(rows), 5)]
    return dict(overall=costs(rows), full=costs([r for r in rows if r["refresh"]]),
                fast=costs([r for r in rows if not r["refresh"]]),
                five_request_cycle=dict(n=len(cycles), mean_ms=statistics.mean(cycles),
                    p50_ms=statistics.median(cycles), execute_actions=25))


def measure_condition(backend, config, record):
    policy = Policy(backend)
    policy.configure(config)
    warmup, rows = [], []
    for index in range(WARMUP_ROUNDS + MEASURED_ROUNDS):
        # Same suite/round seed for every condition; all use the same fixed obs.
        seed = (int(record["seed"]) + index) % (2**32)
        tick = time.perf_counter()
        response = policy.predict(record["obs"], record["instruction"], record["suite"], seed)
        policy.acknowledge(response["actions"][:5])
        elapsed = (time.perf_counter() - tick) * 1000
        expected_full = config["id"] == "full_d10" or index % 5 == 0
        if response["route"]["refresh"] != expected_full or response["round"] != index:
            raise ValueError("Cost probe schedule differs from frozen policy")
        row = dict(round=index, seed=seed, refresh=expected_full, policy_wall_ms=response["timing"]["policy_wall_ms"],
            predict_and_ack_wall_ms=elapsed, model_timing=response["timing"],
            model_noise_sha256=response["model_noise_sha256"],
            history_steps=response["route"]["history_steps"],
            camera_execution_audit=response["route"]["camera_execution_audit"])
        if not np.isfinite(row["policy_wall_ms"]) or row["policy_wall_ms"] <= 0:
            raise ValueError("Invalid cost probe latency")
        (warmup if index < WARMUP_ROUNDS else rows).append(row)
    return dict(condition=config["id"], config=config, warmup=warmup, measured=rows, summary=summarize(rows))


def run(backend, manifest, output):
    from preparations.idea1_v3_1.benchmark import load_banks
    records, identities = load_banks([manifest["bank"]["path"]])
    if identities[0]["sha256"] != manifest["bank"]["sha256"]:
        raise ValueError("Cost probe input bank changed")
    selected = [next(row for row in records if row.get("suite") == suite) for suite in SUITES]
    started = time.perf_counter()
    suites = []
    for suite_index, record in enumerate(selected):
        configs = manifest["conditions"]
        configs = configs[suite_index:] + configs[:suite_index]
        blocks = [measure_condition(backend, config, record) for config in configs]
        full_ms = next(b["summary"]["overall"]["mean_ms"] for b in blocks if b["condition"] == "full_d10")
        for block in blocks:
            block["full_mean_over_policy_mean"] = full_ms / block["summary"]["overall"]["mean_ms"]
        suites.append(dict(suite=record["suite"], input_hash=record["input_hash"], seed=int(record["seed"]),
            order=[c["id"] for c in configs], conditions=blocks))
    aggregate = []
    for config in manifest["conditions"]:
        rows = [row for suite in suites for block in suite["conditions"] if block["condition"] == config["id"] for row in block["measured"]]
        aggregate.append(dict(condition=config["id"], summary=summarize(rows)))
    full_ms = next(b["summary"]["overall"]["mean_ms"] for b in aggregate if b["condition"] == "full_d10")
    for block in aggregate:
        block["full_mean_over_policy_mean"] = full_ms / block["summary"]["overall"]["mean_ms"]
    result = dict(manifest_hash=manifest["manifest_hash"], bank=identities, warmup_rounds_per_condition_per_suite=WARMUP_ROUNDS,
        measured_rounds_per_condition_per_suite=MEASURED_ROUNDS, suites=suites, aggregate=aggregate,
        total_cost_probe_wall_s=time.perf_counter()-started, total_warmup_requests=len(selected)*len(manifest["conditions"])*WARMUP_ROUNDS,
        total_measured_requests=len(selected)*len(manifest["conditions"])*MEASURED_ROUNDS,
        latency_scope="Completed policy.predict including raw vision capture/copy and camera dispatch, model stages, decoding and output transfer; RPC/environment excluded",
        preparation_scope="Service loading and numerical gate precede this probe and are reported separately; ten warmup rounds excluded from measured costs",
        pairing="One real fixed observation per suite; condition reset; identical round/seed sequence; rotated condition order across suites",
        interpretation="Fixed-input inference cost only; no acceptance, success, rollout speed, or quality evidence. Small timing differences remain uncertain; no significance claim.",
        known_limit="Repeated fixed observations check actual vision dispatch and full policy cost, not temporal camera-cache quality")
    atomic_json(output / "fixed_input_cost.json", result)
    return result
