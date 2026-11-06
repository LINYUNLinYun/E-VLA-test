"""pi0.5 vision encoder A/B compression experiment driver.

Stages (experiment plan sections 11-13):
  audit -> check -> collect_obs -> bench -> smoke -> screen -> confirm -> report
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MODEL_PYTHON = ROOT / "policies/openpi/.venv/bin/python"
EVAL_PYTHON = ROOT / "policies/libero-eval-v21/.venv/bin/python"
CHECKPOINT = ROOT / "shared/checkpoints/openpi/openpi-assets/checkpoints/pi05_libero"
OPENPI_DATA_HOME = ROOT / "shared/checkpoints/openpi"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_config(path):
    import yaml

    return yaml.safe_load(Path(path).read_text())


def conditions(cfg):
    out = []
    for item in cfg["conditions"]:
        out.append({"id": item["id"], "mode": item["mode"],
                    "keep_tokens": int(item["keep_tokens"])})
    return out


def results_root(cfg):
    return Path(cfg["paths"]["results_root"])


def make_run_id():
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def run_dir_for(cfg, run_id):
    if run_id is None:
        candidates = sorted(p for p in results_root(cfg).glob("*") if (p / "manifest.json").exists())
        if not candidates:
            raise SystemExit("no run directory; run --stage audit first")
        return candidates[-1]
    return results_root(cfg) / run_id


def model_env(gpu):
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        OPENPI_DATA_HOME=str(OPENPI_DATA_HOME),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        PYTHONUNBUFFERED="1",
    )
    env["PYTHONPATH"] = ":".join(
        [str(ROOT), str(ROOT / "policies/openpi/src"), env.get("PYTHONPATH", "")]
    )
    return env


def eval_env(gpu):
    from preparations.idea1_v2_2.common import evaluation_env

    env = evaluation_env(gpu)
    env["PYTHONPATH"] = ":".join([str(ROOT), env.get("PYTHONPATH", "")])
    env["PYTHONUNBUFFERED"] = "1"
    return env


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, default=_json_default))
    os.replace(tmp, path)


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path):
    rows = []
    with Path(path).open() as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_server(condition, gpu, log_path):
    from preparations.idea1_v2_1.runtime_client import rpc

    port = free_port()
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a")
    config = json.dumps({"condition_id": condition["id"], "mode": condition["mode"],
                         "keep_tokens": condition["keep_tokens"]})
    proc = subprocess.Popen(
        [str(MODEL_PYTHON), "-m", "experiments.pi05_vision_ab.server",
         "--port", str(port), "--config", config, "--launch"],
        cwd=ROOT, env=model_env(gpu), stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.monotonic() + 2400
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"model server exited; see {log_path}")
        try:
            health = rpc(port, {"op": "health"})
            return proc, port, log, health["protocol"]
        except (OSError, RuntimeError):
            if time.monotonic() > deadline:
                raise TimeoutError("model server startup")
            time.sleep(2)


def stop_server(proc, log):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    if log:
        log.close()


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def stage_audit(cfg, args):
    run_id = args.run_id or make_run_id()
    run_dir = results_root(cfg) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    import hashlib

    def sha(path):
        h = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()

    repo = ROOT / "policies/openpi"
    git = {}
    for key, command in {
        "commit": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        "status": ["git", "status", "--porcelain"],
    }.items():
        proc = subprocess.run(command, cwd=repo, capture_output=True, text=True)
        git[key] = proc.stdout.strip()
    diff = subprocess.run(["git", "diff"], cwd=repo, capture_output=True, text=True).stdout
    (run_dir / "implementation_diff.patch").write_text(diff)

    norm_stats = CHECKPOINT / "assets/physical-intelligence/libero/norm_stats.json"
    checkpoint = {
        "path": str(CHECKPOINT),
        "params_manifest": str(CHECKPOINT / "params/manifest.ocdbt"),
        "metadata_bytes": (CHECKPOINT / "params/_METADATA").stat().st_size
        if (CHECKPOINT / "params/_METADATA").exists() else None,
        "norm_stats": str(norm_stats),
        "norm_stats_sha256": sha(norm_stats),
    }

    versions = {}
    code = (
        "import json,sys;"
        "import jax,jaxlib;"
        "print(json.dumps(dict(python=sys.version,jax=jax.__version__,jaxlib=jaxlib.__version__,"
        "devices=[dict(kind=d.device_kind,platform=d.platform) for d in jax.devices()])))"
    )
    proc = subprocess.run([str(MODEL_PYTHON), "-c", code], cwd=ROOT,
                          env=model_env(args.gpu), capture_output=True, text=True)
    versions["model_env"] = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.returncode == 0 else proc.stderr[-2000:]

    code = (
        "import json,sys;"
        "import libero,robosuite,mujoco,numpy;"
        "from libero.libero import benchmark;"
        f"names={list(SUITES)!r};"
        "classes=benchmark.get_benchmark_dict();"
        "counts={k:len(classes[k]().tasks) for k in names};"
        "print('VERSIONS_JSON='+json.dumps(dict(python=sys.version,numpy=numpy.__version__,"
        "robosuite=robosuite.__version__,mujoco=mujoco.__version__,tasks=counts)))"
    )
    proc = subprocess.run([str(EVAL_PYTHON), "-c", code], cwd=ROOT,
                          env=eval_env(args.gpu), capture_output=True, text=True)
    line = next((x for x in reversed(proc.stdout.splitlines()) if x.startswith("VERSIONS_JSON=")), None)
    versions["eval_env"] = json.loads(line.split("=", 1)[1]) if line else {
        "error": (proc.stderr or proc.stdout)[-3000:], "returncode": proc.returncode}

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,power.limit,clocks.max.sm",
         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().splitlines()

    resolved = json.loads(json.dumps(cfg))
    resolved["resolved"] = {
        "run_id": run_id,
        "checkpoint": checkpoint,
        "backend": cfg["model"]["backend"],
        "dtype": cfg["model"]["dtype"],
        "compile_mode": cfg["model"]["compile_mode"],
        "action_horizon": cfg["policy"]["action_horizon"],
        "action_dim": cfg["policy"]["action_dim"],
        "num_steps": cfg["policy"]["num_steps"],
        "replan_steps": cfg["policy"]["replan_steps"],
        "conditions": conditions(cfg),
        "python": {"model": str(MODEL_PYTHON), "eval": str(EVAL_PYTHON)},
        "git": git,
        "gpu": gpu,
        "environment": versions,
        "code_sha256": {
            "siglip.py": sha(repo / "src/openpi/models/siglip.py"),
            "vision_compression.py": sha(repo / "src/openpi/models/vision_compression.py"),
            "experiment": {p.name: sha(p) for p in EXP_DIR.glob("*.py")},
        },
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(run_dir / "manifest.json", resolved)
    (run_dir / "resolved_config.yaml").write_text(
        json.dumps(resolved, indent=2, default=_json_default))
    (run_dir / "environment.txt").write_text(
        json.dumps({"model_env": versions["model_env"], "eval_env": versions["eval_env"],
                    "gpu": gpu, "git": git}, indent=2))
    (run_dir / "commands.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {ROOT}\n"
        f"python {EXP_DIR}/run.py --stage audit --run-id {run_id}\n"
        f"python {EXP_DIR}/run.py --stage check --run-id {run_id} --gpu {args.gpu}\n"
        f"python {EXP_DIR}/run.py --stage collect_obs --run-id {run_id} --gpu {args.gpu}\n"
        f"python {EXP_DIR}/run.py --stage bench --run-id {run_id} --gpu {args.gpu}\n"
        f"python {EXP_DIR}/run.py --stage smoke --run-id {run_id} --gpus {args.gpu}\n"
        f"python {EXP_DIR}/run.py --stage screen --run-id {run_id} --gpus 0,2,3\n"
        f"python {EXP_DIR}/run.py --stage select --run-id {run_id}\n"
        f"python {EXP_DIR}/run.py --stage confirm --run-id {run_id} --gpus 0,2,3\n"
        f"python {EXP_DIR}/run.py --stage report --run-id {run_id}\n")
    print(run_id)
    return run_id


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

def stage_check(cfg, args, run_dir):
    from experiments.pi05_vision_ab import checks

    cpu = checks.cpu_checks()
    parity_dir = run_dir / "parity"
    parity_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for mode, keep in (("off", 256), ("A", 256)):
        name = f"{mode}_{keep}"
        out = parity_dir / f"{name}.npz"
        if not out.exists():
            command = [str(MODEL_PYTHON), "-m", "experiments.pi05_vision_ab.run",
                       "_parity", "--mode", mode, "--keep", str(keep), "--output", str(out)]
            subprocess.run(command, cwd=ROOT, env=model_env(args.gpu), check=True)
        outputs[name] = out

    sanity = {}
    for condition in conditions(cfg):
        if condition["mode"] == "off":
            continue
        out = parity_dir / f"sanity_{condition['id']}.json"
        if not out.exists():
            env = model_env(args.gpu)
            env["PI05_VISION_AB_TRACE"] = "1"
            command = [str(MODEL_PYTHON), "-m", "experiments.pi05_vision_ab.run",
                       "_sanity", "--mode", condition["mode"],
                       "--keep", str(condition["keep_tokens"]), "--output", str(out)]
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        sanity[condition["id"]] = json.loads(out.read_text())

    atol = 1e-5 if cfg["model"]["dtype"] == "float32" else 1e-2
    parity = checks.compare_parity(outputs["off_256"], outputs["A_256"], atol=atol)
    feature_shape = parity["feature_shape"]
    action_shape = parity["action_shape"]
    report = {
        "cpu": cpu,
        "encoder_and_policy_identity_K256": parity,
        "checks": {
            "mapping_matches_independent_reference": all(
                v["exact_match"] for v in cpu["mapping_vs_numpy_reference"].values()),
            "hand_tensor": all(cpu["hand_tensor"].values()),
            "dense_residual": all(cpu[f"dense_residual_{k}"]["same_group_same_update"]
                                  and cpu[f"dense_residual_{k}"]["original_differences_preserved"]
                                  for k in (128, 64)),
            "kN_identity": cpu["kN_identity"]["g_is_arange"] and cpu["kN_identity"]["bit_exact"],
            "encoder_identity": parity["within_tolerance"]
                                and parity["finite_off"] and parity["finite_k256"],
            "policy_identity": parity["within_tolerance"]
                               and parity["finite_off"] and parity["finite_k256"],
            "params_unchanged": parity["params_identical"],
            "shapes": feature_shape[1:] == [3, 1, 256, feature_shape[-1]]
                      and feature_shape[-1] > 0
                      and action_shape[1:] == [1, 10, 32],
            "compressed_finite": all(
                v["finite"] and v["action_shape"][-2:] == [10, 32]
                and all(s == [1, 256, feature_shape[-1]]
                        for fs in v["feature_shapes"] for s in fs)
                for v in sanity.values()),
            "compressed_actual_k_compute": all(
                v["k_length_compute_observed"] and v["unmerge_restores_256"]
                for v in sanity.values()),
        },
        "compressed_sanity": sanity,
        "repeat_run_noise": {
            "feature_max_abs": parity["repeat_feature_max_abs"],
            "action_max_abs": parity["repeat_action_max_abs"],
        },
    }
    report["all_required_passed"] = all(report["checks"].values())
    write_json(run_dir / "correctness.json", report)
    print(json.dumps(report, indent=2))
    if not report["all_required_passed"]:
        raise SystemExit("correctness checks failed; long rollouts are blocked")


# ---------------------------------------------------------------------------
# observation collection
# ---------------------------------------------------------------------------

def _store_observation(obs_dir, index, suite, task, init, replan, instruction, images, state,
                       source):
    import numpy as np

    filename = f"obs_{index:04d}.npz"
    np.savez_compressed(
        obs_dir / filename,
        agentview_image=np.asarray(images[0], dtype=np.uint8),
        wrist_image=np.asarray(images[1], dtype=np.uint8),
        state=np.asarray(state, dtype=np.float32),
    )
    return {
        "observation_id": f"obs_{index:04d}",
        "suite": suite,
        "task_id": task,
        "initial_state_index": init,
        "replan": replan,
        "instruction": instruction,
        "path": filename,
        "source": source,
    }


def _existing_p0_observations():
    import hashlib
    import pickle

    from preparations.idea1_v2_1.runtime_models import native_observation

    pattern = "policies/realtime-vla-flash/results/idea1_v3/p0_*/episodes/E1/*/*/*/task*/init*/profile_inputs.pkl"
    result = []
    seen = set()
    for path in sorted(ROOT.glob(pattern)):
        for item in pickle.loads(path.read_bytes()):
            images, state = native_observation(item["obs"])
            digest = hashlib.sha256(
                images[0].tobytes() + images[1].tobytes() + state.tobytes()
                + item["instruction"].encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            result.append((item["suite"], int(path.parent.parent.name.replace("task", "")),
                           int(path.parent.name.replace("init", "")), 0,
                           item["instruction"], images, state, str(path.relative_to(ROOT))))
    return result


def stage_collect_obs(cfg, args, run_dir):
    import numpy as np

    obs_dir = run_dir / "observations"
    obs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "observations_manifest.jsonl"
    if manifest_path.exists():
        existing = read_jsonl(manifest_path)
        if len(existing) >= int(cfg["benchmark"]["real_observations_min"]):
            print(f"observation manifest already has {len(existing)} rows")
            return
    rows = []
    for suite, task, init, replan, instruction, images, state, source in _existing_p0_observations():
        rows.append(_store_observation(obs_dir, len(rows), suite, task, init, replan,
                                       instruction, images, state, source))

    # Additional real observations from C0 rollouts, sampled across each
    # episode's replans.  Two tasks per suite; 0/2/4/8/16/32/48/64 spans early,
    # approach/contact and late stages for the longer suites.
    from experiments.pi05_vision_ab.policy_adapter import _native_observation
    from preparations.idea1_v2_1.runtime_client import rpc

    c0 = {"id": "C0", "mode": "off", "keep_tokens": 256}
    proc, port, log, protocol = start_server(c0, args.gpu, run_dir / "logs/collect_server.log")
    try:
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        from experiments.pi05_vision_ab.evaluation_adapter import (
            LIMITS, NUM_STEPS_WAIT, REPLAN_STEPS, init_states, make_noise,
        )

        sample_replans = {0, 2, 4, 8, 16, 32, 48, 64}
        for suite_name in SUITES:
            for task_id in (0, 3, 7):
                suite, initial_states = init_states(suite_name, task_id)
                definition = suite.get_task(task_id)
                bddl = Path(get_libero_path("bddl_files")) / definition.problem_folder / definition.bddl_file
                env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
                try:
                    env.seed(7)
                    env.reset()
                    obs = env.set_init_state(initial_states[0])
                    for _ in range(NUM_STEPS_WAIT):
                        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
                    rpc(port, {"op": "reset"})
                    replan = 0
                    steps = 0
                    while steps < LIMITS[suite_name]:
                        noise = make_noise(suite_name, task_id, 0, replan)
                        model_obs = {k: np.asarray(v) for k, v in obs.items()
                                     if isinstance(v, np.ndarray)}
                        if replan in sample_replans:
                            images, state = _native_observation(obs)
                            rows.append(_store_observation(
                                obs_dir, len(rows), suite_name, task_id, 0, replan,
                                definition.language, images, state,
                                f"collect/{suite_name}/task{task_id:02d}/replan{replan:03d}"))
                        response = rpc(port, {"op": "predict", "obs": model_obs,
                                              "instruction": definition.language,
                                              "suite": suite_name, "noise": noise})
                        actions = np.asarray(response["actions"], dtype=np.float32)
                        done = False
                        for action in actions[:REPLAN_STEPS]:
                            obs, _, done, _ = env.step(action.tolist())
                            steps += 1
                            if done:
                                break
                        replan += 1
                        if done:
                            break
                finally:
                    env.close()
    finally:
        stop_server(proc, log)

    write_jsonl(manifest_path, rows)
    counts = {suite: sum(1 for r in rows if r["suite"] == suite) for suite in SUITES}
    print(json.dumps({"observations": len(rows), "per_suite": counts}, indent=2))
    if len(rows) < int(cfg["benchmark"]["real_observations_min"]) or min(counts.values()) < int(
            cfg["benchmark"]["observations_per_suite_min"]):
        raise SystemExit("observation budget not met")


def _load_observations(run_dir):
    import numpy as np

    rows = []
    for row in read_jsonl(run_dir / "observations_manifest.jsonl"):
        data = np.load(run_dir / "observations" / row["path"])
        rows.append({**row, "raw": {
            "agentview_image": data["agentview_image"],
            "wrist_image": data["wrist_image"],
            "state": data["state"],
            "instruction": row["instruction"],
        }})
    return rows


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------

def _bench_worker(args, cfg, run_dir):
    from experiments.pi05_vision_ab import bench

    observations = _load_observations(run_dir)
    condition = [c for c in conditions(cfg) if c["id"] == args.condition]
    if not condition:
        raise SystemExit(f"unknown condition {args.condition}")
    meta = bench.measure_condition(condition[0], args.round, observations, args.output,
                                   args.gpu, args.run_id,
                                   warmup=int(cfg["benchmark"]["warmup_min"]),
                                   calls=int(cfg["benchmark"]["measured_calls_per_round"]))
    print(json.dumps({k: meta[k] for k in ("config_id", "round", "compile_vision_s",
                                           "compile_model_s", "load_s")}, indent=2))


def stage_bench(cfg, args, run_dir):
    conds = [c["id"] for c in conditions(cfg)]
    rounds = int(cfg["benchmark"]["rounds"])
    raw_dir = run_dir / "latency_parts"
    raw_dir.mkdir(parents=True, exist_ok=True)
    progress = run_dir / "bench_progress.jsonl"
    done = set()
    if progress.exists():
        for row in read_jsonl(progress):
            done.add((row["round"], row["condition"]))
    for r in range(rounds):
        order = conds[r % len(conds):] + conds[: r % len(conds)]
        for condition in order:
            if (r, condition) in done:
                continue
            output = raw_dir / f"r{r}_{condition}.csv"
            command = [str(MODEL_PYTHON), "-m", "experiments.pi05_vision_ab.run", "_bench-worker",
                       "--run-id", args.run_id, "--condition", condition, "--round", str(r),
                       "--gpu", str(args.gpu), "--output", str(output)]
            subprocess.run(command, cwd=ROOT, env=model_env(args.gpu), check=True)
            with progress.open("a") as stream:
                stream.write(json.dumps({"round": r, "condition": condition}) + "\n")
    # Combine in a stable order.
    fields = None
    all_rows = []
    metas = []
    for r in range(rounds):
        for condition in conds:
            path = raw_dir / f"r{r}_{condition}.csv"
            with path.open() as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                all_rows.extend(reader)
            metas.append(json.loads(Path(str(path) + ".meta.json").read_text()))
    with (run_dir / "latency_raw.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    write_json(run_dir / "latency_compile_meta.json", metas)
    print(f"wrote {len(all_rows)} latency rows")


# ---------------------------------------------------------------------------
# rollouts
# ---------------------------------------------------------------------------

def _prepare_rollout_manifest(cfg, run_dir, stage):
    from experiments.pi05_vision_ab.evaluation_adapter import build_manifest

    path = run_dir / f"episode_manifest_{stage}.jsonl"
    if path.exists():
        rows = read_jsonl(path)
        print(f"{stage} manifest already has {len(rows)} episodes")
        return rows
    conds = conditions(cfg)
    if stage == "smoke":
        rows = build_manifest("smoke", conds,
                              smoke_tasks=int(cfg["evaluation"]["smoke_tasks_per_suite"]))
    elif stage == "screen":
        span = cfg["evaluation"]["screen_init_indices"]
        rows = build_manifest("screen", conds, screen=(span["start"], span["stop_exclusive"]))
    elif stage == "confirm":
        selected = args_selected_conditions(run_dir)
        span = cfg["evaluation"]["confirm_init_indices"]
        rows = build_manifest("confirm", selected,
                              confirm=(span["start"], span["stop_exclusive"]))
    else:
        raise ValueError(stage)
    write_jsonl(path, rows)
    print(f"wrote {len(rows)} episodes to {path}")
    return rows


def args_selected_conditions(run_dir):
    selection = json.loads((run_dir / "selection.json").read_text())
    ids = selection["confirm_conditions"]
    all_conditions = {c["id"]: c for c in json.loads(
        (run_dir / "manifest.json").read_text())["resolved"]["conditions"]}
    return [all_conditions[i] for i in ids]


def stage_rollout(cfg, args, run_dir):
    from experiments.pi05_vision_ab.evaluation_adapter import (
        append_jsonl, load_done_keys, run_episode,
    )

    stage = args.stage
    rows = read_jsonl(run_dir / f"episode_manifest_{stage}.jsonl")
    workers = args.workers
    rows = [row for row in rows
            if int(row["key"][:8], 16) % workers == args.shard]
    rows.sort(key=lambda row: (row["condition_id"], row["task_rank"], row["initial_state_index"]))
    log_path = run_dir / f"rollouts_{stage}.jsonl"
    done = load_done_keys(log_path)
    if done:
        print(f"resume: {len(done)} completed episodes")
    current = None
    proc = log = None
    protocol = None
    try:
        for row in rows:
            if row["key"] in done:
                continue
            if current != row["condition_id"]:
                stop_server(proc, log)
                current = row["condition_id"]
                proc, port, log, protocol = start_server(
                    row["config"], args.gpu,
                    run_dir / f"logs/{stage}_server_{current}.log")
                print(f"[{stage} shard{args.shard}] server {current} ready port={port}", flush=True)
            try:
                record = run_episode(row, port, root_seed=args.root_seed)
            except Exception:
                record = {
                    "key": row["key"], "stage": stage, "condition_id": row["condition_id"],
                    "suite": row["suite"], "task_id": row["task_id"],
                    "initial_state_index": row["initial_state_index"],
                    "environment_seed": row["environment_seed"],
                    "repeat_seed": row["repeat_seed"],
                    "success": False, "environment_steps": None, "policy_calls": None,
                    "timeout": False, "error_type": "infrastructure",
                    "error": traceback.format_exc(), "protocol": protocol,
                }
                append_jsonl(log_path, record)
                raise
            record["protocol"] = protocol
            append_jsonl(log_path, record)
            done.add(row["key"])
            print(f"[{stage} shard{args.shard}] {row['suite']}/t{row['task_id']}/i"
                  f"{row['initial_state_index']} {row['condition_id']} "
                  f"success={record['success']} steps={record['environment_steps']}", flush=True)
    finally:
        stop_server(proc, log)
    write_json(run_dir / f"workers/{stage}_shard{args.shard}_done.json",
               {"rows": len(rows), "done": len(done), "time": time.time()})


def stage_launch_rollouts(cfg, args, run_dir):
    stage = args.stage
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    _prepare_rollout_manifest(cfg, run_dir, stage)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for shard, gpu in enumerate(gpus):
        command = [str(EVAL_PYTHON), "-m", "experiments.pi05_vision_ab.run", "_rollout",
                   "--stage", stage, "--run-id", args.run_id, "--shard", str(shard),
                   "--workers", str(len(gpus)), "--gpu", str(gpu),
                   "--root-seed", str(args.root_seed)]
        log = (log_dir / f"{stage}_shard{shard}.log").open("a")
        proc = subprocess.Popen(command, cwd=ROOT, env=eval_env(gpu), stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        procs.append((shard, proc, log))
        print(f"launched {stage} shard{shard} gpu{gpu} pid={proc.pid}")
    if args.wait:
        for shard, proc, log in procs:
            code = proc.wait()
            log.close()
            print(f"shard{shard} exit={code}")
            if code:
                raise SystemExit(f"shard{shard} failed")
    else:
        write_json(run_dir / f"workers/{stage}_launched.json",
                   {"pids": [p.pid for _, p, _ in procs], "time": time.time(),
                    "gpus": gpus, "stage": stage})


# ---------------------------------------------------------------------------
# parity / entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(EXP_DIR / "config.yaml"))
    parser.add_argument("--stage", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--root-seed", type=int, default=7)
    # hidden worker args
    parser.add_argument("_command", nargs="?")
    parser.add_argument("--mode")
    parser.add_argument("--keep", type=int)
    parser.add_argument("--output")
    parser.add_argument("--condition")
    parser.add_argument("--round", type=int, default=0)
    args = parser.parse_args()

    if args._command == "_parity":
        from experiments.pi05_vision_ab import checks
        checks.run_parity_condition(args.mode, args.keep, args.output)
        return
    if args._command == "_sanity":
        from experiments.pi05_vision_ab import checks
        checks.run_sanity_condition(args.mode, args.keep, args.output)
        return
    if args._command == "_bench-worker":
        cfg = load_config(args.config)
        run_dir = run_dir_for(cfg, args.run_id)
        _bench_worker(args, cfg, run_dir)
        return
    if args._command == "_rollout":
        cfg = load_config(args.config)
        run_dir = run_dir_for(cfg, args.run_id)
        stage_rollout(cfg, args, run_dir)
        return

    if not args.stage:
        raise SystemExit("--stage is required")
    cfg = load_config(args.config)
    if args.stage == "audit":
        stage_audit(cfg, args)
        return
    run_dir = run_dir_for(cfg, args.run_id)
    if args.stage == "check":
        stage_check(cfg, args, run_dir)
    elif args.stage == "collect_obs":
        stage_collect_obs(cfg, args, run_dir)
    elif args.stage == "bench":
        stage_bench(cfg, args, run_dir)
    elif args.stage in ("smoke", "screen", "confirm"):
        stage_launch_rollouts(cfg, args, run_dir)
    elif args.stage == "select":
        from experiments.pi05_vision_ab import summarize
        selection = summarize.build_selection(cfg, run_dir)
        print(json.dumps(selection["confirm_conditions"]))
    elif args.stage == "report":
        from experiments.pi05_vision_ab import summarize
        summarize.build_report(cfg, run_dir)
    elif args.stage == "system":
        write_json(run_dir / "system_evaluation.json",
                   {"status": "not evaluated",
                    "reason": "no runnable pi0.5 self-speculative draft/verifier harness"})
    else:
        raise SystemExit(f"unknown stage {args.stage}")


if __name__ == "__main__":
    main()
