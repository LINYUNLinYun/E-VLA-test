#!/usr/bin/env python3
"""Read-only E1 preparation inventory. Never imports model code or runs rollouts."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
LIMITS = dict(zip(SUITES, (280, 220, 300, 520)))
MODELS = {
    "xiaomi-robotic0": {"origin": "https://github.com/XiaomiRobotics/Xiaomi-Robotics-0", "budget": [(8, 3), (10, 4), (12, 5)], "depth": 36,
        "sources": ["eval_libero/main.py", "deploy/server.py", "scripts/launch_libero.sh"], "modules": ["torch", "transformers", "flash_attn", "numpy", "PIL"], "client_modules": ["tyro", "libero", "imageio"]},
    "db-cogact": {"origin": "https://github.com/dexmal/dexbotic", "budget": [(7, 3), (8, 4), (9, 5)], "depth": 28,
        "sources": ["dexbotic/exp/cogact_exp.py", "dexbotic/exp/base_exp.py", "dexbotic/model/cogact/cogact_arch.py", "dexbotic/model/dexbotic_arch.py", "dexbotic/policy/cogact_policy.py", "dexbotic/policy/types.py", "playground/benchmarks/libero/libero_cogact.py", "dexbotic/sim_envs/libero/libero_env.py"], "modules": ["torch", "transformers", "numpy"], "optional_modules": ["flash_attn"]},
    "cogact": {"origin": "https://github.com/microsoft/CogACT", "budget": [(8, 3), (9, 4), (11, 5)], "depth": 32,
        "sources": ["README.md", "pyproject.toml"], "modules": ["torch", "transformers", "timm", "numpy"]},
    "isaac-gr00t": {"origin": "https://github.com/NVIDIA/Isaac-GR00T", "budget": [(4, 2), (5, 2)], "depth": None,
        "sources": ["gr00t/model/modules/qwen3_backbone.py", "gr00t/eval/sim/LIBERO/libero_env.py", "tests/examples/test_libero.py"], "modules": ["torch", "transformers", "numpy"]},
}
AUDIT_TERMS = ("replan", "num_trials", "num_trails", "num_workers", "seed", "num_ddim", "num_steps=", "cfg_scale", "chunk_size", "select_layer", "language_model.layers", "trust_remote_code", "conda activate", "fps=", "start_index", "deepstack", "inference_action", "use_cache", "out_features.logits", "AutoModel")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(repo, *args):
    result = subprocess.run(["git", "-c", f"safe.directory={repo.resolve()}", "--no-optional-locks", "-C", str(repo), *args], capture_output=True, text=True, timeout=30)
    return result.stdout.strip() if result.returncode == 0 else None


def file_record(path):
    return {"path": str(path.resolve()), "sha256": digest(path), "bytes": path.stat().st_size}


def source_audit(repo, paths):
    records = []
    for name in paths:
        path = repo / name
        if not path.is_file():
            records.append({"path": str(path), "missing": True})
            continue
        record = file_record(path)
        record["evidence"] = [{"line": i, "text": line.strip()} for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1) if any(term in line for term in AUDIT_TERMS)]
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text())
                record["syntax_valid"] = True
            except (SyntaxError, UnicodeError) as exc:
                record["syntax_valid"] = False
                record["syntax_error"] = str(exc)
        records.append(record)
    return records


# -I -S prevents user site, PYTHONPATH and executable .pth processing.
# PathFinder only locates a top-level module: no import, CUDA init, or model code.
ENV_PROBE = '''import importlib.machinery, importlib.metadata, json, sys
from pathlib import Path
env = Path(sys.argv[2])
paths = [str(env / "lib" / ("python%d.%d" % sys.version_info[:2]) / "site-packages")]
mods = json.loads(sys.argv[1])
found = {m: importlib.machinery.PathFinder.find_spec(m, paths) is not None for m in mods}
versions = {d.metadata.get("Name", "unknown"): d.version for d in importlib.metadata.distributions(path=paths)}
print(json.dumps({"python": sys.version, "executable": sys.executable, "site_paths": paths, "module_discoverable": found, "distributions": versions}))
'''


def environment(repo, modules):
    env = repo / ".venv"
    cfg = env / "pyvenv.cfg"
    result = {"path": str(env), "exists": env.is_dir(), "expected_prompt": repo.name,
              "import_execution": "not performed; static module discovery does not prove binary importability"}
    if cfg.is_file():
        values = dict(line.split("=", 1) for line in cfg.read_text().splitlines() if "=" in line)
        values = {k.strip(): v.strip().strip("'\"") for k, v in values.items()}
        result["config"] = values
        result["prompt_matches"] = values.get("prompt") == repo.name
        result["config_source"] = file_record(cfg)
    executable = env / "bin/python"
    if executable.is_file():
        try:
            proc = subprocess.run([str(executable), "-I", "-S", "-c", ENV_PROBE, json.dumps(modules), str(env.resolve())], capture_output=True, text=True, timeout=30)
            result["probe_returncode"] = proc.returncode
            if proc.returncode == 0:
                result["probe"] = json.loads(proc.stdout)
            else:
                result["probe_error"] = proc.stderr[-2000:]
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            result["probe_error"] = str(exc)
    else:
        result["probe_error"] = "project .venv/bin/python missing"
    result["dependency_sources"] = [file_record(p) for p in [repo / "pyproject.toml", repo / "requirements.txt", repo / "eval_libero/requirements.txt", repo / "uv.lock"] if p.is_file()]
    return result


def metadata_inventory(directory):
    """Only explicit local metadata files; never loads tensors or remote code."""
    result = []
    if directory is None:
        return result
    if not directory.is_dir():
        raise ValueError(f"metadata directory missing: {directory}")
    for path in sorted(directory.rglob("*.json")):
        if path.name not in {"config.json", "model.safetensors.index.json", "preprocessor_config.json", "processor_config.json", "hub_model_info.json"}:
            continue
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError(f"metadata file exceeds 16 MiB: {path}")
        record = file_record(path)
        data = json.loads(path.read_text())
        if path.name == "model.safetensors.index.json":
            # Index proves available layer keys, not that all layers execute.
            record["weight_map_key_count"] = len(data.get("weight_map", {}))
            record["weight_index_only_not_weights_verified"] = True
            import re
            patterns = {}
            for key in data.get("weight_map", {}):
                for match in re.finditer(r"(.*?layers)\.(\d+)\.", key):
                    patterns.setdefault(match[1], set()).add(int(match[2]))
            record["indexed_layer_ids_zero_based"] = {k: sorted(v) for k, v in patterns.items()}
        else:
            record["data"] = data
        result.append(record)
    for path in sorted(directory.glob("*.py")):
        result.extend(source_audit(directory, [path.name]))
    return result


def validate_mask(model, depth, mask, protected_readout=()):
    if type(depth) is not int or depth <= 0:
        raise ValueError("effective depth must be verified positive integer")
    if not isinstance(mask, list) or any(type(v) is not int for v in mask):
        raise ValueError("mask must contain 1-based integer layer IDs")
    if len(mask) != len(set(mask)) or any(v < 1 or v > depth for v in mask):
        raise ValueError("duplicate/out-of-range layer ID")
    protected = set(protected_readout) | ({1, 2, 3} if model == "xiaomi-robotic0" else {1, 2})
    if protected.intersection(mask):
        raise ValueError("mask changes protected fusion/readout/front layers")
    if model == "xiaomi-robotic0" and depth != 36:
        raise ValueError("Xiaomi KV dependency must be re-audited for depth other than 36")
    # Match the implemented conservative adapters, including early-layer shells.
    # This is a declared implementation contract, never measured operator cost.
    shells = sorted(mask) if model in ("xiaomi-robotic0", "db-cogact") else []
    return {"skipped_hidden_update_layers": sorted(mask), "kv_shell_layers": shells,
            "fully_eliminated_layers": sorted(set(mask) - set(shells)), "skip_fraction": len(mask) / depth,
            "classification_basis": "current_all_shell_adapter_contract" if model in ("xiaomi-robotic0", "db-cogact") else "generic_bypass_contract_requires_architecture_verification",
            "operator_execution_verified": False,
            "note": "declared adapter semantics only; real-checkpoint execution and cost not verified"}


def prediction_budgets(depth):
    return {"calibration_episodes": 8, "max_calibration_observations": 80,
            "max_probe_logical_predictions": 80 * (depth + 1) if type(depth) is int else None,
            "max_extra_parity_predictions": 256, "benchmark_observations_minimum": 32,
            "benchmark_warmup_per_condition_observation": 10, "benchmark_measurements_per_condition_observation": 50,
            "invalid_retries_per_key": 1, "invalid_retries_per_model": 8}


def validate_candidates(model, depth, candidates, protected_readout=()):
    """Reject incomplete/duplicate/budget-inconsistent preregistrations."""
    expected = dict(MODELS[model]["budget"])
    if len(candidates) != sum(expected.values()):
        raise ValueError("candidate count differs from preregistered budget")
    expected_depth = MODELS[model]["depth"]
    if expected_depth is not None and depth != expected_depth:
        raise ValueError("effective depth differs from budget; re-audit required")
    masks, ids, counts, records = set(), set(), {}, []
    for candidate in candidates:
        for field in ("id", "mask", "algorithm", "seed", "reason", "probe_source"):
            if field not in candidate or candidate[field] is None:
                raise ValueError(f"candidate lacks {field}")
        record = validate_mask(model, depth, candidate["mask"], protected_readout)
        key = tuple(record["skipped_hidden_update_layers"])
        if key in masks or candidate["id"] in ids:
            raise ValueError("duplicate mask or condition ID")
        source = candidate["probe_source"]
        path = Path(source.get("path", ""))
        if not path.is_file() or source.get("sha256") != digest(path):
            raise ValueError("real probe source missing/hash mismatch")
        masks.add(key)
        ids.add(candidate["id"])
        counts[len(key)] = counts.get(len(key), 0) + 1
        records.append(record)
    if counts != expected:
        raise ValueError("candidate skip counts differ from low/mid/high allocation")
    return {"structurally_valid": True, "runtime_parity_verified": False, "masks": records}


def validate_protocol(data):
    """Validate explicit freeze evidence without inferring unknown protocol values."""
    issues = []
    for field in ("effective_depth", "H", "D"):
        if type(data.get(field)) is not int or data[field] <= 0:
            issues.append(f"{field}: positive verified integer required")
    execution = data.get("R_by_suite", {})
    for suite in SUITES:
        value = execution.get(suite) if isinstance(execution, dict) else None
        if type(value) is not int or value <= 0:
            issues.append(f"R_by_suite.{suite}: unknown; execution blocked")
        elif type(data.get("H")) is int and value > data["H"]:
            issues.append(f"R_by_suite.{suite}: exceeds prediction H")
    for field in ("vision_injection", "action_readout_and_kv", "image_processing", "state_action_dimensions", "normalization", "gripper_and_coordinates", "dtype_and_attention", "cfg_ensemble_rtc", "randomness_and_episode_reset", "environment_and_init_hashes", "checkpoint_and_remote_code", "actual_client_server_overrides"):
        if not data.get(field):
            issues.append(f"{field}: evidence required")
    evidence = data.get("source_files", [])
    if not isinstance(evidence, list) or not evidence:
        issues.append("source_files: local path and sha256 records required")
    else:
        for item in evidence:
            path = Path(item.get("path", "")) if isinstance(item, dict) else Path("")
            if not path.is_file() or item.get("sha256") != digest(path):
                issues.append(f"source_files: missing/hash mismatch: {path}")
    return {"schema_complete": not issues, "issues": issues,
            "execution_authorized": False, "note": "schema/evidence validation only; architecture and parity still require actual verification"}


def task_plan(model):
    """Slots are placeholders, never pretend probe-derived layer IDs exist."""
    conditions = [{"id": "full", "mask": [], "status": "blocked_adapter_and_full_parity"}]
    for k, count in MODELS[model]["budget"]:
        for index in range(count):
            conditions.append({"id": f"skip{k}_candidate{index + 1}", "mask": None,
                               "skip_count": k, "status": "blocked_requires_real_calibration_probe_and_preregistration"})
    # Init is outermost: every condition gets four-suite init0 before init1.
    tasks = [{"condition": c["id"], "suite": suite, "task": task, "init": init,
              "kind": "formal", "dispatchable": False, "env_seed": 7,
              "max_action_steps": LIMITS[suite], "wait_actions": 10, "control_hz": 20}
             for init in range(5) for c in conditions for suite in SUITES for task in range(10)]
    calibration = [{"suite": suite, "task": task, "init": 5, "condition": "full", "kind": "calibration", "dispatchable": False}
                   for suite in SUITES for task in (0, 5)]
    references = [{"suite": suite, "task": 0, "init": 0, "condition": "official_original_full", "kind": "diagnostic_reference", "dispatchable": False} for suite in SUITES]
    return {"model": model, "status": "preparation_only_not_dispatchable", "conditions": conditions,
            "formal_per_condition": 200, "init0_per_condition": 40, "formal_total": len(tasks),
            "calibration_total": 8, "reference_total": 4, "planned_environment_total": len(tasks) + 12,
            "tasks": tasks, "calibration": calibration, "references": references,
            "smoke": "four adapted Full task0/init0, reusable only under identical final protocol; reference episodes separate",
            "budgets": prediction_budgets(MODELS[model]["depth"])}


def inspect_model(root, model, metadata_dir=None):
    repo = root / "policies" / model
    spec = MODELS[model]
    revision = git(repo, "rev-parse", "HEAD")
    origin = git(repo, "remote", "get-url", "origin")
    status = git(repo, "status", "--porcelain=v1", "--untracked-files=normal")
    env = environment(repo, spec["modules"] + spec.get("optional_modules", []) + spec.get("client_modules", []))
    env["required_server_modules"] = spec["modules"]
    env["optional_server_modules"] = spec.get("optional_modules", [])
    env["separate_client_modules"] = spec.get("client_modules", [])
    if metadata_dir is None:
        default_metadata = root / "preparations/idea1_v2_1/metadata" / model
        metadata_dir = default_metadata if default_metadata.is_dir() else None
    blockers = ["formal experiments prohibited during preparation", "native_protocol.json not frozen/validated", "checkpoint normalization and runtime loading not verified", "real-checkpoint official/empty-mask/slow/fast parity not verified", "real-checkpoint operator trace and actual bypass not verified", "closed-loop adapter/telemetry/video/resume/seed pairing not implemented", "probe-derived exact mask IDs not available", "effective executed depth not verified against runtime"]
    receipt = None
    receipt_path = metadata_dir / "weights_verified.json" if metadata_dir else None
    if receipt_path is not None and receipt_path.is_file():
        receipt = {"source": file_record(receipt_path), "recorded_result": json.loads(receipt_path.read_text()),
                   "weight_bytes_rehashed_by_preflight": False,
                   "note": "Prior verification receipt only; current checkpoint bytes were not reverified by this preflight."}
    else:
        blockers.append("weight verification receipt unavailable; run explicit verify_weights.py")
    if not revision:
        blockers.append("repository missing or revision unreadable")
    if not origin or origin.removesuffix(".git").lower() != spec["origin"].lower():
        blockers.append("unexpected repository origin")
    if status:
        blockers.append("upstream repository has local changes; review recorded diff")
    if not env.get("prompt_matches"):
        blockers.append("project virtualenv prompt missing/mismatch")
    if not all(env.get("probe", {}).get("module_discoverable", {}).get(m) for m in spec["modules"]):
        blockers.append("required module not statically discoverable")
    if model == "cogact":
        blockers.append("official reproducible LIBERO checkpoint not verified; do not substitute DB-CogACT/OXE")
    if model == "isaac-gr00t":
        blockers.append("per-suite checkpoints/select_layer require verification; 16-layer version is supplementary only")
    client = None
    if model == "db-cogact":
        client_repo = root / "policies/dexbotic-benchmark"
        paths = ["evaluation/run_libero_evaluation.py", "evaluation/policies/libero_vla_agent.py", "evaluation/evaluator/libero_evaluator.py"]
        paths += [f"evaluation/configs/libero/{suite}.yaml" for suite in SUITES]
        paths += ["evaluation/configs/libero/example_libero.yaml"]
        client = {"revision": git(client_repo, "rev-parse", "HEAD"), "origin": git(client_repo, "remote", "get-url", "origin"),
                  "dirty_porcelain": git(client_repo, "status", "--porcelain=v1"), "sources": source_audit(client_repo, paths),
                  "warning": "published suite YAML R values are evidence only; confirm these configurations match selected DB-CogACT checkpoint before freezing"}
    return {"model": model, "status": "blocked_preparation_only", "repository": {"path": str(repo), "revision": revision, "origin": origin,
            "expected_origin": spec["origin"], "dirty_porcelain": status, "diff_sha256": hashlib.sha256((git(repo, "diff", "HEAD", "--binary") or "").encode()).hexdigest()},
            "environment": env, "source_audit": source_audit(repo, spec["sources"]),
            "local_metadata": metadata_inventory(metadata_dir), "weight_verification_receipt": receipt, "separate_client": client, "blockers": blockers,
            "native_protocol": {"H": None, "R": None, "D": None, "effective_depth": None,
                "status": "unknown_until_checkpoint_and_actual_client_server_overrides_verified", "unknown_R_blocks_execution": True}}


def write_new(path, data):
    # Explicit output only and no overwrite, including symlinks.
    with path.open("x", encoding="utf-8") as output:
        json.dump(data, output, indent=2, ensure_ascii=False)
        output.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--model", choices=tuple(MODELS))
    parser.add_argument("--metadata-dir", type=Path, help="explicit local checkpoint metadata directory; requires --model")
    parser.add_argument("--write-manifest", type=Path, help="explicit NEW output file; never overwritten")
    parser.add_argument("--write-plan", type=Path, help="explicit NEW blocked task-plan file; requires --model")
    parser.add_argument("--protocol-file", type=Path, help="optional explicit local native_protocol JSON to validate; never treated as execution authorization")
    args = parser.parse_args(argv)
    if (args.metadata_dir or args.write_plan) and not args.model:
        parser.error("--metadata-dir/--write-plan require --model")
    try:
        data = {"schema_version": 1, "mode": "preparation_only", "can_run_experiments": False,
                "models": [inspect_model(args.root.resolve(), model, args.metadata_dir) for model in ([args.model] if args.model else MODELS)]}
        data["preparation_sources"] = [file_record(p) for p in sorted(Path(__file__).parent.glob("*.py"))]
        if args.protocol_file:
            data["protocol_validation"] = {"source": file_record(args.protocol_file), "validation": validate_protocol(json.loads(args.protocol_file.read_text()))}
        if args.write_manifest:
            write_new(args.write_manifest, data)
        if args.write_plan:
            write_new(args.write_plan, task_plan(args.model))
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        parser.exit(2, f"preflight error: {exc}\n")
    # A blocked inventory must never look like a runnable validation success.
    return 2


if __name__ == "__main__":
    sys.exit(main())
