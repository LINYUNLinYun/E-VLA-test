"""Generate a bounded batch experiment plan. Never launches jobs or inspects GPUs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex


ROOT = Path(__file__).resolve().parents[2]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, help="New directory for plan.json and commands.md")
    p.add_argument("--gpu-uuid", help="Optional future GPU assignment; no availability check")
    p.add_argument("--python", default=str(ROOT / "policies/openpi/.venv/bin/python"))
    p.add_argument("--checkpoint", default="shared/checkpoints/openpi/pi05_libero_pytorch_idea2")
    p.add_argument("--bank", default="results/idea1_v3_1/input_bank.pkl")
    p.add_argument("--batch-sizes", type=int, nargs="+", choices=(1, 2, 4), default=(1, 2, 4))
    p.add_argument("--input-modes", nargs="+", choices=("prepared", "online"), default=("prepared", "online"))
    p.add_argument("--green-sm-mins", type=int, nargs="+", default=(16,))
    p.add_argument("--pairs", type=int, default=100)
    p.add_argument("--length", type=int, default=8)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--verify-count", type=int, default=4)
    p.add_argument("--lead", type=int, choices=(0, 1, 3, 6, 18), default=3)
    p.add_argument("--diagnostic-batch-drift", action="store_true",
                   help="For B>1 only, prepare same-B scheduling diagnostics; retain failed B1 equivalence and never promote diagnostic-only results")
    p.add_argument("--no-profiles", action="store_true")
    return p


def validate(args):
    if args.pairs < 100:
        raise ValueError("Performance plan requires at least 100 pairs; smoke jobs are generated separately")
    if args.length < 2 or args.warmup < 2 or args.verify_count < 4:
        raise ValueError("Require length/warmup >= 2 and verify-count >= 4 to exercise both slots")
    if any(x <= 0 for x in args.green_sm_mins):
        raise ValueError("Green Context SM minima must be positive")
    if args.gpu_uuid is not None and not args.gpu_uuid.startswith("GPU-"):
        raise ValueError("Use a physical GPU UUID, not a GPU index")
    if args.diagnostic_batch_drift and not any(batch > 1 for batch in args.batch_sizes):
        raise ValueError("Batch-drift diagnostic requires at least one B>1 condition")


def source_hashes():
    directory = ROOT / "preparations/idea2"
    names = ["progressive_kv.py", "progressive_kv_pool.py", "progressive_kv_kernels.py",
             "runtime_graph.py", "graph_sampler.py", "cewp_preparation.py", "cewp_segmented.py",
             "openpi_backend.py", "green_context.py", "external_events.py", "layer_scheduler.py",
             "scheduler.py", "gpu_guard.py", "layer_bench.py", "experiments.py"]
    paths = [directory / name for name in names] + sorted(directory.glob("batch_*.py"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.is_file()}


def build_plan(args):
    validate(args)
    base = Path(args.output).resolve()
    gpu = args.gpu_uuid or "GPU-SET-ME-BEFORE-RUNNING"
    environment = dict(CUDA_VISIBLE_DEVICES=gpu, JAX_PLATFORMS="cpu", JAX_PLATFORM_NAME="cpu",
                       XLA_PYTHON_CLIENT_PREALLOCATE="false", CUDA_DEVICE_MAX_CONNECTIONS="32",
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                       OPENPI_DATA_HOME=str(ROOT / "shared/checkpoints/openpi"),
                       PYTHONPATH=str(ROOT) + ":" + str(ROOT / "policies/openpi/src"))
    jobs = []

    def job(stage, batch, mode, input_mode, *, sm=16):
        suffix = f"_sm{sm}" if mode == "green" else ""
        key = f"{stage}_b{batch}_{mode}{suffix}_{input_mode}"
        output = base / "runs" / key
        argv = [args.python, "-u", "-m", "preparations.idea2.batch_bench",
                "--output", str(output), "--gpu-uuid", gpu,
                "--checkpoint", str((ROOT / args.checkpoint).resolve()),
                "--bank", str((ROOT / args.bank).resolve()),
                "--batch-size", str(batch), "--mode", mode, "--input-mode", input_mode,
                "--lead", str(args.lead), "--ae-sm-min", str(sm),
                "--pairs", str(2 if stage == "smoke" else 1 if stage == "profile" else args.pairs),
                "--length", str(4 if stage == "smoke" else args.length),
                "--warmup", str(2 if stage == "smoke" else args.warmup),
                "--verify-count", str(args.verify_count)]
        diagnostic = args.diagnostic_batch_drift and batch > 1
        if diagnostic:
            argv.append("--diagnostic-batch-drift")
        if stage == "profile":
            argv.append("--profile")
            argv = ["nsys", "profile", "--trace=cuda,nvtx",
                    "--cuda-graph-trace=node:host-only:nvtx-live", "--sample=none", "--cpuctxsw=none",
                    "--nvtx-domain-exclude=TSL,CCCL", "--capture-range=cudaProfilerApi",
                    "--capture-range-end=stop", "--kill=none", "--force-overwrite=false",
                    "--output=" + str(base / key), *argv]
        command = shlex.join(["env", *(f"{k}={v}" for k, v in environment.items()), *argv])
        jobs.append(dict(id=key, stage=stage, batch_size=batch, mode=mode, input_mode=input_mode,
                         ae_sm_min=sm if mode == "green" else None, lead=args.lead, output=str(output),
                         diagnostic_batch_drift=diagnostic,
                         numerical_policy="same_batch_diagnostic" if diagnostic else "strict_b1",
                         cwd=str(ROOT), environment=environment.copy(), argv=argv, command=command,
                         status="not_started"))

    batches = list(dict.fromkeys(args.batch_sizes))
    input_modes = list(dict.fromkeys(args.input_modes))
    minima = list(dict.fromkeys(args.green_sm_mins))
    for batch in batches:
        job("smoke", batch, "paired", "prepared")
        for minimum in minima:
            job("smoke", batch, "green", "prepared", sm=minimum)
    for batch in batches:
        for input_mode in input_modes:
            job("performance", batch, "paired", input_mode)
            for minimum in minima:
                job("performance", batch, "green", input_mode, sm=minimum)
    if not args.no_profiles:
        for batch in batches:
            for input_mode in input_modes:
                for mode in ("serial", "pipeline"):
                    job("profile", batch, mode, input_mode)
                for minimum in minima:
                    job("profile", batch, "green", input_mode, sm=minimum)
    return dict(schema_version=1, status="prepared_not_run", created_utc=datetime.now(timezone.utc).isoformat(),
                gpu_assigned=bool(args.gpu_uuid), gpu_uuid=args.gpu_uuid, cuda_initialized=False,
                model="pi05_libero", num_steps=10, action_horizon=10, batch_sizes=batches,
                diagnostic_batch_drift_requested=bool(args.diagnostic_batch_drift),
                precision="unchanged checkpoint BF16 / native FP32 Euler", source_sha256=source_hashes(),
                correctness_gate=dict(max_abs=1e-5, mae=1e-6, per_lane=True,
                                      batch_vs_b1="fresh KV only; verify before performance",
                                      same_batch_eager="Required for diagnostic B>1; independent original eager computation",
                                      pipeline="same observed frozen frontier schedule serial oracle"),
                execution_policy="No automatic execution, GPU polling, resource claim or background monitor",
                stage_gates=["Pass CPU checks", "Assign an exclusively idle GPU",
                             "Run smoke and retain all failures",
                             "Strict jobs require B1 equivalence; diagnostic jobs retain B1 failures and require same-B numerical gates",
                             "Capture selected representative traces separately from performance"],
                limits=["Fixed lane membership/order; whole-batch reset only",
                        "Recorded input replay does not establish closed-loop quality or online observation freshness",
                        "Diagnostic-only timings are scheduling evidence, not B1 equivalence or promotion evidence",
                        "Event envelopes do not establish kernel overlap",
                        "Green resource modes use separate processes; do not call their measurements paired AB"],
                jobs=jobs)


def render_commands(plan):
    parts = ["# Fixed-batch pipeline commands", "",
             "Prepared only. Nothing has been launched. Run from the repository root.", "",
             "Replace the GPU UUID placeholder after a GPU is allocated. Each command checks the shared lease and actual occupancy.", "",
             "Pass smoke numerical checks before performance; capture selected profiles separately. Do not run all jobs concurrently.", ""]
    if plan.get("diagnostic_batch_drift_requested"):
        parts += ["B>1 commands explicitly enable same-B diagnostics. B1 commands remain strict.", "",
                  "A failed B1-equivalence gate remains failed. Diagnostic-only results cannot establish B1 equivalence, closed-loop quality, or promotion eligibility.", ""]
    for row in plan["jobs"]:
        parts += [f"## {row['id']}", "", f"Numerical policy: {row['numerical_policy']}", "",
                  "```bash", row["command"], "```", ""]
    return "\n".join(parts)


def main(argv=None):
    args = parser().parse_args(argv)
    plan = build_plan(args)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (destination / "commands.md").write_text(render_commands(plan))
    print(json.dumps(dict(status=plan["status"], jobs=len(plan["jobs"]),
                          gpu_queried=False, launched=0, output=str(destination.resolve()))))


if __name__ == "__main__":
    main()
