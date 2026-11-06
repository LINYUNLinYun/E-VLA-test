"""Sequential GPU experiment queue; reuse immutable weights, never KV/Graphs."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--suite", choices=("core", "followup"), default="core")
    args = parser.parse_args()
    if args.wait_pid:
        import psutil
        try:
            previous = psutil.Process(args.wait_pid)
            created = previous.create_time()
            while previous.is_running() and previous.create_time() == created and previous.status() != psutil.STATUS_ZOMBIE:
                time.sleep(5)
        except psutil.NoSuchProcess:
            pass
    from . import cewp_bench as bench
    original_loader = bench.load_backend
    cached = None
    def loader(checkpoint, **kwargs):
        nonlocal cached
        if cached is None:
            cached = original_loader(checkpoint, **kwargs)
        else:
            cached.num_steps = kwargs["num_steps"]
        for name in ("graph_engine_class", "graph_core_class", "selective_overlap"):
            if hasattr(cached, name):
                delattr(cached, name)
        return cached
    bench.load_backend = loader
    root = Path(args.root)
    status_file = root/("matrix_status.json" if args.suite == "core" else "followup_matrix_status.json")
    common = ["--checkpoint", "shared/checkpoints/openpi/pi05_libero_pytorch_idea2",
              "--warmup", "10", "--count", "100", "--continuous-count", "100"]
    main_pairs = ["eager_serial:eager_dual_ae_first", "eager_serial:eager_dual_prefill_first",
                  "graph_serial:graph_ae_first", "graph_serial:graph_prefill_first",
                  "graph_serial_high:graph_ae_high", "graph_serial_high:graph_pre_high"]
    continuous = ["eager_serial", "eager_dual_ae_first", "eager_dual_prefill_first",
                  "graph_serial", "graph_ae_first", "graph_prefill_first",
                  "graph_ae_high", "graph_pre_high", "graph_fresh_serial"]
    jobs = []
    for steps in (5, 1):
        jobs.append((f"d{steps}_main", steps, ["--pairs", *main_pairs, "--continuous", *continuous]))
    for steps in (10, 5, 1):
        jobs.append((f"d{steps}_isolated", steps,
            ["--kind", "isolated", "--pairs", "--continuous", "eager_serial", "graph_serial"]))
    for steps in (10, 5, 1):
        for selective, mode in (("all", "graph_ae_high"), ("vision_only", "graph_ae_high"),
                                ("vlm_only", "graph_pre_high")):
            jobs.append((f"d{steps}_selective_{selective}", steps,
                ["--selective", selective, "--pairs", f"graph_serial_high:{mode}",
                 "--continuous", "graph_serial_high", mode]))
        jobs.append((f"d{steps}_hoist", steps,
            ["--hoist-schedule", "--pairs", "graph_serial_high:graph_ae_high",
             "--continuous", "graph_serial_high", "graph_ae_high"]))
    if args.suite == "followup":
        jobs = []
        for steps in (10, 5, 1):
            for name, switches in (
                ("wait_full", ["--await-previous-prefix"]),
                ("vision_wait", ["--selective", "vision_only", "--await-previous-prefix"]),
                ("modulation", ["--hoist-modulation"]),
                ("vision_modulation_wait", ["--selective", "vision_only", "--await-previous-prefix", "--hoist-modulation"]),
            ):
                jobs.append((f"d{steps}_{name}", steps, [*switches, "--pairs", "graph_serial_high:graph_ae_high",
                             "--continuous", "graph_serial_high", "graph_ae_high"]))
    status = []
    for name, steps, extra in jobs:
        sys.argv = ["cewp_bench", "--output", str(root/name), "--steps", str(steps), *common, *extra]
        print("MATRIX START", name, flush=True)
        started = time.time()
        item = dict(name=name, argv=sys.argv, started_unix=started, status="running")
        status.append(item)
        status_file.write_text(json.dumps(status, indent=2))
        try:
            bench.main()
            item["status"] = "passed"
        except Exception:
            item["status"] = "failed"
            item["traceback"] = traceback.format_exc()
            print(item["traceback"], flush=True)
            # Do not continue using a possibly corrupted CUDA execution context.
            status_file.write_text(json.dumps(status, indent=2))
            raise
        finally:
            item["finished_unix"] = time.time()
            status_file.write_text(json.dumps(status, indent=2))
            gc.collect()
        print("MATRIX END", name, flush=True)


if __name__ == "__main__":
    main()
