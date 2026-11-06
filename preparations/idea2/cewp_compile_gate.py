"""Small D1/D5 compiler correctness gates; reject before long timing runs."""
import argparse
import contextlib
import gc
import json
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
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
        cached.await_previous_prefix = False
        return cached
    bench.load_backend = loader
    status = []
    for steps in (1, 5):
        for execution, pair in (("only", "eager_serial:eager_dual_ae_first"),
                                ("graph", "graph_serial:graph_ae_first")):
            name = f"compile_d{steps}_{execution}_gate"
            argv = ["cewp_bench", "--output", str(root/name), "--steps", str(steps),
                    "--checkpoint", "shared/checkpoints/openpi/pi05_libero_pytorch_idea2",
                    "--warmup", "3", "--count", "3", "--compile", "--pairs", pair, "--continuous"]
            item = dict(name=name, argv=argv, status="running", tolerance=dict(max_abs=1e-5, mae=1e-6))
            status.append(item)
            (root/"compile_gate_status.json").write_text(json.dumps(status, indent=2))
            print("COMPILE GATE", name, flush=True)
            with (root/(name+".log")).open("x") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                sys.argv = argv
                try:
                    bench.main()
                    item["status"] = "passed_numerical_gate"
                except AssertionError as exc:
                    item["traceback"] = traceback.format_exc()
                    print(item["traceback"], flush=True)
                    expected = "Predeclared numerical tolerance failed; evidence retained in result.json"
                    if str(exc) != expected:
                        item["status"] = "unexpected_failure"
                        raise
                    item["status"] = "rejected_numerical_tolerance"
                except BaseException:
                    item["status"] = "unexpected_failure"
                    item["traceback"] = traceback.format_exc()
                    raise
                finally:
                    (root/"compile_gate_status.json").write_text(json.dumps(status, indent=2))
            if cached is not None and hasattr(cached, "compiler_originals"):
                raise RuntimeError("Compiler forwards were not restored; stopping queue")
            gc.collect()
            print(name, item["status"], flush=True)
    bench.load_backend = original_loader


if __name__ == "__main__":
    main()
