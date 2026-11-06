"""Prepare batch-study commands without querying GPUs or launching experiments.

Reuse the existing fixed-batch planner and runtime. Keep this command-only
adapter outside the runtime source snapshot used by experiments already running.
"""
import hashlib
import json
from pathlib import Path
import shlex

from preparations.idea2 import batch_plan


def parser():
    result = batch_plan.parser()
    result.description = __doc__
    result.add_argument("--prepare-stream", choices=("caller", "prefix", "dedicated"),
                        default="caller", help="Placement for online prepare only")
    result.add_argument("--disable-lookahead", action="store_true",
                        help="Online-only measurement: current input prepare through CPU action; no next-input preparation or next Encoder")
    return result


def build_plan(args):
    inputs = set(args.input_modes)
    if args.prepare_stream != "caller" and "online" not in inputs:
        raise ValueError("Non-caller prepare stream requires online jobs")
    if args.disable_lookahead and inputs != {"online"}:
        raise ValueError("Fresh-input measurements require --input-modes online")
    plan = batch_plan.build_plan(args)
    plan["preparation_tool_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    plan["online_prepare_stream"] = args.prepare_stream
    plan["online_lookahead_enabled"] = not args.disable_lookahead
    plan["measurement_contract"] = {
        "prepared": "Input preparation precedes timing; capacity measurement only",
        "online_lookahead": "Admission-to-response service segments and group throughput; also report each input's own prepare-to-CPU lifetime",
        "online_fresh": "Each current input is prepared inside its own request; no future CPU preparation or next Encoder",
        "scope": "Recorded replay; no real camera arrival, batching wait, robot execution, or closed-loop quality claim",
    }
    for job in plan["jobs"]:
        online = job["input_mode"] == "online"
        job["prepare_stream"] = args.prepare_stream if online else "caller"
        job["lookahead_enabled"] = not (online and args.disable_lookahead)
        job["measurement_boundary"] = (
            "validation_only" if job["stage"] == "smoke" else
            "preprepared_capacity" if not online else
            "fresh_prepare_to_cpu" if args.disable_lookahead else
            "lookahead_service_segment"
        )
        # Smoke remains the existing prepared-input numerical gate. Performance
        # and profile rows alone define the requested measurement boundary.
        if online and args.prepare_stream != "caller":
            job["argv"] += ["--prepare-stream", args.prepare_stream]
        if online and args.disable_lookahead:
            job["argv"].append("--disable-lookahead")
        job["command"] = shlex.join([
            "env", *(f"{key}={value}" for key, value in job["environment"].items()),
            *job["argv"],
        ])
    return plan


def render_commands(plan):
    note = [
        "# Batch follow-up preparation", "",
        "This file only contains commands. No GPU was queried or reserved; no experiment was launched.", "",
        "Reuse compatible completed measurements. Run only missing cells, then selected representative profiles.", "",
        "Prepared input measures capacity. Online lookahead measures service segments; it does not establish fresh-observation latency.", "",
        "With --disable-lookahead, each current prepare is timed inside its request and no next Encoder is submitted. Smoke remains a prepared-input numerical gate.", "",
        "Always report batch response latency without dividing by batch size. Keep throughput, own-input lifetime, and numerical eligibility separate.", "",
    ]
    return "\n".join(note) + batch_plan.render_commands(plan)


def main(argv=None):
    args = parser().parse_args(argv)
    plan = build_plan(args)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (destination / "commands.md").write_text(render_commands(plan))
    print(json.dumps(dict(status="prepared_not_run", jobs=len(plan["jobs"]),
                          gpu_queried=False, launched=0,
                          output=str(destination.resolve()))))


if __name__ == "__main__":
    main()
