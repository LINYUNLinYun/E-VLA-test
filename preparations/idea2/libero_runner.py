"""Run in libero-eval-v21/.venv; GPU model is served by idea2.server."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import uuid

from preparations.idea1_v2_1 import runtime_client as client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=9012)
    parser.add_argument("--smoke", action="store_true", help="Four suites task0/init0, all seven conditions")
    parser.add_argument("--suite", choices=client.SUITES)
    parser.add_argument("--mode", choices=("fresh_serial", "serial_reuse", "async_reuse"))
    parser.add_argument("--interval", type=int, choices=(1, 2, 4))
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    conditions = [("fresh_serial", 1)] + [(mode, interval) for mode in ("serial_reuse", "async_reuse") for interval in (1, 2, 4)]
    if args.mode:
        conditions = [x for x in conditions if x[0] == args.mode]
    if args.interval:
        conditions = [x for x in conditions if x[1] == args.interval]
    scenes = [(suite, task, init) for suite in (client.SUITES if not args.suite else [args.suite])
              for task in ([0] if args.smoke else [0, 5]) for init in ([0] if args.smoke else [0, 1])]
    for mode, interval in conditions:
        client.rpc(args.port, dict(op="configure", mode=mode, interval=interval))
        for suite, task, init in scenes:
            directory = root / f"{mode}_r{interval}" / suite / f"task{task:02d}" / f"init{init:02d}"
            if ((directory / "complete.json").exists() and (directory / "video_verified.json").exists()
                    and (directory / "result.json").exists()):
                continue
            if directory.exists():
                previous = root / "attempts" / f"{mode}_r{interval}_{suite}_task{task:02d}_init{init:02d}_{uuid.uuid4().hex}"
                previous.parent.mkdir(parents=True, exist_ok=True)
                directory.rename(previous)
            begin = time.perf_counter()
            try:
                result = client.run_episode(directory, args.port, "pi0", suite, task, init, [], kind="idea2")
                # Drain in-flight background work and include it in episode time.
                client.rpc(args.port, dict(op="drain"))
                result.update(mode=mode, interval=interval, all_work_wall_s=time.perf_counter()-begin)
                client.write_json(directory / "result.json", result, exclusive=False)
                infers = [json.loads(line) for line in (directory / "infer.jsonl").read_text().splitlines()]
                telemetry = [json.loads(line) for line in (directory / "telemetry.jsonl").read_text().splitlines()]
                for row in infers:
                    origin = row.get("source_host_receipt_unix_s")
                    executed = telemetry[row["step"]+1:row["step"]+row["executed_length"]+1]
                    ages = [(state["wall_timestamp_s"]-origin)*1000 for state in executed] if origin is not None else []
                    row["execution_source_receipt_age_ms_start"] = ages[0] if ages else None
                    row["execution_source_receipt_age_ms_end"] = ages[-1] if ages else None
                    row["execution_camera_capture_age_ms"] = None
                (directory / "infer.jsonl").write_text("".join(json.dumps(row)+"\n" for row in infers))
                # Decode every recorded frame; telemetry already contains true EEF.
                subprocess.run(["ffmpeg", "-v", "error", "-i", str(directory / "video.mp4"), "-f", "null", "-"],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                client.write_json(directory / "video_verified.json", dict(decoded=True, frames=result["frames"]))
                client.write_json(directory / "complete.json", dict(valid=True, mode=mode, interval=interval,
                                  all_background_work_drained=True, video_decoded=True))
            except Exception as exc:
                directory.mkdir(parents=True, exist_ok=True)
                client.write_json(directory / "technical_error.json", dict(error=repr(exc)), exclusive=False)
                raise
            print(json.dumps(dict(mode=mode, interval=interval, suite=suite, task=task, init=init,
                                  success=result["success"], directory=str(directory))), flush=True)


if __name__ == "__main__":
    main()
