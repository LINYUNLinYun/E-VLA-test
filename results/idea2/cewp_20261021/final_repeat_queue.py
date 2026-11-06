"""Finish GPU2 matched-step repeats before releasing the native baseline hold."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import psutil


root = Path(__file__).resolve().parent
state_path = root / "final_repeat_status.json"
watched = psutil.Process(2998637)
created = watched.create_time()
if "preparations.idea2.cewp_matrix" not in watched.cmdline() or "followup" not in watched.cmdline():
    raise RuntimeError("Expected the live followup matrix process")
state = dict(pid=os.getpid(), watched_pid=watched.pid, watched_create_time=created,
             phase="waiting_followup", jobs=[])


def save():
    state["updated_unix"] = time.time()
    temporary = state_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(state_path)


try:
    while True:
        try:
            alive = watched.is_running() and watched.create_time() == created and watched.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            alive = False
        if not alive:
            break
        state["last_verified_live_unix"] = time.time()
        save()
        time.sleep(5)
    for filename, count in (("matrix_status.json", 17), ("followup_matrix_status.json", 12)):
        jobs = json.loads((root/filename).read_text())
        if len(jobs) != count or not all(job["status"] == "passed" for job in jobs):
            raise RuntimeError(f"{filename}: required complete passing queue")
    state["phase"] = "repeating"
    for steps in (10, 5, 1):
        output = root/f"d{steps}_vision_modulation_wait_repeat"
        if output.exists():
            raise RuntimeError(f"Refusing to overwrite {output}")
        command = [sys.executable, "-u", "-m", "preparations.idea2.cewp_bench",
                   "--checkpoint", "shared/checkpoints/openpi/pi05_libero_pytorch_idea2",
                   "--output", str(output), "--steps", str(steps),
                   "--warmup", "10", "--count", "100", "--continuous-count", "100",
                   "--pairs", "graph_serial_high:graph_ae_high",
                   "--continuous", "graph_serial_high", "graph_ae_high",
                   "--selective", "vision_only", "--await-previous-prefix", "--hoist-modulation"]
        item = dict(steps=steps, command=command, status="running", started_unix=time.time())
        state["jobs"].append(item)
        with (root/f"{output.name}.log").open("x") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            item["pid"] = child.pid
            save()
            while child.poll() is None:
                item["last_live_poll_unix"] = time.time()
                save()
                time.sleep(5)
            item["returncode"] = child.returncode
        if child.returncode:
            raise RuntimeError(f"{output.name}: exit {child.returncode}")
        result = json.loads((output/"result.json").read_text())
        if len(result["groups"]) != 3 or not all(g["correct"] is True for g in result["groups"]):
            raise RuntimeError(f"{output.name}: incomplete or failed numerical gate")
        if len(result["rows"]) != 400:
            raise RuntimeError(f"{output.name}: expected 400 measured outputs")
        item.update(status="passed", finished_unix=time.time())
        save()
    state["phase"] = "complete"
    state["finished_unix"] = time.time()
    save()
    (root/"native_queue"/"hold").unlink()
    print("All primary and repeat queues passed; native hold released", flush=True)
except BaseException:
    state["phase"] = "failed"
    state["traceback"] = traceback.format_exc()
    save()
    raise
