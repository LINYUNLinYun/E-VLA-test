"""One resource dispatcher: use idle GPU0, otherwise GPU3 after matched quality."""
import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

R = Path(__file__).resolve().parent
GPUS = {0:"GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1", 3:"GPU-555cde36-5e41-bc44-0c69-ea47358fd703"}
state = dict(status="waiting_for_resource", observations=[])


def save():
    (R/"sm32_dispatch.json").write_text(json.dumps(state, indent=2)+"\n")


def idle(index):
    root = ET.fromstring(subprocess.check_output(["nvidia-smi", "-i", GPUS[index], "-q", "-x"], text=True)).find("gpu")
    processes = [dict(pid=p.findtext("pid"), type=p.findtext("type")) for p in root.findall("processes/process_info")]
    mem = float(root.findtext("fb_memory_usage/used").split()[0])
    util = float(root.findtext("utilization/gpu_util").split()[0])
    state["observations"].append(dict(gpu=index, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), processes=processes, memory_mib=mem, util=util))
    return not processes and mem < 256 and util <= 1


for _ in range(120):
    ready = R/"MATCHED_READY.json"
    quality = json.loads(ready.read_text()) if ready.exists() else {}
    if idle(0):
        chosen = 0; break
    if quality.get("status") == "passed" and quality.get("exit_code") == 0 and quality.get("cleanup_passed") and idle(3):
        chosen = 3; break
    save()
    print("SM32 waiting: GPU0 busy; GPU3 reserved for matched quality", flush=True)
    time.sleep(20)
else:
    raise RuntimeError("Finite resource wait exhausted; no matrix launched")
state.update(status="running", selected_gpu=chosen, selected_uuid=GPUS[chosen]); save()
lease = R.parents[2]/"results/idea1_v3_3/external_gpu_leases.json"
with lease.open("r+") as stream:
    fcntl.flock(stream, fcntl.LOCK_EX)
    data=json.load(stream);data[str(chosen)].update(active=True, owner="user-authorized idea2 fresh-tail", purpose="SM32 formal/repeat and two profiles after matched quality boundary")
    stream.seek(0);json.dump(data,stream,indent=2);stream.truncate()
cpus = range(32,48) if chosen == 0 else range(96,112)
for matrix in ["formal_sm32_matrix.json", "profile_matrix.json"]:
    command=[sys.executable,str(R/"run_matrix.py"),str(R/matrix),GPUS[chosen],",".join(map(str,cpus))]
    code=subprocess.call(command)
    if code:
        state.update(status="failed", failed_matrix=matrix, exit_code=code);save();raise SystemExit(code)
state["status"]="passed";save()
