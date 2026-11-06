"""Read the known E4 lease every30s; never infer release from a quiet GPU."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

R=Path(__file__).resolve().parent
LEASE=R.parents[2]/"results/idea1_e4_p4_20261025/gpu_leases.json"
UUID="GPU-555cde36-5e41-bc44-0c69-ea47358fd703"


def idle():
    root=ET.fromstring(subprocess.check_output(["nvidia-smi","-i",UUID,"-q","-x"],text=True)).find("gpu")
    processes=[dict(pid=x.findtext("pid"),type=x.findtext("type")) for x in root.findall("processes/process_info")]
    return dict(processes=processes,memory_mib=float(root.findtext("fb_memory_usage/used").split()[0]))


seen_reserved=False
for i in range(1440):
    value=json.loads(LEASE.read_text()) if LEASE.exists() else {}
    worker=value.get("workers",{}).get("3",{})
    if value.get("active") or worker.get("active"):seen_reserved=True
    if seen_reserved and value.get("active") is False and worker.get("active") is False:
        one=idle();time.sleep(1);two=idle()
        if all(not x["processes"] and x["memory_mib"]<256 for x in [one,two]):
            proof=dict(status="released_and_idle",time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=str(LEASE),source_sha256=hashlib.sha256(LEASE.read_bytes()).hexdigest(),lease=value,idle_observations=[one,two])
            (R/"gpu3_handoff.json").write_text(json.dumps(proof,indent=2)+"\n")
            (R/"ENABLE_GPU3").write_text("Known E4 owner explicitly released; idle checked twice.\n")
            print("GPU3 handoff verified; dispatcher enabled",flush=True);break
    if i%4==0:print("Awaiting known E4 lease release; worker PID",worker.get("pid"),"active",worker.get("active"),flush=True)
    time.sleep(30)
else:raise TimeoutError("Finite resource wait exhausted without a handoff; no GPU3 work launched")
