"""Wait for the verified matrix process, then exclusively queue native GPU2 runs."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import psutil


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--matrix-pid",type=int,required=True)
    parser.add_argument("--root",required=True)
    parser.add_argument("--gpu-uuid",required=True)
    args=parser.parse_args()
    root=Path(args.root).resolve()
    queue=root/"native_queue"
    status_path=queue/"status.json"
    watched=psutil.Process(args.matrix_pid)
    if not watched.is_running() or watched.status()==psutil.STATUS_ZOMBIE or "preparations.idea2.cewp_matrix" not in watched.cmdline():
        raise RuntimeError("Requested PID is not the running CEWP matrix")
    created=watched.create_time()
    state=dict(supervisor_pid=os.getpid(),supervisor_create_time=psutil.Process().create_time(),
               matrix_pid=args.matrix_pid,matrix_create_time=created,matrix_cmdline=watched.cmdline(),
               gpu_uuid=args.gpu_uuid,phase="waiting_matrix",started_unix=time.time(),jobs=[])
    def save():
        state["updated_unix"]=time.time()
        temporary=status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state,indent=2))
        temporary.replace(status_path)
    def gpu_processes():
        output=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,process_name",
                                        "--format=csv,noheader"],text=True,timeout=10)
        return [dict(pid=int(row[0].strip()),gpu_uuid=row[1].strip(),name=row[2].strip())
                for row in csv.reader(io.StringIO(output)) if len(row)>=3 and row[1].strip()==args.gpu_uuid]
    def pending_gpu2_hosts():
        # A performance process may be loading weights before it owns a CUDA context.
        found=[]
        for process in psutil.process_iter(["pid","cmdline","status"]):
            try:
                if process.pid==os.getpid() or process.info["status"]==psutil.STATUS_ZOMBIE:
                    continue
                command=process.info["cmdline"] or []
                if not command or "python" not in Path(command[0]).name:
                    continue
                target=process.environ().get("CUDA_VISIBLE_DEVICES")
                if target in ("2",args.gpu_uuid):
                    found.append(dict(pid=process.pid,create_time=process.create_time(),cmdline=command))
            except (psutil.NoSuchProcess,psutil.AccessDenied):
                continue
        return found
    try:
        save()
        while True:
            try:
                alive=watched.is_running() and watched.create_time()==created and watched.status()!=psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                alive=False
            if not alive:
                break
            state["matrix_live_verified_unix"]=time.time()
            save()
            time.sleep(5)
        state["matrix_exit_observed_unix"]=time.time()
        matrix=json.loads((root/"matrix_status.json").read_text())
        expected={f"d{d}_main" for d in (5,1)} | {f"d{d}_isolated" for d in (10,5,1)}
        expected|={f"d{d}_selective_{kind}" for d in (10,5,1) for kind in ("all","vision_only","vlm_only")}
        expected|={f"d{d}_hoist" for d in (10,5,1)}
        if len(matrix)!=17 or {job["name"] for job in matrix}!=expected or not all(job["status"]=="passed" for job in matrix):
            raise RuntimeError("Matrix did not finish all 17 expected unique jobs successfully")
        state["matrix_passed_jobs"]=sorted(expected)
        save()
        for steps in (10,5,1):
            output=root/f"native_d{steps}_gpu2"
            if output.exists():
                raise RuntimeError(f"Refusing to overwrite native output {output}")
            command=[sys.executable,"-u","-m","preparations.idea2.cewp_native_baseline",
                     "--output",str(output),"--steps",str(steps),"--count","100","--warmup","10"]
            item=dict(steps=steps,command=command,output=str(output),status="waiting_idle")
            state["jobs"].append(item)
            idle_checks=0
            while True:
                hold=(queue/"hold").exists()
                active=gpu_processes()
                hosts=pending_gpu2_hosts()
                state["phase"]="held" if hold else "waiting_gpu_idle"
                item["last_idle_check"]=dict(time_unix=time.time(),hold=hold,gpu_compute_processes=active,pending_gpu_hosts=hosts)
                idle_checks=idle_checks+1 if not hold and not active and not hosts else 0
                save()
                if idle_checks>=2:
                    break
                time.sleep(5)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES="2",JAX_PLATFORMS="cuda",JAX_PLATFORM_NAME="gpu",
                     XLA_PYTHON_CLIENT_PREALLOCATE="false",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",
                     OPENPI_DATA_HOME="/root/nfs/embodied/benchmark/shared/checkpoints/openpi",PYTHONPATH=".:policies/openpi/src")
            log=root/f"native_d{steps}_gpu2.log"
            state["phase"]="native_running"
            item.update(status="running",started_unix=time.time(),log=str(log))
            with log.open("x") as stream:
                process=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
                item["pid"]=process.pid
                item["create_time"]=psutil.Process(process.pid).create_time()
                save()
                while process.poll() is None:
                    item["last_live_poll_unix"]=time.time()
                    save()
                    time.sleep(5)
                item.update(returncode=process.returncode,finished_unix=time.time())
            if process.returncode:
                item["status"]="failed"
                raise RuntimeError(f"Native D{steps} failed with exit code {process.returncode}")
            result=json.loads((output/"result.json").read_text())
            if len(result.get("rows",[]))!=100 or not result.get("repeatability",{}).get("passed") or result.get("mode")!="steady":
                item["status"]="invalid_result"
                raise RuntimeError("Native result did not pass full steady/repeatability requirements")
            item["status"]="passed"
            save()
        state["phase"]="complete"
        state["finished_unix"]=time.time()
        save()
    except BaseException:
        state["phase"]="failed"
        state["traceback"]=traceback.format_exc()
        save()
        raise


if __name__=="__main__":
    main()
