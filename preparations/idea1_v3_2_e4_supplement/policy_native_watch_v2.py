"""Bounded idle-only GPU0/1/2 allocator and automatic native100 shard supervisor.

Never inspects/uses GPU3. An unchanged busy GPU is expected, not a quality zero.
No model construction until an empty sample plus immediate UUID/empty recheck, child initial
guard, and child's final pre-CUDA guard. All processes and cleanup are our own.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import pickle
import signal
import socket
import struct
import subprocess
import sys
import time
import traceback
from preparations.idea1_v3_2.transfer_protocol import load_manifest,sha
from preparations.idea1_v3_1.common import atomic_json

ROOT=Path(__file__).resolve().parents[2]


def rpc(port,payload,timeout=10):
    def receive(connection,n):
        result=b""
        while len(result)<n:
            chunk=connection.recv(n-len(result))
            if not chunk:raise ConnectionError("RPC closed early")
            result+=chunk
        return result
    with socket.create_connection(("127.0.0.1",port),timeout=timeout) as connection:
        connection.settimeout(timeout);data=pickle.dumps(payload,protocol=4)
        connection.sendall(struct.pack("!I",len(data))+data)
        size=struct.unpack("!I",receive(connection,4))[0]
        result=pickle.loads(receive(connection,size))
    if not result.get("ok"):raise RuntimeError(str(result))
    return result


def sample(index):
    if index not in (0,1,2):raise ValueError("Only non-owned physical candidates0/1/2 allowed")
    command=["nvidia-smi","-i",str(index)]
    processes=subprocess.check_output(command+["--query-compute-apps=pid,gpu_uuid,used_memory","--format=csv,noheader"],text=True).strip()
    device=subprocess.check_output(command+["--query-gpu=index,uuid,memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True).strip()
    fields=[field.strip() for field in device.split(",")]
    if len(fields)!=4:raise RuntimeError("Unrecognized nvidia identity: "+device)
    memory,utilization=int(fields[2]),int(fields[3])
    return dict(index=index,uuid=fields[1],compute_processes=processes,memory_mib=memory,
        utilization_percent=utilization,idle=not processes and memory<=16 and utilization==0,time_unix=time.time())


def stop_owned(process,port=None):
    if process is None or process.poll() is not None:return
    if port is not None:
        try:rpc(port,{"op":"shutdown"},timeout=5)
        except Exception:pass
        try:process.wait(timeout=15);return
        except subprocess.TimeoutExpired:pass
    process.terminate()
    try:process.wait(timeout=15)
    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=15)


def supervise(args):
    manifest=load_manifest(args.protocol)
    if manifest["sequence_count"]!=100 or manifest["modes"]!=["full_d10","v14_d10"]:
        raise ValueError("Fixed native100 two-mode protocol required")
    output=Path(args.output)
    if output.exists():raise FileExistsError("New immutable supervisor output required")
    output.mkdir(parents=True)
    receipt=dict(status="waiting_idle_gpu",supervisor_pid=os.getpid(),started_unix=time.time(),
        deadline_unix=args.deadline_unix,manifest_hash=manifest["manifest_hash"],protocol=str(args.protocol),
        source_sha256={str(Path(__file__)):sha(__file__)},candidate_indices=[0,1,2],never_scan_or_use_gpu3=True,
        required_stable_idle_seconds=0,sample_interval_seconds=30,watcher_version=2,allocations=[],shards=[])
    service=client=None;selected=None
    atomic_json(output/"supervisor.json",receipt)
    def expired(signum,frame):raise TimeoutError("SUPERVISOR_DEADLINE_OR_TERMINATION")
    signal.signal(signal.SIGTERM,expired);signal.signal(signal.SIGINT,expired)
    signal.signal(signal.SIGALRM,expired)
    signal.alarm(max(1,int(args.deadline_unix-time.time())))
    previous={}
    try:
        while time.time()<args.deadline_unix:
            for index in (0,1,2):
                current=sample(index)
                with (output/"gpu_samples.jsonl").open("a") as stream:
                    stream.write(json.dumps(current)+"\n");stream.flush()
                before=previous.get(index)
                if current["idle"]:
                    # Explicit immediate final parent-side launch check.
                    final=sample(index)
                    if final["idle"] and final["uuid"]==current["uuid"]:
                        selected=final;break
                previous[index]=current
            if selected:break
            time.sleep(min(30,max(0,args.deadline_unix-time.time())))
        if selected is None:
            receipt.update(status="deadline_no_idle_gpu",completed_unix=time.time());return
        with socket.socket() as probe:
            probe.bind(("127.0.0.1",args.port))
        stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        service_receipt=output/("service_"+stamp+".json")
        command=[str(ROOT/"policies/realtime-vla-flash/.venv/bin/python"),"-m",
            "preparations.idea1_v3_2_e4_supplement.policy_calvin_server","--family","calvin_abc_d_native",
            "--protocol",str(args.protocol),"--port",str(args.port),"--output",str(service_receipt),
            "--deadline-unix",str(args.deadline_unix),"--checkpoint",str(ROOT/"shared/checkpoints/e4_supplement/RLinf/RLinf-Pi0-CALVIN-ABC-D-SFT/model.safetensors"),
            "--norm-stats",str(ROOT/"shared/checkpoints/e4_supplement/RLinf/RLinf-Pi0-CALVIN-ABC-D-SFT/InternRobotics/InternData-Calvin_ABC/norm_stats.json"),"--launch"]
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES=selected["uuid"],JAX_PLATFORMS="cpu")
        service_log=(output/"service.log").open("x")
        service=subprocess.Popen(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,stdout=service_log,stderr=subprocess.STDOUT)
        receipt.update(status="loading_native_model",selected_gpu=selected,service_pid=service.pid,
            service_command=command,service_receipt=str(service_receipt))
        atomic_json(output/"supervisor.json",receipt)
        while time.time()<args.deadline_unix:
            if service.poll() is not None:raise RuntimeError("Native service exited during startup: "+str(service.returncode))
            try:
                health=rpc(args.port,{"op":"health"},timeout=2)["protocol"]
                if health.get("manifest_hash")!=manifest["manifest_hash"] or health.get("server_pid")!=service.pid or health.get("gpu_uuid")!=selected["uuid"]:
                    raise ValueError("Ready service namespace/manifest/physical UUID mismatch")
                if not health.get("final_model_cuda_allocation_guard",{}).get("checked"):
                    raise ValueError("Final immediate allocation guard missing")
                receipt["service_health"]=health;break
            except (ConnectionError,OSError,socket.timeout):time.sleep(2)
        else:raise TimeoutError("Model startup exceeded execution deadline")
        receipt["status"]="executing_fixed100_shards";atomic_json(output/"supervisor.json",receipt)
        driver_environment=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",
            PYTHONPATH=str(ROOT)+":"+str(ROOT/"calvin/calvin_env"))
        driver=[str(ROOT/"calvin/.venv/bin/python"),"-m","preparations.idea1_v3_2_e4_supplement.evaluation_v2",
            "--protocol",str(args.protocol),"--sequence-file",str(ROOT/"results/idea1_v3_2_e4_supplement/native_preflight/calvin_formal1000/sequences.json"),
            "--sequence-count","100","--max-new-sequences","10","--modes","full_d10","v14_d10",
            "--preflight-input",str(ROOT/"results/idea1_v3_2_e4_supplement/native_preflight/calvin_captured_formalD0/native_input_gripper_sign.npz"),
            "--port",str(args.port),"--output",str(output/"episodes"),"--deadline-unix",str(args.deadline_unix),"--video","--launch"]
        last_completed=-1
        for shard in range(11):
            if time.time()>=args.deadline_unix:raise TimeoutError("Fixed100 execution deadline reached")
            log=(output/("shard_%02d.log"%shard)).open("x")
            client=subprocess.Popen(driver,cwd=ROOT,env=driver_environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            row=dict(index=shard,client_pid=client.pid,started_unix=time.time(),command=driver)
            receipt["shards"].append(row);atomic_json(output/"supervisor.json",receipt)
            while client.poll() is None:
                if time.time()>=args.deadline_unix:raise TimeoutError("Native shard deadline reached")
                if service.poll() is not None:raise RuntimeError("Native owned service exited during shard")
                time.sleep(2)
            row.update(exit_code=client.returncode,completed_unix=time.time());log.close()
            if client.returncode:raise RuntimeError("Technical shard failure preserved; not a policyzero: "+str(client.returncode))
            progress=json.loads((output/"episodes/progress.json").read_text())
            completed=progress["completed_cases"];row["completed_cases"]=completed
            atomic_json(output/"supervisor.json",receipt)
            if completed==200 and progress["status"]=="complete":
                receipt.update(status="complete",completed_cases=200,completed_unix=time.time());return
            if completed<=last_completed:raise RuntimeError("No forward progress in automatic native100 continuation")
            last_completed=completed
        raise RuntimeError("All100 target not achieved after bounded10-sequence continuations")
    except Exception:
        receipt.update(status="technical_failure",error=traceback.format_exc(),completed_unix=time.time());raise
    finally:
        stop_owned(client)
        stop_owned(service,args.port if service is not None else None)
        if service is not None:receipt["service_exit_code"]=service.poll()
        atomic_json(output/"supervisor.json",receipt)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--protocol",required=True);p.add_argument("--output",required=True)
    p.add_argument("--port",type=int,default=19144);p.add_argument("--deadline-unix",type=float,required=True);p.add_argument("--launch",action="store_true")
    args=p.parse_args()
    if not args.launch:
        print(json.dumps(dict(dry_run=True,candidates=[0,1,2],stable_idle_seconds=0,target_sequences=100,conditions=200,shard_size=10,**vars(args))));return
    supervise(args)


if __name__=="__main__":main()
