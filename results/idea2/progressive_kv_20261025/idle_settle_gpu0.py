"""Allow utilization telemetry to decay between processes; never ignore owners."""
import subprocess
import time
import xml.etree.ElementTree as ET

UUID='GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1'

def wait_for_idle(timeout=30):
    begin=time.monotonic()
    history=[]
    stable=0
    while True:
        raw=subprocess.run(['nvidia-smi','-i',UUID,'-q','-x'],check=True,capture_output=True,text=True).stdout
        gpu=ET.fromstring(raw).find('gpu')
        if gpu is None or gpu.findtext('uuid')!=UUID:
            raise RuntimeError('GPU UUID mismatch during idle settling')
        processes=[dict(pid=x.findtext('pid'),type=x.findtext('type')) for x in gpu.findall('processes/process_info')]
        memory=float(gpu.findtext('fb_memory_usage/used').split()[0])
        utilization=float(gpu.findtext('utilization/gpu_util').split()[0])
        row=dict(elapsed_seconds=time.monotonic()-begin,memory_mib=memory,
                 utilization_percent=utilization,processes=processes)
        history.append(row)
        if processes:
            raise RuntimeError('GPU has a process owner; do not wait it away: '+str(row))
        stable=stable+1 if memory<=256 and utilization<=1 else 0
        if stable>=2:
            return history
        if time.monotonic()-begin>=timeout:
            raise RuntimeError('GPU did not settle while unowned: '+str(history))
        time.sleep(1.5)

if __name__=='__main__':
    import json
    print(json.dumps(wait_for_idle(),indent=2))
