"""Read-only 30-second GPU availability watch while implementation is prepared."""
import json
import subprocess
import time
from preparations.idea1_v3.io import atomic
from .common import OUT


def scan():
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader'], text=True)
    occupied = {line.split(',')[0].strip() for line in apps.splitlines() if line.strip()}
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
        '--format=csv,noheader,nounits'], text=True)
    free, rows = [], []
    for line in raw.splitlines():
        index, uuid, used = [x.strip() for x in line.split(',')]
        row = dict(gpu=int(index), uuid=uuid, used_mib=float(used), occupied=uuid in occupied)
        rows.append(row)
        if not row['occupied'] and row['used_mib'] <= 1500:
            free.append((row['gpu'], uuid))
    result = dict(time=time.time(), gpus=rows, apps=apps, free=free)
    atomic(OUT/'resource_scan.json', result)
    return result


def main():
    import os
    atomic(OUT/'resource_watch_process.json', dict(pid=os.getpid(), started=time.time()))
    previous = None
    while not (OUT/'stop_resource_watch.json').exists():
        try:
            result = scan()
        except (OSError, subprocess.SubprocessError) as error:
            atomic(OUT/'resource_scan_error.json', dict(error=repr(error), time=time.time()))
            time.sleep(30)
            continue
        signature = json.dumps([(r['gpu'], r['occupied']) for r in result['gpus']])
        if signature != previous:
            print(json.dumps(result), flush=True)
            previous = signature
        time.sleep(30)
    atomic(OUT/'resource_watch_exited.json', dict(time=time.time()))


if __name__ == '__main__':
    main()
