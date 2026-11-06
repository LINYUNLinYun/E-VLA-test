"""Launch one owned resource handoff using observation pidfds, never GPU lease FDs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/root/nfs/embodied/benchmark')
RESOURCE=ROOT/'results/ease_insights_20261103/resource_pool_v3'
EXPECTED='bccfc4c41d630bb4c9609a67e0972a119abb0d529ad367760dfc8302be6b6ff0'


def main(uuid):
    plan_path=RESOURCE/'plan.json';assert hashlib.sha256(plan_path.read_bytes()).hexdigest()==EXPECTED
    plan=json.loads(plan_path.read_text());spec=plan['slots'][uuid]
    registry=RESOURCE/'launches.json';rows=json.loads(registry.read_text())['processes'] if registry.exists() else []
    assert not any(row['gpu_uuid']==uuid for row in rows),'Already launched; inspect the same process'
    slot=RESOURCE/'sessions'/uuid;slot.mkdir(parents=True,exist_ok=True);assert not (slot/'guardian.json').exists()
    fds=[]
    try:
        for reference in (spec['guardian'],dict(pid=spec['latest_ack']['pid'],start_ticks=str(spec['latest_ack']['proc_start_ticks']))):
            process=Path('/proc')/str(reference['pid'])
            fields=(process/'stat').read_text().rsplit(')',1)[1].split()
            assert fields[19]==str(reference['start_ticks']) and fields[0] not in ('Z','X')
            fd=os.pidfd_open(reference['pid']);fds.append(fd)
            assert (process/'stat').read_text().rsplit(')',1)[1].split()[19]==str(reference['start_ticks'])
        argv=[str(ROOT/'policies/realtime-vla-flash/.venv/bin/python'),'-B','-u','-m',
            'preparations.ease_insights_20261103.resource_pool_v3.guardian','--plan',str(plan_path),'--plan-sha256',EXPECTED,
            '--gpu-uuid',uuid,'--old-guardian-pidfd',str(fds[0]),'--old-helper-pidfd',str(fds[1]),'--launch']
        with (slot/'guardian.log').open('a') as log:
            child=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                close_fds=True,pass_fds=tuple(fds),start_new_session=True)
        ticks=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
        row=dict(pid=child.pid,start_ticks=ticks,argv=argv,gpu_uuid=uuid,launched_at=time.time(),plan_sha256=EXPECTED,
            old_guardian_pid=spec['guardian']['pid'],old_helper_pid=spec['latest_ack']['pid'],observation_pidfds_only=True,
            launcher_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        rows.append(row);temporary=registry.with_suffix('.tmp');temporary.write_text(json.dumps(dict(time=time.time(),processes=rows),indent=2)+'\n');temporary.replace(registry)
        print(json.dumps(row))
    finally:
        for fd in fds:os.close(fd)


if __name__=='__main__':main(sys.argv[1])
