"""Retain the two owned GPUs across independently frozen research cohorts.

The new6GiB helper must be ready before the old16GiB guardian receives its
explicit release file. After its owned children exit and its borrower lock is
released, the new context grows to16GiB. Original scheduler GPU leases and
64MiB helpers are unchanged. Only --launch performs this resource handoff.
"""
import argparse
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import select
import subprocess
import time

from preparations.ease_insights_20261103.reservoir_v1 import guardian as old
from preparations.ease_insights_20261103.resource_pool_v3.keeper import STAGING_MIB

ROOT = old.ROOT
RESOURCE = old.bc.EXPERIMENT / 'resource_pool_v3'
MODULE = 'preparations.ease_insights_20261103.resource_pool_v3.keeper'
OLD_AMENDMENT_SHA = '524357cd305ecde913362f023474465d1067711e9f214970a1729b219d8bca83'
require, read, sha, write = old.require, old.read, old.sha, old.write


class Client(old.Client):
    def start(self):
        control_read, self.writer = os.pipe()
        self.reader, ready_write = os.pipe()
        own = self.runtime.self_identity()
        argv = ['/usr/bin/python3','-B','-u','-m',MODULE,'--gpu-uuid',self.gpu_uuid,
            '--control-fd',str(control_read),'--ready-fd',str(ready_write),'--nonce',self.nonce,
            '--parent-pid',str(own['pid']),'--parent-start-ticks',own['start_ticks']]
        self.log = (self.output/'reservoir.log').open('a')
        try:
            self.child = subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=self.log,
                stderr=subprocess.STDOUT,close_fds=True,pass_fds=(control_read,ready_write))
        finally:
            os.close(control_read); os.close(ready_write)
        self.identity = self.runtime.identity(self.child.pid)
        self.staging_ready = self.receive('ready',STAGING_MIB)
        write(self.output/'staging_ready.json',self.staging_ready)
        return self.staging_ready


def verify_plan(path, expected):
    require(sha(path)==expected, 'Resource pool plan SHA differs')
    value=read(path)
    require(value.get('schema')=='ease-resource-pool-v3' and value.get('idle_mib')==16384
        and value.get('running_mib')==8192 and value.get('staging_mib')==6144
        and value.get('maximum_model_GPUs')==2 and set(value.get('slots',{}))==set(old.HOST_KEEPERS),
        'Resource pool settings differ')
    old.verify_amendment(old.RESOURCE_ROOT/'resource_amendment.json',OLD_AMENDMENT_SHA)
    for name,digest in value['source_hashes'].items():
        p=ROOT/name
        require(p.resolve().is_relative_to(ROOT) and sha(p)==digest,'Resource source changed: '+name)
    for name,pin in value['completed_study_evidence'].items():
        require(sha(pin['path'])==pin['sha256'],'Completed-study evidence changed: '+name)
    accepted=read(value['completed_study_evidence']['module_cases']['path'])
    selection=read(value['completed_study_evidence']['module_selection']['path'])
    independent=read(value['completed_study_evidence']['module_statistics']['path'])
    require(accepted.get('status')=='pass' and accepted.get('new_cases')==320
        and accepted.get('preserved_control_cells')==160
        and independent.get('status')=='pass' and independent.get('differences')==[]
        and independent.get('analytical_cells')==480
        and selection.get('selection',{}).get('status')=='no_candidate',
        'Resource upgrade must follow the accepted module branch without a candidate')
    return value


def old_idle(spec, observer):
    actual=observer.identity(spec['guardian']['pid'])
    require(old.bc.legacy.identity.same_process(actual,spec['guardian'])
        and actual.get('uid')==os.getuid(), 'Previous guardian identity changed')
    slot=Path(spec['directory'])
    status,ack=read(slot/'status.json'),read(slot/'keeper_latest_ack.json')
    require(status.get('phase')=='idle_holding_16GiB' and 0<=time.time()-status['time']<=15
        and ack==spec['latest_ack'] and ack.get('allocated_bytes')==16384*old.km.base.MIB,
        'Previous guardian is not idle with its pinned16GiB helper')
    helper=observer.identity(ack['pid'])
    require(helper.get('start_ticks')==str(ack['proc_start_ticks']) and helper.get('ppid')==actual['pid']
        and helper.get('state') not in old.bc.legacy.identity.DEAD_STATES
        and actual.get('children')==[str(ack['pid'])], 'Previous guardian still has science or lost its helper')
    require(not (slot/'release').exists(), 'Previous release was already requested')
    return dict(guardian=actual,helper=helper,status=status,ack=ack)


def process_memory(gpu_uuid):
    raw=subprocess.check_output(['nvidia-smi','--id='+gpu_uuid,
        '--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=15)
    rows=[]
    for line in raw.splitlines():
        if line.strip():
            actual,pid,memory=[s.strip() for s in line.split(',')]
            require(actual==gpu_uuid,'Assigned GPU query returned another UUID')
            rows.append((int(pid),float(memory)))
    return rows


def staging_binding(gpu_uuid,rows,old_host_pid):
    original=old.HOST_KEEPERS[gpu_uuid]
    pids={p for p,_ in rows};extra=pids-{original,old_host_pid}
    require(len(rows)==3 and len(extra)==1 and {original,old_host_pid}<=pids,
            'Staging requires exactly original, previous16GiB and new6GiB contexts')
    new_host=next(iter(extra));memory=dict(rows)
    require(0<memory[original]<=512 and 16384<=memory[old_host_pid]<=16896
        and STAGING_MIB<=memory[new_host]<=STAGING_MIB+512,'Staging context sizes differ')
    return new_host


def transfer(client, spec, observer, gpu_uuid, release_old, wait_old, take_borrower_lock,
             observe=process_memory, observation_timeout=0):
    """Dependency-injected ordering contract; no old release before a real6GiB receipt."""
    before=old_idle(spec,observer)
    ready=client.start()
    require(ready.get('allocated_bytes')==STAGING_MIB*old.km.base.MIB and client.live(),
            'New6GiB context is not ready; preserve the old guardian')
    def probe(validate):
        deadline=time.monotonic()+observation_timeout
        while True:
            try:
                return validate(observe(gpu_uuid))
            except (RuntimeError,OSError,ValueError,subprocess.SubprocessError):
                if time.monotonic()>=deadline:
                    raise
                time.sleep(.25)
    new_host=probe(lambda rows:staging_binding(gpu_uuid,rows,spec['old_host_pid']))
    old_idle(spec,observer)
    release_old(before,ready,new_host)
    wait_old()
    take_borrower_lock()
    probe(lambda rows:old.owned_observation(gpu_uuid,rows,new_host,STAGING_MIB))
    grown=client.resize(16384)
    require(grown.get('allocated_bytes')==16384*old.km.base.MIB,'New16GiB target not reached')
    observation=probe(lambda rows:old.owned_observation(gpu_uuid,rows,new_host,16384))
    return dict(new_host_pid=new_host,staging_ready=ready,grown_ready=grown,observation=observation)


class Session(old.Session):
    def execution_revision(self,observation):
        value=super().execution_revision(observation)
        value.update(schema='ease-resource-pool-science-v3',controller_source_sha256=sha(__file__),
            reused_resource_v1_source_sha256=sha(old.__file__),scientific_runtime_unchanged=False,
            original_scientific_manifests_changed=False,
            scope='New independently frozen cohort under retained original GPU leases; explicit6-to16GiB resource handoff')
        return value


def request_binding(value,plan):
    cohort=value.get('cohort')
    require(cohort in plan['allowed_cohorts'] and re.fullmatch('[a-z][a-z0-9_]*',cohort),
        'Unapproved local scientific cohort')
    output=Path(value['output']).resolve();base=old.bc.EXPERIMENT/cohort
    require(output in (base,base/'discovery',base/'confirmation',base/'calibration'),
        'Scientific output escaped the allowed cohort roots')
    require(value.get('schema')=='ease-resource-pool-request-v3'
        and sha(output/'manifest.json')==value.get('manifest_file_sha256'),'Request raw manifest differs')
    manifest=read(output/'manifest.json')
    payload=dict(manifest);declared=payload.pop('manifest_hash',None)
    canonical=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    require(declared==canonical==value.get('manifest_hash'),'Request canonical manifest differs')
    prefix='preparations.ease_insights_20261103.'+cohort
    for name,digest in manifest['source_files'].items():
        p=ROOT/name
        require(p.resolve().is_relative_to(ROOT) and sha(p)==digest,'Scientific source changed: '+name)
    for tail in ('common.py','runner.py','server.py','worker.py'):
        require(prefix.replace('.','/')+'/'+tail in manifest['source_files'],
            'Scientific module absent from frozen source map: '+tail)
    return output,prefix


def serve_request(path,args,plan,guard,client,new_host,slot):
    value=read(path);output,prefix=request_binding(value,plan)
    common=importlib.import_module(prefix+'.common')
    runner=importlib.import_module(prefix+'.runner');worker=importlib.import_module(prefix+'.worker')
    manifest=common.read_manifest(output/'manifest.json',verify_sources=True)
    stage=common.stage_for(manifest)
    conditions=manifest['planned_conditions'];all_scenes=common.cases(manifest,stage)
    require(0<len(all_scenes)*len(conditions)<=plan['maximum_episodes_per_request'], 'Scientific cohort budget differs')
    shards=value.get('shards',2)
    require(type(shards) is int and shards in (1,2),'Only one or two fixed task shards are supported')
    shard_index=plan['slots'][args.gpu_uuid]['shard_index'] if shards==2 else 0
    scenes=common.cases(manifest,stage,shard_index,shards)
    label=f'shard{shard_index:02d}of{shards:02d}'
    results=[]
    for cid in conditions:
        with old.bc.lock(output/'claims'/(cid+'_'+label+'.lock'),optional=True) as acquired:
            if not acquired:
                continue
            marker=output/cid/(stage+'_'+label+'_complete.json')
            if marker.exists():
                done=read(marker)
                require(done.get('complete') is True and done.get('manifest_hash')==manifest['manifest_hash']
                    and done.get('condition')==cid
                    and done.get('completed_after',done.get('already_completed'))==len(scenes),
                    'Existing shard completion marker differs')
                continue
            verify_plan(args.plan,args.plan_sha256);guard.full()
            port=old.bc.fresh_port()
            session=Session(guard,args.gpu_uuid,port,manifest,client,new_host,
                dict(path=str(args.plan),sha256=args.plan_sha256))
            session.runtime.scan_owned(args.gpu_uuid,session.keeper,512)
            options=argparse.Namespace(output=str(output),condition=cid,stage=stage,shard_index=shard_index,
                shards=shards,gpu_uuid=args.gpu_uuid,port=port,launch=False,preflight_only=False)
            prepared=runner.run(options)
            require(prefix+'.server' in prepared['server'] and prefix+'.worker' in prepared['worker'],
                    'Command builder did not use the pinned cohort modules')
            write(slot/'status.json',dict(phase='running_condition',time=time.time(),guardian_pid=os.getpid(),
                cohort=value['cohort'],condition=cid,manifest_hash=manifest['manifest_hash']))
            result=old.execute_condition(prepared,session,old.bc.legacy.Science(runner=runner,worker=worker),client)
            results.append(dict(condition=cid,result=result))
    return dict(phase='assignment_complete',time=time.time(),request_sha256=sha(path),results=results)


def hold_attention(slot,client,error,guard,gpu_uuid,resume_validator=None):
    value=dict(phase='attention_holding',time=time.time(),error=repr(error),guardian_pid=os.getpid())
    write(slot/'attention.json',value)
    attention_sha=sha(slot/'attention.json');seen=None
    while not (slot/'release').exists():
        require(client.live(),'Retained helper died during attention')
        write(slot/'status.json',dict(value,time=time.time(),reservoir=client.ready))
        control=slot/'resume.json'
        if resume_validator is not None and control.exists() and sha(control)!=seen:
            seen=sha(control)
            try:
                resume_validator(read(control),attention_sha)
                write(slot/'resume_receipts'/(str(time.time_ns())+'.json'),
                    dict(accepted=True,request=read(control),request_sha256=seen,time=time.time()))
                if sha(control)==seen:
                    control.unlink()
                return True
            except Exception as rejected:
                write(slot/'resume_rejected.json',dict(error=repr(rejected),request_sha256=seen,time=time.time()))
        time.sleep(2)
    return False


def run(args):
    plan=verify_plan(args.plan,args.plan_sha256)
    require(args.gpu_uuid in old.HOST_KEEPERS,'Unassigned GPU UUID')
    spec=plan['slots'][args.gpu_uuid];guard=old.bc.HolderGuard();observer=old.bc.legacy.Runtime()
    if not args.launch:
        return dict(mode='dry_run',original_holders=guard.full(),old_idle=old_idle(spec,observer),no_CUDA_started=True)
    slot=RESOURCE/'sessions'/args.gpu_uuid;slot.mkdir(parents=True,exist_ok=True)
    client=Client(args.gpu_uuid,slot,observer);released=[False];borrower=[None]
    with old.bc.lock(RESOURCE/'singleton'/(args.gpu_uuid+'.lock')),old.bc.lock(old.bc.CAMPAIGN/'broker.lock',shared=True):
        guard.full();old_idle(spec,observer)
        write(slot/'guardian.json',dict(observer.self_identity(),source_sha256=sha(__file__),
            plan_path=str(args.plan),plan_sha256=args.plan_sha256,gpu_uuid=args.gpu_uuid))
        old_pids=(spec['guardian']['pid'],spec['latest_ack']['pid'])
        pidfds=[args.old_guardian_pidfd,args.old_helper_pidfd]
        # The system-Python launcher supplies identity-bound pidfds; the model
        # interpreter need not expose os.pidfd_open. These are observation
        # descriptors, never either scheduler GPU lease.
        for fd,pid in zip(pidfds,old_pids):
            require(type(fd) is int and fd>=3 and os.readlink('/proc/self/fd/'+str(fd))=='anon_inode:[pidfd]',
                    'Inherited old-process pidfd is missing')
            fields=dict(line.split(':',1) for line in Path('/proc/self/fdinfo/'+str(fd)).read_text().splitlines() if ':' in line)
            require(int(fields.get('Pid','-1'))==pid,'Inherited pidfd does not identify the pinned old process')
        def release_old(before,ready,new_host):
            guard.full();old_idle(spec,observer)
            write(slot/'before_old_release.json',dict(time=time.time(),before=before,staging_ready=ready,
                new_host_pid=new_host,old_host_pid=spec['old_host_pid']))
            write(Path(spec['directory'])/'release',dict(schema='ease-resource-handoff-release-v3',
                old_guardian_pid=old_pids[0],old_helper_pid=old_pids[1],new_guardian_pid=os.getpid(),
                new_ready=ready,plan_sha256=args.plan_sha256,time=time.time()))
            released[0]=True
        def wait_old():
            deadline=time.monotonic()+120
            for fd in pidfds:
                poll=select.poll();poll.register(fd,select.POLLIN)
                while not poll.poll(200):
                    require(client.live(),'Staging helper lost while waiting for old exit')
                    require(time.monotonic()<deadline,'Previous guardian/helper did not exit')
        def take_borrower():
            path=old.bc.NEW_ROOT/'resources'/(args.gpu_uuid+'.borrower.lock')
            path.parent.mkdir(parents=True,exist_ok=True)
            borrower[0]=path.open('a')
            fcntl.flock(borrower[0].fileno(),fcntl.LOCK_EX)
        try:
            write(slot/'status.json',dict(phase='staging_handoff',time=time.time()))
            result=transfer(client,spec,observer,args.gpu_uuid,release_old,wait_old,take_borrower,
                            observation_timeout=30)
            new_host=result['new_host_pid'];write(slot/'handoff_complete.json',dict(result,time=time.time(),
                old_guardian_exited=True,old_helper_exited=True,original_GPU_leases_unchanged=True))
            def validate_resume(request,attention_sha):
                own=observer.self_identity()
                require(request.get('schema')=='ease-resource-pool-resume-v3'
                    and request.get('guardian_pid')==own['pid']
                    and str(request.get('guardian_start_ticks'))==own['start_ticks']
                    and request.get('plan_sha256')==args.plan_sha256
                    and request.get('attention_sha256')==attention_sha,
                    'Resume does not bind this exact guardian and attention')
                require(own['children']==[str(client.child.pid)],'Science children remain; do not resume')
                verify_plan(args.plan,args.plan_sha256);guard.full()
                client.resize(16384)
                old.owned_observation(args.gpu_uuid,process_memory(args.gpu_uuid),new_host,16384)
            while not (slot/'release').exists():
                try:
                    require(client.live(),'Persistent resource pool helper died')
                    guard.quick(args.gpu_uuid)
                    for request in sorted((RESOURCE/'requests').glob('*.json')):
                        outcome=slot/'outcomes'/request.name
                        if outcome.exists():
                            continue
                        try:
                            value=serve_request(request,args,plan,guard,client,new_host,slot)
                        except Exception as error:
                            write(outcome,dict(phase='attention',error=repr(error),time=time.time(),
                                request_path=str(request),request_sha256=sha(request),automatic_scientific_retry=False))
                            raise
                        write(outcome,value)
                    write(slot/'status.json',dict(phase='idle_holding_16GiB',time=time.time(),guardian_pid=os.getpid(),
                        gpu_uuid=args.gpu_uuid,reservoir=client.ready,original_GPU_leases_unchanged=True))
                    time.sleep(2)
                except Exception as error:
                    if not hold_attention(slot,client,error,guard,args.gpu_uuid,validate_resume):
                        break
        except Exception as error:
            if released[0] and client.live():
                # A routine execution/IO failure must not close the context
                # after the previous holder has already released its memory.
                hold_attention(slot,client,error,guard,args.gpu_uuid)
            else:
                raise
        finally:
            client.close()
            if borrower[0] is not None:
                borrower[0].close()
            for fd in pidfds:
                os.close(fd)
    return dict(phase='explicit_release',gpu_uuid=args.gpu_uuid)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--gpu-uuid',required=True);parser.add_argument('--launch',action='store_true')
    parser.add_argument('--old-guardian-pidfd',type=int);parser.add_argument('--old-helper-pidfd',type=int)
    print(json.dumps(run(parser.parse_args()),indent=2))


if __name__=='__main__':
    main()

