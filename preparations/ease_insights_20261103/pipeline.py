"""Deterministic experiment continuation, separate from the single GPU scheduler.

Without --launch this only inspects. --launch enables stage transitions and queue
append; --once bounds it to one pass. This module never queries GPUs or terminates
processes. It does not complete the user's goal: terminal success means report_ready.
"""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

WORKSPACE=Path(__file__).resolve().parents[2]
PREP=Path(__file__).resolve().parent
MODULE='preparations.ease_insights_20261103.scheduler'
VLM_MODULE='preparations.ease_insights_20261103.vlm'
FLASH_PYTHON=WORKSPACE/'policies/realtime-vla-flash/.venv/bin/python'
TERMINAL={'report_ready','needs_attention','needs_scientific_review'}


def read(path,default=None):
    path=Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.%d.tmp'%os.getpid())
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


@contextmanager
def file_lock(path,blocking=True):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:yield handle
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


def append_jobs(queue_path,new_jobs):
    """Atomic, cooperative-lock queue append; existing entries are immutable."""
    queue_path=Path(queue_path)
    with file_lock(queue_path.with_name(queue_path.name+'.lock')):
        original_bytes=queue_path.read_bytes();queue=json.loads(original_bytes)
        existing={j['id']:j for j in queue['jobs']}
        if len(existing)!=len(queue['jobs']):raise ValueError('Current queue has duplicate IDs')
        proposed=set();added=[]
        for job in new_jobs:
            job_id=job['id']
            if job_id in proposed:raise ValueError('New jobs contain duplicate ID: '+job_id)
            proposed.add(job_id)
            if not isinstance(job.get('argv'),list) or not job['argv'] or not job.get('complete_file'):
                raise ValueError('Malformed queue job: '+job_id)
            if job_id in existing:
                if existing[job_id]!=job:raise ValueError('Conflicting existing job ID: '+job_id)
            else:
                if any(Path(old['complete_file']).resolve()==Path(job['complete_file']).resolve() for old in existing.values()):
                    raise ValueError('Different job ID already owns the same completion target: '+job_id)
                added.append(job)
        all_ids=set(existing)|proposed
        if any(not set(j.get('depends_on',[])).issubset(all_ids) for j in added):
            raise ValueError('New job dependency is absent from the resulting queue')
        if added:
            # Detect writers that ignored this lock before committing.
            if queue_path.read_bytes()!=original_bytes:raise RuntimeError('Concurrent noncooperating queue writer detected')
            queue['jobs'].extend(added);write(queue_path,queue)
        return dict(added_ids=[j['id'] for j in added],already_present_ids=sorted(proposed-set(j['id'] for j in added)),
                    queue_sha256=sha(queue_path))


def with_assigned_gpu_wait(job,status_directory):
    """Add a resource wait without changing the original experiment command."""
    module='preparations.ease_insights_20261103.await_assigned_gpu'
    if module in job['argv']:return job
    wrapped=dict(job)
    wrapped['argv']=[sys.executable,'-m',module,'--gpu-uuid','{gpu_uuid}',
        '--status-file',str(Path(status_directory)/(job['id']+'.json')),'--',*job['argv']]
    wrapped['launch_gate']=dict(continuous_idle_seconds=60,poll_seconds=10,
        assigned_gpu_only=True,original_argv=job['argv'])
    return wrapped


def argv_value(argv,flag,default=None):
    for i,arg in enumerate(argv):
        if arg==flag:return argv[i+1] if i+1<len(argv) else default
        if arg.startswith(flag+'='):return arg.split('=',1)[1]
    return default


def proc_identity(pid,proc_root=Path('/proc')):
    """OS process identity, never a heartbeat/receipt inference."""
    directory=Path(proc_root)/str(pid)
    try:
        stat=(directory/'stat').read_text().rsplit(')',1)[1].split()
        command=(directory/'cmdline').read_bytes().split(b'\0')
        argv=[x.decode(errors='replace') for x in command if x]
        return dict(pid=int(pid),status='dead' if stat[0] in ('Z','X') else 'live',
                    state=stat[0],start_ticks=stat[19],argv=argv)
    except FileNotFoundError:
        return dict(pid=int(pid),status='absent' if not directory.exists() else 'unknown')
    except (OSError,IndexError,ValueError) as error:
        return dict(pid=int(pid),status='unknown',error=str(error))


def scheduler_match(process,queue,output):
    argv=process.get('argv',[])
    module=any(argv[i]=='-m' and i+1<len(argv) and argv[i+1]==MODULE for i in range(len(argv)))
    script=any(Path(x).name=='scheduler.py' and 'ease_insights_20261103' in x for x in argv)
    if not (module or script):return False
    return (Path(argv_value(argv,'--queue','/invalid')).resolve()==Path(queue).resolve() and
            Path(argv_value(argv,'--output','/invalid')).resolve()==Path(output).resolve())


def marker_status(path,kind,expected=None):
    path=Path(path)
    if not path.is_file():return dict(status='waiting',path=str(path),reason='missing_marker')
    try:value=read(path)
    except (OSError,ValueError) as error:return dict(status='invalid',path=str(path),reason=str(error))
    complete=(value.get('status')=='complete' if kind=='draft' else value.get('complete') is True)
    if not complete:
        failed=value.get('status') in ('technical_failure','failed','interrupted')
        return dict(status='failed' if failed else 'waiting',path=str(path),reason=value.get('status','not_complete'))
    for key,wanted in (expected or {}).items():
        if value.get(key)!=wanted:return dict(status='invalid',path=str(path),reason='Marker '+key+' differs from plan')
    return dict(status='complete',path=str(path))


@dataclass
class Paths:
    root:Path
    queue:Path
    scheduler_output:Path
    output:Path
    draft_plan:Path
    reports:Path
    @classmethod
    def defaults(cls,root,queue=None,scheduler_output=None,output=None,draft_plan=None,reports=None):
        root=Path(root).resolve()
        return cls(root,Path(queue or root/'queue.json').resolve(),
                   Path(scheduler_output or root/'scheduler').resolve(),Path(output or root/'pipeline').resolve(),
                   Path(draft_plan or PREP/'draft/replication').resolve(),
                   Path(reports or WORKSPACE/'reports/ease_insights_20261103').resolve())


class SchedulerWatch:
    """Adopt verified live scheduler; restart only after OS-confirmed exit."""
    def __init__(self,paths,initial_pid=None,probe=proc_identity,spawn=None):
        self.paths=paths;self.initial_pid=initial_pid;self.probe=probe;self.spawn=spawn or self._spawn
    def _spawn(self,argv,log_path):
        log_path.parent.mkdir(parents=True,exist_ok=True)
        with log_path.open('a') as stream:
            process=subprocess.Popen(argv,cwd=WORKSPACE,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        return process
    def observe(self,state):
        receipt=read(self.paths.scheduler_output/'controller_process.json',{})
        continuation=read(self.paths.root/'continuation.json',{})
        pids={int(p) for p in (state.get('pid'),self.initial_pid,receipt.get('pid'),continuation.get('scheduler_pid')) if p}
        processes=[self.probe(pid) for pid in sorted(pids)]
        matching=[p for p in processes if p['status']=='live' and scheduler_match(p,self.paths.queue,self.paths.scheduler_output)]
        unknown=[p for p in processes if p['status']=='unknown']
        if len(matching)>1:return dict(status='needs_attention',reason='multiple_matching_live_schedulers',processes=processes)
        if matching:
            process=matching[0];argv=process['argv']
            if int(argv_value(argv,'--max-gpus','2'))!=2 or float(argv_value(argv,'--idle-seconds','25'))!=0:
                return dict(status='needs_attention',reason='live_scheduler_configuration_differs',processes=processes)
            return dict(status='live',process=process,processes=processes)
        if unknown:return dict(status='needs_attention',reason='process_observation_uncertain_no_restart',processes=processes)
        return dict(status='exited',processes=processes)
    def ensure(self,state,queue,job_state,allow_restart=True):
        observation=self.observe(state)
        if observation['status']=='live':
            process=observation['process'];state.update(pid=process['pid'],start_ticks=process.get('start_ticks'),
                argv=process['argv'],status='live',last_verified_unix=time.time())
            return observation
        if observation['status']=='needs_attention':return observation
        jobs={j['id']:j for j in queue['jobs']}
        # Only CURRENT queue jobs matter; superseded failures in state are history.
        pending=[j for j in queue['jobs'] if job_state.get(j['id'],{}).get('status','pending')=='pending']
        if not pending:return dict(status='exited_no_pending',processes=observation['processes'])
        current_running=[(key,record) for key,record in job_state.items() if key in jobs and record.get('status')=='running']
        active=[]
        for key,record in current_running:
            process=self.probe(record['pid'])
            if process['status'] in ('live','unknown'):active.append(dict(job=key,process=process))
        if active:return dict(status='needs_attention',reason='scheduler_absent_but_job_handle_live_or_uncertain',jobs=active)
        ready=[j for j in pending if all(job_state.get(d,{}).get('status')=='complete' for d in j.get('depends_on',[]))
               and all(Path(p).is_file() for p in j.get('ready_files',[]))]
        if not ready:return dict(status='needs_attention',reason='pending_jobs_have_unmet_dependencies_or_ready_files')
        if state.get('launched_job_ids'):
            unfinished=[key for key in state['launched_job_ids'] if key in jobs and job_state.get(key,{}).get('status') not in ('complete','failed','interrupted')]
            if unfinished:return dict(status='needs_attention',reason='previous_restart_exited_with_unfinished_work_no_blind_retry',jobs=unfinished)
        if (self.paths.scheduler_output/'stop_after_jobs.json').exists():
            return dict(status='needs_attention',reason='scheduler_stop_marker_present')
        if not allow_restart:return dict(status='restart_would_be_needed',ready_jobs=[j['id'] for j in ready])
        # Holding this restart lock serializes all cooperating pipeline instances.
        with file_lock(self.paths.output/'scheduler_restart.lock'):
            again=self.observe(state)
            if again['status']!='exited':return again
            try:
                with file_lock(self.paths.scheduler_output/'controller.lock',blocking=False):pass
            except BlockingIOError:
                return dict(status='needs_attention',reason='controller_lock_owned_without_verified_pid_no_restart')
            argv=['python3','-m',MODULE,'--queue',str(self.paths.queue),'--output',str(self.paths.scheduler_output),
                  '--max-gpus','2','--scan-seconds','30','--idle-seconds','0']
            process=self.spawn(argv,self.paths.output/'scheduler_restart.log')
            state.update(pid=process.pid,status='starting',started_unix=time.time(),argv=argv,
                         launched_job_ids=list(jobs),restart_reason='previous_scheduler_OS_confirmed_exited_and_pending_work',
                         last_observation=again)
            write(self.paths.output/'scheduler_restart.json',state)
            return dict(status='restarted',pid=process.pid,argv=argv)


class Pipeline:
    def __init__(self,paths,initial_scheduler_pid=None,command_runner=None,scheduler_watch=None):
        self.paths=paths;self.state_path=paths.output/'state.json'
        self.state=read(self.state_path,dict(revision='ease-insights-pipeline-v1',created_unix=time.time(),branches={},scheduler={}))
        self.command_runner=command_runner
        self.watch=scheduler_watch or SchedulerWatch(paths,initial_scheduler_pid)
        self.marker_cache={}
        for name in ('draft','vlm'):self.state['branches'].setdefault(name,dict(phase='waiting_discovery'))
    def save(self):
        self.state['updated_unix']=time.time();self.state['goal_complete_claim']=False;write(self.state_path,self.state)
    def check(self,specs):
        results=[]
        for path,kind,expected in specs:
            path=Path(path)
            try:stat=path.stat();key=(stat.st_mtime_ns,stat.st_size)
            except FileNotFoundError:key=None
            cached=self.marker_cache.get(str(path))
            if cached is None or cached[0]!=key:
                self.marker_cache[str(path)]=(key,marker_status(path,kind,expected))
            results.append(self.marker_cache[str(path)][1])
        return dict(complete=bool(results) and all(r['status']=='complete' for r in results),
                    completed=sum(r['status']=='complete' for r in results),expected=len(results),markers=results)
    def discovery_specs(self,name):
        root=self.paths.root
        if name=='draft':
            master=read(root/'draft_discovery_manifest.json')
            if not master:raise ValueError('Missing frozen draft discovery manifest')
            full=[j[len('discovery_full_'):] for j in master['jobs'] if j.startswith('discovery_full_')]
            if len(full)!=1:raise ValueError('Draft master has no unique Full artifact source')
            specs=[(root/'draft_discovery'/('full_'+full[0])/'result.json','draft',dict(cases=40,mode='full',source_suite=full[0]))]
            for mode in ('flash','draft'):
                for source in master['source_suites']:
                    n=36 if source=='libero_spatial' else 40
                    specs.append((root/'draft_discovery'/(mode+'_'+source)/'result.json','draft',dict(cases=n,mode=mode,source_suite=source)))
                specs.append((root/'draft_smoke'/(mode+'_spatial')/'result.json','draft',dict(cases=4,mode=mode,source_suite='libero_spatial')))
            return specs
        manifest=read(root/'vlm/manifest.json')
        if not manifest:raise ValueError('Missing frozen VLM manifest')
        return [(root/'vlm'/c['id']/'discovery_shard00of01_complete.json','vlm',
                 dict(condition=c['id'],manifest_hash=manifest['manifest_hash'])) for c in manifest['conditions']]
    def relevant_failures(self,specs,queue,job_state):
        paths={str(Path(spec[0]).resolve()) for spec in specs}
        return [dict(job=j['id'],status=job_state.get(j['id'],{}).get('status')) for j in queue['jobs']
                if str(Path(j['complete_file']).resolve()) in paths and
                job_state.get(j['id'],{}).get('status') in ('failed','interrupted') and
                marker_status(j['complete_file'],'draft' if '/draft_' in j['complete_file'] else 'vlm')['status']!='complete']
    def run_command(self,branch,label,argv):
        record=self.state['branches'][branch]
        actions=record.setdefault('actions',{})
        if actions.get(label,{}).get('status')=='complete':return
        if label in actions:raise RuntimeError('Previous stage action was not completed; inspect before any retry: '+label)
        log=self.paths.output/'logs'/(branch+'_'+label+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        actions[label]=dict(status='starting',argv=argv,started_unix=time.time(),log=str(log));self.save()
        if self.command_runner is not None:
            code=self.command_runner(argv,log)
        else:
            with log.open('w') as stream:
                process=subprocess.Popen(argv,cwd=WORKSPACE,stdout=stream,stderr=subprocess.STDOUT)
                actions[label].update(status='running',pid=process.pid);self.save()
                try:code=process.wait(timeout=3600)
                except subprocess.TimeoutExpired:
                    # Do not silently kill/restart an unknown long analysis; preserve the handle.
                    raise RuntimeError('Stage command exceeded observation bound; process handle preserved: '+str(process.pid))
        actions[label].update(status='complete' if code==0 else 'failed',returncode=code,finished_unix=time.time());self.save()
        if code!=0:raise RuntimeError('Stage action failed, no automatic retry: '+label)
    def draft_selection(self):
        plan=self.paths.draft_plan;selection_path=plan/'selection.json';jobs_path=plan/'jobs_draft.json'
        if not selection_path.exists():
            self.run_command('draft','freeze_selection',['python3',str(PREP/'draft/prepare_replication.py'),
                '--discovery-manifest',str(self.paths.root/'draft_discovery_manifest.json'),
                '--runs',str(self.paths.root/'draft_smoke'),str(self.paths.root/'draft_discovery'),
                '--output',str(plan),'--run-output-root',str(self.paths.root/'draft_replication'),
                '--heldout-search-roots',str(self.paths.root)])
        selection=read(selection_path);jobs=read(jobs_path)
        if not selection or not jobs:raise ValueError('Frozen draft selection/jobs artifact missing; no blind prepare retry')
        if (selection.get('discovery_conditions')!=360 or
            selection.get('discovery_manifest_sha256')!=sha(self.paths.root/'draft_discovery_manifest.json') or
            jobs.get('selection_sha256')!=sha(selection_path)):
            raise ValueError('Frozen draft selection identity/evidence mismatch')
        audit_path=plan/'discovery_audit.json'
        if not audit_path.is_file() or selection.get('discovery_audit_sha256')!=sha(audit_path) or read(audit_path).get('status')!='complete':
            raise ValueError('Draft selection is not backed by a complete strict audit')
        if selection['status']=='none':
            self.state['branches']['draft'].update(phase='needs_scientific_review',reason='Frozen rule found no positive pure-draft source gap',selection=str(selection_path));return
        if selection['status']!='selected' or not jobs.get('jobs'):raise ValueError('Unexpected draft selection status')
        record=self.state['branches']['draft']
        record.update(phase='enqueue_confirmation',selection=str(selection_path),selection_sha256=sha(selection_path),
                      jobs=jobs['jobs'],expected_confirmation_episodes=jobs['expected_episodes']);self.save()
    def vlm_selection(self):
        output=self.paths.root/'vlm/analysis'
        self.run_command('vlm','freeze_selection',[str(FLASH_PYTHON),'-m',VLM_MODULE+'.analyze',
            '--root',str(self.paths.root/'vlm'),'--output',str(output),'--freeze-selection'])
        selection_path=output/'selection.json';selection=read(selection_path)
        if not selection or selection.get('complete_discovery_episodes')!=160:raise ValueError('Complete VLM selection not produced')
        status=selection['selection']['status']
        if status=='none':
            self.state['branches']['vlm'].update(phase='needs_scientific_review',reason='Frozen rule found no eligible crossing pair',selection=str(selection_path));return
        if status!='selected':raise ValueError('Unexpected VLM selection status')
        self.state['branches']['vlm'].update(phase='prepare_confirmation',selection=str(selection_path),selection_hash=selection['selection_hash'],selection_sha256=sha(selection_path));self.save()
    def vlm_prepare(self):
        record=self.state['branches']['vlm'];root=self.paths.root/'vlm_confirmation'
        self.run_command('vlm','prepare_confirmation',[str(FLASH_PYTHON),'-m',VLM_MODULE+'.prepare_confirmation',
            '--discovery-root',str(self.paths.root/'vlm'),'--selection',record['selection'],'--output',str(root)])
        manifest=read(root/'manifest.json')
        if (not manifest or manifest.get('total_confirmation_episodes')!=600 or manifest.get('cases_per_condition')!=200 or
            manifest.get('confirmation_inits')!=[31,32,33,34,35] or len(manifest.get('planned_conditions',[]))!=3):
            raise ValueError('VLM confirmation manifest does not describe frozen3×200 plan')
        jobs=[]
        for condition in manifest['planned_conditions']:
            for shard in range(4):
                job_id='vlm_confirm_%s_shard%02dof04'%(condition,shard)
                jobs.append(dict(id=job_id,argv=[str(FLASH_PYTHON),'-m',VLM_MODULE+'.confirm_runner',
                    '--output',str(root),'--condition',condition,'--stage','confirm',
                    '--shard-index',str(shard),'--shards','4','--gpu-uuid','{gpu_uuid}','--port','{port}','--launch'],
                    complete_file=str(root/condition/('confirm_shard%02dof04_complete.json'%shard)),
                    depends_on=[],ready_files=[str(root/'manifest.json'),record['selection']],
                    expected_episodes=50,scientific_phase='independent_confirmation',manifest_hash=manifest['manifest_hash']))
        write(self.paths.output/'vlm_confirmation_jobs.json',dict(jobs=jobs,episodes=600,
            shard_rule='confirm_common.cases: suite/task index modulo4; each task retains consecutive5 initial states'))
        record.update(phase='enqueue_confirmation',jobs=jobs,confirmation_root=str(root),manifest_hash=manifest['manifest_hash'],expected_confirmation_episodes=600);self.save()
    def confirmation_specs(self,name):
        record=self.state['branches'][name];specs=[]
        for job in record['jobs']:
            expected={'cases':40} if name=='draft' else {'manifest_hash':record['manifest_hash']}
            specs.append((Path(job['complete_file']),'draft' if name=='draft' else 'vlm',expected))
        return specs
    def report(self,name):
        root,reports=self.paths.root,self.paths.reports;record=self.state['branches'][name]
        if name=='draft':
            discovery=reports/'draft_discovery_analysis';confirmation=reports/'draft_confirmation_analysis';stats=reports/'draft_statistics'
            commands=[('analyze_discovery',['python3',str(PREP/'draft/analyze.py'),'--runs',str(root/'draft_smoke'),str(root/'draft_discovery'),'--output',str(discovery)]),
                ('analyze_confirmation',['python3',str(PREP/'draft/analyze.py'),'--runs',str(root/'draft_replication'),'--output',str(confirmation)]),
                ('heldout_statistics',['python3',str(PREP/'draft/heldout_statistics.py'),'--selection',record['selection'],'--analysis',str(confirmation),'--output',str(stats)]),
                ('plot_results',['python3',str(PREP/'draft/plot_results.py'),'--analysis',str(discovery),'--replication-analysis',str(confirmation),'--selection',record['selection'],'--output',str(reports/'draft_figures')])]
            for label,argv in commands:self.run_command(name,label,argv)
            audit=read(confirmation/'audit.json',{});statistics=read(stats/'statistics.json',{})
            if audit.get('complete_conditions')!=record['expected_confirmation_episodes'] or audit.get('uncompleted') or statistics.get('status')!='complete':
                raise ValueError('Draft final analysis is not complete despite completion markers')
            if audit.get('initial_pairing_mismatches')!=[]:
                raise ValueError('Draft initial-observation/noise pairing audit is missing or mismatched; report readiness withheld')
            ready=dict(analysis=str(confirmation),statistics=str(stats),figures=str(reports/'draft_figures'))
        else:
            output=reports/'vlm_confirmation'
            self.run_command(name,'analyze_confirmation',[str(FLASH_PYTHON),'-m',VLM_MODULE+'.analyze_confirmation',
                '--root',record['confirmation_root'],'--output',str(output),'--selection',record['selection'],
                '--contention-root',str(root/'resource_contention')])
            analysis=read(output/'analysis.json',{})
            if not analysis.get('complete') or analysis.get('valid_episodes')!=600:raise ValueError('VLM confirmation analysis is not complete')
            ready=dict(analysis=str(output),scientific_conclusion=analysis.get('conclusion'))
        record.update(phase='report_ready',reports=ready,ready_unix=time.time(),goal_complete_claim=False)
        write(self.paths.output/(name+'_report_ready.json'),dict(branch=name,**ready,goal_complete_claim=False));self.save()
    def advance(self,name,queue,job_state):
        record=self.state['branches'][name];phase=record['phase']
        if phase in TERMINAL:return
        if phase=='waiting_discovery':
            specs=self.discovery_specs(name);gate=self.check(specs);record['gate']=gate
            errors=[r for r in gate['markers'] if r['status'] in ('invalid','failed')]
            failures=self.relevant_failures(specs,queue,job_state)
            if errors or failures:raise RuntimeError('Discovery requires attention: '+json.dumps(dict(markers=errors,current_queue_failures=failures)))
            if not gate['complete']:return
            record['phase']='freeze_selection';self.save();phase='freeze_selection'
        if phase=='freeze_selection':
            self.draft_selection() if name=='draft' else self.vlm_selection()
        elif phase=='prepare_confirmation':self.vlm_prepare()
        elif phase=='enqueue_confirmation':
            frozen_sha=record.get('selection_sha256')
            if not frozen_sha or sha(record['selection'])!=frozen_sha:
                raise ValueError('Frozen selection SHA is missing or changed before queue append')
            jobs=[with_assigned_gpu_wait(job,self.paths.scheduler_output/'launch_gates') for job in record['jobs']]
            receipt=append_jobs(self.paths.queue,jobs);record.update(phase='waiting_confirmation',queue_append=receipt);self.save()
        elif phase=='waiting_confirmation':
            specs=self.confirmation_specs(name);gate=self.check(specs);record['gate']=gate
            errors=[r for r in gate['markers'] if r['status'] in ('invalid','failed')]
            failures=self.relevant_failures(specs,queue,job_state)
            if errors or failures:raise RuntimeError('Confirmation requires attention: '+json.dumps(dict(markers=errors,current_queue_failures=failures)))
            if gate['complete']:record['phase']='analyze_confirmation';self.save()
        elif phase=='analyze_confirmation':self.report(name)
        else:raise ValueError('Unknown pipeline phase: '+phase)
    def inspect(self):
        status=dict(mode='read_only_inspection',goal_complete_claim=False,branches={})
        for name in ('draft','vlm'):
            try:status['branches'][name]=dict(phase=self.state['branches'][name]['phase'],gate=self.check(self.discovery_specs(name)))
            except Exception as error:status['branches'][name]=dict(status='needs_attention',error=str(error))
        status['scheduler']=self.watch.observe(self.state['scheduler']);return status
    def tick(self):
        queue=read(self.paths.queue);job_state=read(self.paths.scheduler_output/'jobs_state.json',{})
        for name in ('draft','vlm'):
            try:self.advance(name,queue,job_state)
            except Exception:
                self.state['branches'][name].update(phase='needs_attention',error=traceback.format_exc(),attention_unix=time.time());self.save()
        # Reload after append; never mutate scheduler-owned jobs_state.
        queue=read(self.paths.queue);job_state=read(self.paths.scheduler_output/'jobs_state.json',{})
        if not all(r['phase'] in TERMINAL for r in self.state['branches'].values()):
            try:observation=self.watch.ensure(self.state['scheduler'],queue,job_state)
            except Exception:observation=dict(status='needs_attention',reason='scheduler_observation_or_restart_failed',error=traceback.format_exc())
            self.state['scheduler_observation']=observation
        self.save()
        phases={k:v['phase'] for k,v in self.state['branches'].items()}
        status=dict(time=time.time(),branches=phases,scheduler=self.state.get('scheduler_observation'),
            terminal=all(p in TERMINAL for p in phases.values()),goal_complete_claim=False,
            notification_policy='State transitions, report readiness, scientific review or technical attention only; unchanged waiting is silent')
        write(self.paths.output/'status.json',status)
        return status


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',default=str(WORKSPACE/'results/ease_insights_20261103'))
    parser.add_argument('--queue');parser.add_argument('--scheduler-output');parser.add_argument('--scheduler-pid',type=int)
    parser.add_argument('--output');parser.add_argument('--draft-plan');parser.add_argument('--reports')
    parser.add_argument('--poll-seconds',type=float,default=30.)
    parser.add_argument('--launch',action='store_true');parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    paths=Paths.defaults(args.root,args.queue,args.scheduler_output,args.output,args.draft_plan,args.reports)
    pipeline=Pipeline(paths,args.scheduler_pid)
    if not args.launch:
        print(json.dumps(pipeline.inspect(),ensure_ascii=False,indent=2));return
    if args.poll_seconds<3:parser.error('--poll-seconds must be >=3')
    paths.output.mkdir(parents=True,exist_ok=True)
    with file_lock(paths.output/'supervisor.lock',blocking=False):
        write(paths.output/'supervisor_process.json',dict(pid=os.getpid(),started_unix=time.time(),argv=sys.argv,
            source_sha256=sha(__file__),scheduler_only_allocates_GPUs=True))
        previous=None
        while True:
            status=pipeline.tick()
            signature=(tuple(status['branches'].items()),(status.get('scheduler') or {}).get('status'))
            if signature!=previous:print(json.dumps(status,ensure_ascii=False),flush=True);previous=signature
            if args.once or status['terminal']:break
            time.sleep(args.poll_seconds)

if __name__=='__main__':main()
