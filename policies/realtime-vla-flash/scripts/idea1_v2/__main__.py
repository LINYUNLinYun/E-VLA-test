"""idea1 v2: dry-run by default; --launch is required for any persistent action."""
import argparse
import json
import subprocess
import time
from .core import OLD, NEW, REPO, SUITES, LIMITS, WEIGHT, GPU_DISCOVERY_INTERVAL_SECONDS, output_root, sha, write_json

def manifest(source):
    import platform
    from importlib import metadata
    versions={}
    for package in ('numpy','torch','triton','mujoco','robosuite','libero'):
        try:versions[package]=metadata.version(package)
        except metadata.PackageNotFoundError:versions[package]='not_registered_in_this_interpreter'
    dirty=subprocess.run(['git','status','--short'],cwd=REPO,capture_output=True,text=True,check=True).stdout
    diff=subprocess.run(['git','diff','--stat'],cwd=REPO,capture_output=True,text=True,check=True).stdout
    return {'schema':1,'checkpoint_sha256':WEIGHT,'source':str(source.resolve()),
            'suites':list(SUITES),'tasks':list(range(10)),'inits':list(range(5)),
            'D':3,'predict':50,'execute':5,'dt':.05,'seed':7,'dummy_steps':10,
            'noise':'SeedSequence([12345, official_init_index, replan_idx])',
            'E3_gate':{'E2_valid':200,'success_min':.8,'E4_complete':True},
            'budgets':{'E2_predictions':13200,'E3_predictions':13200,'diagnostic_predictions':256,
                       'baseline_smoke_episodes':8,'invalid_retries':8,'per_key_retry':1},
            'gpu_discovery_interval_seconds':GPU_DISCOVERY_INTERVAL_SECONDS,'max_gpu_workers':4,
            'code_model':'gpt-6.1-sol','code_reasoning':'medium',
            'python':platform.python_version(),'library_versions':versions,'git_dirty_status':dirty,'git_dirty_diff_stat':diff,
            'code_hashes':{str(p.relative_to(REPO)):sha(p) for p in sorted((REPO/'scripts/idea1_v2').glob('*.py'))}}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('preflight','freeze','e4','serve','episode','parity','worker','smoke','launch','resume','reconcile','analyze','render','audit','gpu-discover'))
    parser.add_argument('--source',type=__import__('pathlib').Path,default=OLD)
    parser.add_argument('--output',type=__import__('pathlib').Path,default=NEW/'UNLAUNCHED')
    parser.add_argument('--launch',action='store_true',help='Execute this command. Default prints a read-only dry-run.')
    parser.add_argument('--suite',choices=SUITES,default=SUITES[0])
    parser.add_argument('--task',type=int,choices=range(10),default=0)
    parser.add_argument('--init',type=int,choices=range(5),default=0)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--mode',choices=('alternating','phase','full','v14'),default='alternating')
    parser.add_argument('--condition',default='ALT_V14x2_FULLx1_D3_R5')
    parser.add_argument('--episode-attempt',default='attempt00')
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--retry-invalid',action='store_true')
    parser.add_argument('--fix-note',default='',help='Required diagnosis/fix description before one invalid retry')
    args=parser.parse_args()
    args.output=output_root(args.output)
    preview={'command':args.command,'dry_run':not args.launch,'output':str(args.output),'source':str(args.source.resolve()),
             'episodes_per_condition':200,'suite_order':list(SUITES),'task_init_order':'task0..9 then init0..4',
             'max_actions_per_suite':{s:LIMITS[s]*50 for s in SUITES},'new_e4_model_calls':0,
             'required_launch_gates':['freeze/source hashes','single-card parity, baseline reproduction, E2 smoke',
                                      'global campaign ledger and GPU leases','E3: E2>=80%, 200 valid and E4 complete']}
    if not args.launch:
        print(json.dumps(preview,indent=2));return
    if args.command in ('preflight','freeze','e4'):
        from .data import load
        groups,sources=load(args.source.resolve())
        if args.command=='preflight':
            summary={k:v for k,v in sources.items() if k!='files'}
            summary['hashed_source_files']=len(sources['files'])
            print(json.dumps(summary,indent=2));return
        if args.command=='freeze':
            if args.output.exists():
                raise FileExistsError('Freeze requires a new attempt')
            from scripts.experiment10.common import CONVERTED
            actual=sha(CONVERTED/'base/base_weights.pkl')
            if actual!=WEIGHT:raise RuntimeError('Checkpoint hash differs from frozen protocol')
            sources['files'][str(CONVERTED/'base/base_weights.pkl')]=actual
            write_json(args.output/'protocol/execution_manifest.json',manifest(args.source),exclusive=True)
            write_json(args.output/'protocol/source_manifest.json',sources,exclusive=True)
            print(json.dumps({'frozen':str(args.output),'counts':sources['split']}));return
        frozen=json.loads((args.output/'protocol/source_manifest.json').read_text())
        from scripts.experiment10.common import CONVERTED
        sources['files'][str(CONVERTED/'base/base_weights.pkl')]=sha(CONVERTED/'base/base_weights.pkl')
        if frozen!=sources:
            raise RuntimeError('Source provenance changed after freeze')
        from .kalman import run
        result=run(groups,args.output)
        write_json(args.output/'kalman/completion.json',result,exclusive=True)
        print(json.dumps(result));return
    if args.command in ('serve','episode','worker'):
        if __import__('os').environ.get('IDEA1_V2_ORCHESTRATED')!='1':
            raise SystemExit('Internal entrypoint: use budgeted smoke/launch/resume rather than starting an unaccounted worker.')
        if args.command=='serve':
            from .policy import serve
            serve(args)
        elif args.command=='episode':
            from .client import run
            run(args)
        else:
            from .runner import worker
            worker(args)
        return
    if args.command in ('smoke','launch','resume','parity','gpu-discover','reconcile'):
        from . import runner
        if args.command=='parity' and __import__('os').environ.get('IDEA1_V2_ORCHESTRATED')!='1':
            raise SystemExit('Internal parity entrypoint: use single-card smoke to acquire an idle GPU lease and account diagnostics')
        if args.command=='gpu-discover':
            print(json.dumps(runner.discover()));return
        if args.command=='reconcile':
            print(json.dumps({'invalidated_dead_reservations':runner.Ledger().reconcile(args.fix_note)}));return
        if not (args.output/'protocol/execution_manifest.json').exists():raise SystemExit('Freeze required before GPU launch')
        frozen=json.loads((args.output/'protocol/execution_manifest.json').read_text())
        if frozen.get('gpu_launch_eligible') is False:raise SystemExit('CPU validation attempt is ineligible for GPU/episode launch; freeze a separate experiment attempt')
        if frozen['code_hashes']!=manifest(args.source)['code_hashes']:raise SystemExit('Code differs from frozen manifest')
        if args.command in ('smoke','launch','resume','parity'):
            sources=json.loads((args.output/'protocol/source_manifest.json').read_text())
            for path,digest in sources['files'].items():
                if sha(path)!=digest:raise SystemExit('Source changed after freeze: '+path)
        if args.command=='parity':runner.parity(args)
        elif args.command=='smoke':runner.smoke(args)
        else:runner.launch(args)
        return
    if args.command=='analyze':
        from .analysis import run
        print(json.dumps(run(args.output,args.source),indent=2));return
    from .media import run
    print(json.dumps(run(args.output,args.command,args.source),indent=2))

if __name__=='__main__':
    main()
