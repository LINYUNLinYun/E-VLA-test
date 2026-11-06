"""Revision-pinned LIBERO-Plus stratification and isolated native environment.

Never installs/uninstalls libero and never reads or writes ~/.libero/config.yaml.
Each runtime process selects exactly one repository/config before importing it.
"""
import argparse
import ast
from collections import Counter,defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from .native_envs import ROOT,sha,digest,write_json

PLUS=Path('/policies/LIBERO-plus')
PLUS_COMMIT='4976dc30028e805ff8094b55501d532c48fec182'
ORIGINAL=ROOT/'LIBERO'
SUITES=('libero_spatial','libero_object','libero_goal','libero_10')
CATEGORIES=('Objects Layout','Camera Viewpoints','Robot Initial States','Language Instructions',
            'Light Conditions','Background Textures','Sensor Noise')
CAPS={suite:600 for suite in SUITES}
SETTLE_STEPS=5
HEADS=dict(zip(SUITES,('draft_spatial','draft_object','draft_goal','draft_10')))


def task_map(repository):
    path=Path(repository)/'libero/libero/benchmark/libero_suite_task_map.py'
    tree=ast.parse(path.read_text())
    return ast.literal_eval(next(node.value for node in tree.body if isinstance(node,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='libero_task_map' for t in node.targets)))


def audit_and_select(output,per_cell=10):
    """Actual category×difficulty instances, balanced mother-suite/base-task selection."""
    path=PLUS/'libero/libero/benchmark/task_classification.json'
    classified=json.loads(path.read_text());registered=task_map(PLUS);original=task_map(ORIGINAL)
    cells=defaultdict(list);unknown=[];audited=[]
    for suite in SUITES:
        for item in classified[suite]:
            row=dict(item,mother_suite=suite,task_index=int(item['id'])-1)
            if registered[suite][row['task_index']]!=row['name']:
                raise ValueError('Classification1-based ID/nativeindex mismatch')
            candidates=[name for name in original[suite] if row['name'].startswith(name)]
            if not candidates: raise ValueError('Perturbed instance has no unique original mother task')
            mother=max(candidates,key=len)
            row.update(mother_task=mother,matching_draft=HEADS[suite],native_max_steps=CAPS[suite],trial=0,
                environment='libero_plus',seed=int(hashlib.sha256(('plus|'+suite+'|'+row['name']).encode()).hexdigest()[:8],16))
            audited.append(row)
            if item['difficulty_level'] is None: unknown.append(row)
            else: cells[(row['category'],int(row['difficulty_level']))].append(row)
    selected=[];strata=[]
    for ci,category in enumerate(CATEGORIES):
        for difficulty in range(1,6):
            pool=cells[(category,difficulty)]
            # Rotate extra suite allocations between strata (10 instances means
            # counts3/3/2/2); rank by deterministic hash, then diversify bases.
            by_suite={suite:defaultdict(list) for suite in SUITES}
            for row in pool: by_suite[row['mother_suite']][row['mother_task']].append(row)
            for groups in by_suite.values():
                for rows in groups.values(): rows.sort(key=lambda r:digest(['plus-v1',r['mother_suite'],r['name']]))
            rotation=(ci*5+difficulty-1)%len(SUITES)
            suite_order=SUITES[rotation:]+SUITES[:rotation]
            chosen=[];used=set();base_cursors=Counter()
            while len(chosen)<min(int(per_cell),len(pool)):
                progress=False
                for suite in suite_order:
                    groups=by_suite[suite]
                    bases=sorted(groups,key=lambda name:digest(['plus-base-v1',category,difficulty,suite,name]))
                    if not bases: continue
                    for _ in range(len(bases)):
                        base=bases[base_cursors[suite]%len(bases)];base_cursors[suite]+=1
                        choices=[r for r in groups[base] if r['name'] not in used]
                        if choices:
                            chosen.append(choices[0]);used.add(choices[0]['name']);progress=True;break
                    if len(chosen)>=min(int(per_cell),len(pool)): break
                if not progress: break
            selected.extend(chosen)
            strata.append(dict(category=category,difficulty=difficulty,available=len(pool),selected=len(chosen),
                mother_suite_counts=dict(Counter(r['mother_suite'] for r in chosen)),
                distinct_mother_tasks=len(set((r['mother_suite'],r['mother_task']) for r in chosen))))
    baselines=[]
    for suite in SUITES:
        for index,name in enumerate(original[suite]):
            baselines.append(dict(environment='libero_original',mother_suite=suite,mother_task=name,name=name,
                task_index=index,category='Original unperturbed sanity',difficulty_level=None,
                matching_draft=HEADS[suite],trial=0,native_max_steps=CAPS[suite],seed=int(hashlib.sha256(('original|'+suite+'|'+name).encode()).hexdigest()[:8],16)))
    result=dict(status='complete',plus_repository=str(PLUS),plus_commit=PLUS_COMMIT,
        classification_sha256=sha(path),registered_task_map_sha256=sha(PLUS/'libero/libero/benchmark/libero_suite_task_map.py'),
        total_classified=len(audited),category_difficulty_strata=strata,unknown_difficulty_count=len(unknown),
        unknown_difficulty_by_suite=dict(Counter(r['mother_suite'] for r in unknown)),
        unknown_difficulty_policy='Preserved and reported separately; never imputed into1..5 or used to pretend complete graded coverage.',
        unknown_difficulty_instances=unknown,selection_rule='Deterministic hash-ranked base-task-diversified round robin among mother suites in each actual category×level; frozen before quality inference',
        per_stratum_requested=int(per_cell),selected=selected,selected_count=len(selected),
        selected_mother_suite_counts=dict(Counter(r['mother_suite'] for r in selected)),
        selected_distinct_mother_tasks=len(set((r['mother_suite'],r['mother_task']) for r in selected)),
        original_baselines=baselines,original_baseline_count=len(baselines),
        conditions_per_instance='Full, V14, exact matching mother-suite draft only',
        episode_protocol=dict(max_steps=600,settle_steps=5,settle_action=[0.]*7,hard_reset=True,
            reference='Local official Plus libero/configs/eval/default.yaml max_steps600 and libero/lifelong/metric.py five zero dummy actions',
            reference_sha256={str(p):sha(p) for p in (PLUS/'libero/configs/eval/default.yaml',PLUS/'libero/lifelong/metric.py')},
            scope='Preregistered stratified subset with this repository reference cadence; not a claim of reproducing every external pi0 paper evaluator/leaderboard setting'),
        official_trials_per_instance=1,selection_sha256=digest(selected),baseline_sha256=digest(baselines))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write_json(output/'selection.json',result)
    return result


def isolated_configuration(repository,directory):
    import yaml
    repository=Path(repository).resolve();directory=Path(directory).resolve()
    if directory==Path.home()/'.libero' or directory==Path.home(): raise ValueError('Global LIBERO config is forbidden')
    directory.mkdir(parents=True,exist_ok=True)
    root=repository/'libero/libero'
    values=dict(benchmark_root=str(root),bddl_files=str(root/'bddl_files'),init_states=str(root/'init_files'),
                datasets=str(repository/'libero/datasets'),assets=str(root/'assets'))
    path=directory/'config.yaml'
    if path.exists():
        if yaml.safe_load(path.read_text())!=values: raise ValueError('Existing isolated config belongs to another repository')
    else: path.write_text(yaml.safe_dump(values,sort_keys=True))
    return path


def isolated_child_environment(case,config_directory,gpu=None,gpu_uuid=None):
    repository=PLUS if case['environment']=='libero_plus' else ORIGINAL
    isolated_configuration(repository,config_directory)
    env=dict(os.environ)
    # The selected repo precedes all editable installs, and each child later
    # asserts actual imported package paths. No global installation is changed.
    env['PYTHONPATH']=str(ROOT)+os.pathsep+str(repository)
    env['LIBERO_CONFIG_PATH']=str(Path(config_directory).resolve())
    env['OMP_NUM_THREADS']='1';env['PYTHONUNBUFFERED']='1'
    # Official revision-pinned init_files contain numpy pickles, not weights.
    # Scope this compatibility setting to our isolated child, never old envs.
    env['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    runtime=PLUS/'.runtime'
    if repository==PLUS and runtime.exists():
        env.update(MAGICK_HOME=str(runtime/'usr'),LD_LIBRARY_PATH=str(runtime/'usr/lib/x86_64-linux-gnu'),
            MAGICK_CONFIGURE_PATH=str(runtime/'etc/ImageMagick-6'),
            MAGICK_CODER_MODULE_PATH=str(runtime/'usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders'))
    if gpu is not None:
        if not gpu_uuid: raise ValueError('Physical GPU UUID required for rendering')
        env['CUDA_VISIBLE_DEVICES']=gpu_uuid;env['MUJOCO_GL']='egl';env['MUJOCO_EGL_DEVICE_ID']=str(gpu)
    return env


class NativeLiberoPlusEnv:
    def __init__(self,case,render=False,gpu=None,gpu_uuid=None):
        repository=PLUS if case['environment']=='libero_plus' else ORIGINAL
        if not os.environ.get('LIBERO_CONFIG_PATH') or Path(os.environ['LIBERO_CONFIG_PATH']).resolve()==Path.home()/'.libero':
            raise RuntimeError('Isolated LIBERO_CONFIG_PATH must precede import')
        import libero.libero as package
        if repository.resolve() not in Path(package.__file__).resolve().parents:
            raise RuntimeError('Wrong LIBERO import: '+str(package.__file__))
        from libero.libero import benchmark,get_libero_path
        from libero.libero.envs.env_wrapper import ControlEnv
        if render:
            if gpu is None or not gpu_uuid: raise ValueError('Explicit GPU binding required for native offscreen rendering')
            from preparations.idea1_v3_2.transfer_envs import gpu_renderer_binding
            binding=gpu_renderer_binding(gpu,gpu_uuid,require_idle=False)
        else: binding=dict(renderer_constructed=False,gpu_used=False)
        selected=benchmark.get_benchmark_dict()[case['mother_suite']]()
        names=selected.get_task_names()
        if case['name'] not in names: raise ValueError('Frozen task name not registered in selected repository')
        index=names.index(case['name']);definition=selected.get_task(index)
        if case['environment']=='libero_plus' and index!=case['task_index']:
            raise ValueError('Frozen Plus classification ID/runtimeindex differs')
        self.case,self.selected,self.definition,self.index=case,selected,definition,index
        self.render=render
        bddl=Path(get_libero_path('bddl_files'))/definition.problem_folder/definition.bddl_file
        self.env=ControlEnv(str(bddl),use_camera_obs=render,has_renderer=False,has_offscreen_renderer=render,
            render_gpu_device_id=int(gpu) if render else -1,camera_heights=256,camera_widths=256,hard_reset=True)
        actual_instruction=str(self.env.language_instruction)
        self.info=dict(environment=case['environment'],repository=str(repository),actual_import=str(package.__file__),
            isolated_config=os.environ['LIBERO_CONFIG_PATH'],runtime_task_index=index,task_name=case['name'],
            instruction=actual_instruction,benchmark_definition_language=str(definition.language),
            benchmark_language_differs_from_parsed_bddl=actual_instruction!=str(definition.language),
            instruction_source='actual effective BDDL :language (rewritten language-perturbation BDDL kept); filename camera/noise/robot tags never sent as prompt',
            native_hz=float(self.env.env.control_freq),H=50,R=5,max_steps=600,settle_steps=SETTLE_STEPS,
            category=case['category'],difficulty=case['difficulty_level'],matching_draft=case['matching_draft'],
            binding=binding,physical_bddl_requested=str(bddl),
            noise_pipeline='Official Plus ControlEnv wrapper, including actual image corruption during rendered steps' if render else 'Camera-free physical probe only; no image corruption claim')
        controller=self.env.robots[0].controller
        self.info['controller']={k:np.asarray(getattr(controller,k)).tolist() for k in ('input_min','input_max','output_min','output_max')}

    def reset(self):
        self.env.seed(int(self.case['seed'])%(2**31-1));self.env.reset()
        states=self.selected.get_task_init_states(self.index)
        vector=np.asarray(states[0])
        self.obs=self.env.set_init_state(vector.copy())
        # Camera-free probes exercise original simulator physical integration,
        # not wrapper image operators requiring absent camera observations.
        for _ in range(SETTLE_STEPS):
            self.obs,_,_,_=self.env.step([0.]*7) if self.render else self.env.env.step([0.]*7)
        self.info['initial_vector_sha256']=hashlib.sha256(vector.tobytes()).hexdigest()
        return self.obs

    def step_native(self,action):
        action=np.asarray(action,dtype=np.float64)
        if action.shape!=(7,) or not np.isfinite(action).all(): raise ValueError('LIBERO native finite7D action required')
        clipped=np.clip(action,-1.,1.)
        self.obs,reward,done,info=self.env.step(clipped.tolist()) if self.render else self.env.env.step(clipped.tolist())
        success=bool(self.env.check_success())
        return self.obs,dict(success=success,native_done=bool(done),reward=float(reward),
            raw_action=action.tolist(),native_action=clipped.tolist(),clipped_components=np.flatnonzero(action!=clipped).tolist())

    def actual(self):
        return dict(eef_xyz_world=np.asarray(self.obs['robot0_eef_pos']).tolist(),
            eef_quaternion_xyzw=np.asarray(self.obs['robot0_eef_quat']).tolist(),
            gripper_qpos=np.asarray(self.obs['robot0_gripper_qpos']).tolist())

    def rgb(self):
        if not self.render: raise RuntimeError('Camera-free probe has no RGB')
        return np.ascontiguousarray(self.obs['agentview_image'][::-1,::-1])
    def close(self): self.env.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('select','import-preflight','native-preflight'),required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--selection')
    parser.add_argument('--case',type=int,default=0)
    parser.add_argument('--per-cell',type=int,default=10)
    parser.add_argument('--launch',action='store_true')
    args=parser.parse_args()
    if not args.launch: print(json.dumps(dict(dry_run=True,**vars(args))));return
    if args.stage=='select':
        result=audit_and_select(args.output,args.per_cell)
        print(json.dumps({k:result[k] for k in ('status','selected_count','original_baseline_count','unknown_difficulty_count','selection_sha256')}));return
    if args.stage=='import-preflight':
        import libero.libero as package
        from libero.libero import benchmark
        from libero.libero.envs.env_wrapper import ControlEnv
        if PLUS not in Path(package.__file__).resolve().parents: raise RuntimeError('Wrong package imported')
        result=dict(status='complete',passed=True,actual_import=str(package.__file__),
            isolated_config=os.environ.get('LIBERO_CONFIG_PATH'),global_libero_config_used=False,
            suite_counts={suite:len(benchmark.task_maps[suite]) for suite in SUITES},
            native_wrapper_source=ControlEnv.__module__,renderer_constructed=False,gpu_used=False)
        write_json(Path(args.output)/'receipt.json',result);print(json.dumps(result));return
    if not args.selection: parser.error('--selection required')
    case=json.loads(Path(args.selection).read_text())['selected'][args.case]
    env=None;output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    try:
        env=NativeLiberoPlusEnv(case,render=False);env.reset()
        _,step=env.step_native(np.zeros(7))
        receipt=dict(status='complete',passed=True,scope='Native camera-free physical environment/init/controller/oracle probe; renderer/noise pipeline not exercised',
            case=case,native_info=env.info,actual=env.actual(),noop_step=step,model_used=False,gpu_used=False)
    except Exception:
        import traceback
        receipt=dict(status='technical_failure',passed=False,case=case,error=traceback.format_exc(),model_used=False,gpu_used=False)
    finally:
        if env is not None: env.close()
    write_json(output/'receipt.json',receipt)
    print(json.dumps({k:receipt[k] for k in ('status','passed')}))


if __name__=='__main__': main()
