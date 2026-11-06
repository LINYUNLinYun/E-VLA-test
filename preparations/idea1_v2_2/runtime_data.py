"""Freeze verified native protocols, official initial states and real observations."""
import ast, json, re, shutil
from pathlib import Path
import numpy as np
from common import ROOT, MODELS, SUITES, conditions, digest, sha, atomic_json, seed, bootstrap_environment,RUNTIME_FILES,UPSTREAM_FILES

def expanded_schedule(model,schedule):
    from schedules import flow_schedule,ddim_indices
    if model!='db-cogact':
        out=flow_schedule(model,schedule['m'],schedule['num_steps'])
        out.update(update_formula='x_next=x+dt*v(x,t); second evaluation uses first updated x',model_time_coordinate='t=q, q:1→0' if model=='pi0' else 't=1-q, q:1→0',evidence='CPU expected schedule; actual model timesteps/NFE separately gated')
        return out
    import importlib.util
    directory=ROOT/'policies/db-cogact/dexbotic/model/cogact/action_model'; source=directory/'action_models.py'
    tree=ast.parse(source.read_text()); constructor=next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='ActionModel' for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    defaults=dict(zip([a.arg for a in constructor.args.args][-len(constructor.args.defaults):],[ast.literal_eval(x) for x in constructor.args.defaults]))
    spec=importlib.util.spec_from_file_location('v22_cpu_native_diffusion',directory/'diffusion.py'); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    kwargs=dict(noise_schedule=defaults['noise_schedule'],diffusion_steps=defaults['diffusion_steps'],sigma_small=True,learn_sigma=False)
    native=module.create_diffusion('ddim10',**kwargs)
    if schedule['kind']=='native': sampler=module.create_diffusion('ddim'+str(schedule['num_steps']),**kwargs)
    else:
        base=module.create_diffusion('',**kwargs); sampler=module.SpacedDiffusion(use_timesteps=ddim_indices(max(native.timestep_map),schedule['m']),betas=base.betas,model_mean_type=native.model_mean_type,model_var_type=native.model_var_type,loss_type=native.loss_type)
    alpha=sampler.alphas_cumprod[::-1]
    return dict(timesteps=list(reversed(sampler.timestep_map)),nodes=list(reversed(sampler.timestep_map))+['clean'],timestep_map=list(sampler.timestep_map),alpha_cumprod=alpha.tolist(),sigma=np.sqrt(1-alpha).tolist(),SNR=(alpha/(1-alpha)).tolist(),alpha_clean=1.,eta=0.,CFG=1.5,prediction_type=str(sampler.model_mean_type),variance_type=str(sampler.model_var_type),training_defaults=defaults,NFE=len(sampler.timestep_map),update_formula='native DDIM eta=0 using checkpoint beta/alpha; final alpha_prev=1 is clean endpoint, not training timestep zero',evidence='CPU native sampler construction; actual denoiser mapped calls separately gated',source=str(source),source_sha256=sha(source))

def load_observation(point):
    p=Path(point['input_path'])
    if sha(p)!=point['sha256']: raise ValueError('Frozen observation changed')
    with np.load(p,allow_pickle=False) as z:
        instruction=str(z['instruction'].item()) if 'instruction' in z else str(z['prompt'].item()) if 'prompt' in z else point['instruction']
        obs={k:z[k] for k in z.files if k not in ('instruction','seed','noise','metadata_json','prompt')}
        if 'agent_image' in z:
            # These old Full observations are already native model inputs;
            # the pi0 adapter recognizes them and avoids rotating them twice.
            obs=dict(agent_image=z['agent_image'],wrist_image=z['wrist_image'],robot_state=z['robot_state'])
    return obs,instruction,point['seed']

def observation_provenance(point):
    """Derive Full identity and phase from the original episode, never caller labels."""
    from runtime_client import legacy
    point=dict(point); p=Path(point['input_path']); evidence={}
    with np.load(p,allow_pickle=False) as z:
        metadata=json.loads(str(z['metadata_json'].item())) if 'metadata_json' in z else None
    if metadata is not None:
        if metadata.get('skip_layers')!=[] or not metadata.get('condition','').startswith('Full18'): raise ValueError('Offline source is not a real Full observation')
        suite=metadata['suite']; task=metadata['task_id']; init=metadata['episode_idx']; infer_id=metadata['infer_id']
        source_root=next(x.parent for x in p.parents if x.name=='full_observations')
        matches=list((source_root/'closed_loop'/metadata['condition']/suite/'run/episodes').glob(f'task{task:02d}_*/ep{init:03d}_*'))
        if len(matches)!=1: raise ValueError('Full observation original episode is ambiguous/missing')
        episode=matches[0]; inference=[json.loads(x) for x in (episode/'infer.jsonl').read_text().splitlines()]
        record=next(x for x in inference if x['infer_id']==infer_id)
        if record.get('route_type')!='full' or record['task_suite_name']!=suite or record['task_id']!=task or record['episode_idx']!=init: raise ValueError('Original Full inference identity mismatch')
        import hashlib
        with np.load(p,allow_pickle=False) as z:
            h=hashlib.sha256()
            for key in ('agent_image','wrist_image','robot_state','noise'): h.update(np.ascontiguousarray(z[key]).tobytes())
            h.update(str(z['prompt'].item()).encode()); h.update(suite.encode()); noise_hash=hashlib.sha256(np.ascontiguousarray(z['noise']).tobytes()).hexdigest()
        saved=record.get('experiment11',{})
        if h.hexdigest()!=metadata['input_sha256'] or saved.get('full_input_sha256')!=h.hexdigest() or noise_hash!=metadata['noise_sha256'] or saved.get('noise_sha256')!=noise_hash: raise ValueError('pi0 source arrays/noise detached from original Full input identity')
        step=record['frame_idx']; point['source_metadata']=metadata; point['source_env_step']=record['env_step']
    else:
        episode=Path(point.get('source_episode',p.parent.parent)); episode=episode if episode.is_absolute() else ROOT/episode
        identity=json.loads((episode/'identity.json').read_text())
        if identity['skip']!=[]: raise ValueError('Offline source episode is not Full')
        suite=identity['suite']; task=identity['task']; init=identity['init']
        inference=[json.loads(x) for x in (episode/'infer.jsonl').read_text().splitlines()]
        record=next(x for x in inference if Path(x['input_path']).resolve()==p.resolve())
        if record['input_file_sha256']!=sha(p): raise ValueError('Original Full input provenance mismatch')
        step=record['step']; evidence[str(episode/'identity.json')]=sha(episode/'identity.json')
    telemetry=[json.loads(x) for x in (episode/'telemetry.jsonl').read_text().splitlines()]
    from runtime_client import phase_evidence
    phase=next((phase_evidence(x).get('semantic_phase','unknown') for x in telemetry if x['step']==step),'unknown')
    if point.get('suite')!=suite: raise ValueError('Supplied observation suite differs from true source')
    for name in ('infer.jsonl','telemetry.jsonl'): evidence[str(episode/name)]=sha(episode/name)
    point.update(suite=suite,source_episode=str(episode.resolve()),source_step=step,source_task=task,source_init=init,phase=phase,source_evidence=evidence)
    obs,instruction,_=load_observation(point); point['content_hash']=legacy.input_digest(obs,instruction)
    return point

def native_limits():
    path=ROOT/'policies/realtime-vla-flash/examples/libero/main.py'
    tree=ast.parse(path.read_text()); limits={}
    for node in ast.walk(tree):
        if isinstance(node,ast.If) and isinstance(node.test,ast.Compare):
            strings=[n.value for n in ast.walk(node.test) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
            for suite in strings:
                if suite in SUITES:
                    for s in node.body:
                        if isinstance(s,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='max_steps' for t in s.targets): limits[suite]=ast.literal_eval(s.value)
    if set(limits)!=set(SUITES): raise ValueError('Cannot parse official native action limits')
    return dict(limits=limits,source=str(path),sha256=sha(path))

def inherited_diagnostics(config):
    """Freeze immutable prior receipts; reserve failed/incomplete calls too."""
    result={}
    for model in MODELS:
        sources={}
        def add(path,predictions=0,conservative=None):
            path=Path(path).resolve()
            record=dict(path=str(path),sha256=sha(path),predictions=int(predictions),
                        conservative_predictions=int(predictions if conservative is None else conservative))
            if str(path) in sources:
                old=sources[str(path)]
                if old['sha256']!=record['sha256']: raise ValueError('Prior diagnostic receipt changed')
                record['predictions']=max(old['predictions'],record['predictions'])
                record['conservative_predictions']=max(old['conservative_predictions'],record['conservative_predictions'])
            sources[str(path)]=record
        for name in ('prediction_ledger.jsonl','native_reference_ledger.jsonl'):
            path=Path(__file__).parent/'readiness'/f'model_smoke_{model}'/name
            if path.exists(): add(path,len(path.read_text().splitlines()))
        for entry in config.get('prior_attempts',[]):
            root=Path(entry if isinstance(entry,str) else entry['path']).resolve()
            manifest_path=root/'condition_manifest.json'; old=json.loads(manifest_path.read_text())
            if digest({k:v for k,v in old.items() if k!='manifest_hash'})!=old['manifest_hash']:
                raise ValueError('Prior attempt manifest identity changed')
            if isinstance(entry,dict) and entry.get('manifest_sha256',sha(manifest_path))!=sha(manifest_path):
                raise ValueError('Explicit prior attempt manifest hash differs')
            add(manifest_path)
            for source in old.get('prior_diagnostics',{}).get(model,{}).get('sources',[]):
                if sha(source['path'])!=source['sha256']: raise ValueError('Inherited prior receipt changed')
                add(source['path'],source.get('predictions',0),source.get('conservative_predictions',source.get('predictions',0)))
            parity_path=root/'parity'/f'{model}.json'
            if not parity_path.exists(): continue
            parity=json.loads(parity_path.read_text())
            actual=0
            for name in ('prediction_ledger.jsonl','native_reference_ledger.jsonl'):
                path=root/'native'/model/name
                if path.exists():
                    actual+=len(path.read_text().splitlines()); add(path)
            check_path=root/'native'/model/'parity_checks.jsonl'
            if check_path.exists():
                checks=[json.loads(x) for x in check_path.read_text().splitlines()]
                actual=max(actual,sum(x.get('actual_predictions',0) for x in checks)); add(check_path)
            actual=max(actual,parity.get('actual_predictions',0))
            reserved=max(actual,parity.get('reserved_calls',0))
            budget_path=root/'readiness'/model/'diagnostic_budget.json'
            if budget_path.exists():
                budget=json.loads(budget_path.read_text()); add(budget_path)
                actual+=budget.get('smoke_actual',0); reserved+=max(budget.get('smoke_reserved',0),budget.get('smoke_actual',0))
            add(parity_path,actual,reserved)
        rows=list(sources.values())
        actual=sum(x['predictions'] for x in rows); conservative=sum(x['conservative_predictions'] for x in rows)
        if not 0<=actual<=conservative<=256: raise ValueError('Inherited model diagnostic budget exceeds 256')
        result[model]=dict(actual_predictions=actual,conservative_predictions=conservative,sources=rows)
    return result

def freeze(root,config,extension=False):
    bootstrap_environment()
    from runtime_client import legacy
    from libero.libero import benchmark,get_libero_path
    from provenance import collect_checkpoint_assets,collect_scientific_sources
    root=Path(root)
    if (root/'condition_manifest.json').exists(): raise FileExistsError('Attempt already frozen; resume it or create a new attempt')
    protocols={}; points={}; official=[]
    for model in MODELS:
        cfg=config['models'][model]; source=Path(cfg['protocol_source']); protocol=json.loads(source.read_text())
        for evidence in protocol.get('source_evidence',[]):
            if sha(evidence['path'])!=evidence['sha256']: raise ValueError('Curated historical protocol evidence changed')
        if model!='pi0':
            verification_path=ROOT/f'preparations/idea1_v2_1/metadata/{model}/weights_verified.json'; verification=json.loads(verification_path.read_text())
            if verification.get('ok') is not True: raise ValueError('Historical checkpoint verification failed')
            checkpoint=Path(verification['checkpoint']); shards=[]; assets={}
            for record in verification['shards']:
                path=Path(record['path']); actual=sha(path)
                if path.stat().st_size!=record['expected_bytes'] or actual!=record['actual_sha256']: raise ValueError('Actual checkpoint shard differs from verified source')
                shards.append(dict(path=str(path),sha256=actual,bytes=path.stat().st_size))
            for name in ('config.json','model.safetensors.index.json','processor_config.json','preprocessor_config.json','norm_stats.json'):
                path=checkpoint/name
                if path.exists(): assets[name]=dict(path=str(path),sha256=sha(path))
            protocol.update(checkpoint=str(checkpoint),checkpoint_revision=verification['revision'],checkpoint_sha256=legacy.value_digest(shards),checkpoint_shards=shards,checkpoint_assets=assets,checkpoint_verification_source=str(verification_path),checkpoint_verification_source_sha256=sha(verification_path))
        for name in ('H','R','dtype','attention','cameras','state_used','image_rotation','skip_semantics','checkpoint','checkpoint_sha256'):
            if name not in protocol or protocol[name] is None: raise ValueError(f'{model} native protocol missing {name}')
        if model=='pi0' and sha(protocol['checkpoint'])!=protocol['checkpoint_sha256']: raise ValueError('Historical pi0 checkpoint changed')
        protocol.update(checkpoint_assets=collect_checkpoint_assets(model,protocol['checkpoint']),scientific_code_hashes=collect_scientific_sources(model),v22_protocol_source=str(source.resolve()),v22_protocol_source_sha256=sha(source),seed_rule='SHA256 canonical JSON [idea1-v22,1701,suite,task,init,replan], first 8 hex digits',native_action_limits=native_limits())
        protocol['execution']=dict(control_frequency_hz=20,warmup_noop_steps=10,success_check='after every actual env.step',gripper=protocol.get('gripper','native continuous; no extra threshold'),native_limits=native_limits())
        protocol['expanded_schedules']={json.dumps(c['schedule'],sort_keys=True):expanded_schedule(model,c['schedule']) for c in conditions(extension) if c['model']==model}
        protocols[model]=protocol; atomic_json(root/'protocols'/f'{model}.json',protocol)
        candidates=cfg.get('observations')
        if candidates is None:
            probes=json.loads(Path(cfg['probe_source']).read_text()); candidates=[]
            for p in probes['points']:
                s=p['source']; ep=Path(s['episode']); suite=next((x for x in SUITES if x in ep.parts),None)
                candidates.append(dict(input_path=s['input_path'],suite=suite,phase=s.get('phase','unknown'),source_episode=str(ep)))
        groups={}
        for x in candidates:
            if x.get('suite') not in SUITES: continue
            groups.setdefault((x['suite'],x.get('phase','unknown')),[]).append(x)
        selected=[]; seen=set(); content_seen=set()
        while len(selected)<32 and any(groups.values()):
            for key in sorted(groups):
                if not groups[key]: continue
                x=dict(groups[key].pop(0)); path=str(Path(x['input_path']).resolve())
                if path in seen: continue
                seen.add(path); x.update(input_path=path,sha256=sha(path),seed=seed(x['suite'],0,5,len(selected)),index=len(selected)); x=observation_provenance(x)
                if x['content_hash'] in content_seen: continue
                content_seen.add(x['content_hash']); selected.append(x)
                if len(selected)==32: break
        if len(selected)!=32 or set(x['suite'] for x in selected)!=set(SUITES): raise ValueError(f'{model}: require 32 distinct real observations covering four suites')
        if len({x['phase'] for x in selected if x['phase']!='unknown'})<2: raise ValueError('Frozen observations must cover multiple evidenced operation phases')
        points[model]=selected
    for suite in SUITES:
        selected=benchmark.get_benchmark_dict()[suite]()
        for task in range(10):
            states=legacy.official_init_states(selected,task)
            if len(states)<11: raise ValueError(f'Missing init10: {suite}/{task}')
            definition=selected.get_task(task)
            p=Path(get_libero_path('init_states'))/definition.problem_folder/definition.init_states_file
            official.append(dict(suite=suite,task=task,path=str(p),sha256=sha(p),indices={str(i):sha_array(states[i]) for i in range(6,11)}))
    prior=inherited_diagnostics(config)
    manifest=dict(version='2.2',conditions=conditions(extension),protocol_hashes={m:digest(p) for m,p in protocols.items()},official_initial_states=official,native_action_limits=native_limits(),observations=points,prior_diagnostics=prior,config=config,code_hashes={name:sha(Path(__file__).parent/name) for name in RUNTIME_FILES},upstream_code_hashes={name:sha(ROOT/name) for name in UPSTREAM_FILES},xiaomi_alias_requires_parity=True)
    manifest['manifest_hash']=digest(manifest); atomic_json(root/'condition_manifest.json',manifest); return manifest
def sha_array(x):
    import hashlib
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()

def verify_frozen(root,runtime=None,model=None):
    """Reject source/protocol drift before executing scientific predictions."""
    root=Path(root); manifest=json.loads((root/'condition_manifest.json').read_text())
    image={k:v for k,v in manifest.items() if k!='manifest_hash'}
    if digest(image)!=manifest['manifest_hash']: raise ValueError('Frozen manifest identity changed')
    for name,h in manifest['code_hashes'].items():
        if sha(Path(__file__).parent/name)!=h: raise ValueError(f'Code changed after freeze: {name}; create a new attempt')
    for name,h in manifest.get('upstream_code_hashes',{}).items():
        if sha(ROOT/name)!=h: raise ValueError('Inherited native runtime dependency changed')
    models=[model] if model else list(MODELS)
    for name in models:
        p=json.loads((root/'protocols'/f'{name}.json').read_text())
        if digest(p)!=manifest['protocol_hashes'][name]: raise ValueError('Frozen native protocol changed')
        if sha(p['v22_protocol_source'])!=p['v22_protocol_source_sha256']: raise ValueError('Historical protocol source changed')
        for evidence in p.get('source_evidence',[]):
            if sha(evidence['path'])!=evidence['sha256']: raise ValueError('Historical protocol evidence changed')
        if p.get('checkpoint_verification_source_sha256') and sha(p['checkpoint_verification_source'])!=p['checkpoint_verification_source_sha256']: raise ValueError('Checkpoint verification receipt changed')
        for point in manifest['observations'][name]:
            if sha(point['input_path'])!=point['sha256']: raise ValueError('Frozen real observation source changed')
            for source,h in point['source_evidence'].items():
                if sha(source)!=h: raise ValueError('Original Full/phase evidence changed')
        if runtime is not None:
            mandatory=('H','R','dtype','attention','cameras','state_used','image_rotation','skip_semantics','checkpoint','checkpoint_sha256','checkpoint_assets','scientific_code_hashes')
            for field in mandatory:
                if field not in p or field not in runtime or p[field]!=runtime[field]: raise ValueError(f'Frozen/runtime {field} missing or differs for {name}')
            for field in ('CFG','backend','gripper','checkpoint_revision','checkpoint_shards','checkpoint_assets','processor_sha256','normalization_sha256'):
                if field in p and (field not in runtime or p[field]!=runtime[field]): raise ValueError(f'Frozen/runtime {field} differs for {name}')
        for asset in p.get('checkpoint_assets',{}).values():
            if sha(asset['path'])!=asset['sha256']: raise ValueError('Checkpoint processor/config/normalization changed')
        for source,h in p.get('scientific_code_hashes',{}).items():
            if sha(ROOT/source)!=h: raise ValueError('Scientific model dependency changed')
    for row in manifest['official_initial_states']:
        if sha(row['path'])!=row['sha256']: raise ValueError('Official initial state source changed')
    for record in manifest.get('prior_diagnostics',{}).values():
        for source in record['sources']:
            if sha(source['path'])!=source['sha256']: raise ValueError('Prior diagnostic accounting source changed')
    return manifest

def parity_identity(manifest,runtime,model):
    stable={k:v for k,v in runtime.items() if k not in ('runtime_epoch','device','gpu_name')}
    return digest(dict(manifest_hash=manifest['manifest_hash'],runtime_protocol=stable,points=manifest['observations'][model],code_hashes=manifest['code_hashes']))
def verify_parity(root,runtime,model):
    manifest=verify_frozen(root,runtime,model); p=json.loads((Path(root)/'parity'/f'{model}.json').read_text())
    if p.get('passed') is not True or p.get('identity')!=parity_identity(manifest,runtime,model): raise ValueError('Parity is stale or failed')
    if not 0<p.get('actual_predictions',0)<=p.get('reserved_calls',0)<=256 or len(p.get('checks',[]))!=4: raise ValueError('Parity coverage/prediction accounting invalid')
    for check in p['checks']:
        if check.get('passed') is not True or not check.get('coverage',{}).get('roundtrip') or not check.get('coverage',{}).get('native_reference_steps'): raise ValueError('Parity missing native/switching evidence')
        comparisons=check.get('comparisons',[]); kinds={r['comparison'] for r in comparisons}
        native_steps=5 if model=='xiaomi-robotic0' else 10
        if set(check['coverage']['native_reference_steps'])!={2,native_steps}: raise ValueError('Native reference step coverage incomplete')
        if not {'cache_vs_independent','native_roundtrip','upstream_native_D2',f'upstream_native_D{native_steps}','minimal_vs_instrumented'}<=kinds or not all(r.get('passed') is True for r in comparisons): raise ValueError('Parity comparison evidence incomplete')
    return p
def audited_protocol(root,model,base=None):
    protocol=dict(base if base is not None else json.loads((Path(root)/'protocols'/f'{model}.json').read_text()))
    p=json.loads((Path(root)/'parity'/f'{model}.json').read_text()); evidence={}
    for check in p['checks']:
        for comparison in check['comparisons']:
            if comparison.get('comparison')=='cache_vs_independent' and comparison.get('passed') is True:
                evidence[json.dumps(comparison['schedule'],sort_keys=True)]=comparison['left_audit']
    protocol['v22_audited_schedules']=evidence; return protocol
def verify_acceptance(root,model,parity):
    root=Path(root); m=json.loads((root/'condition_manifest.json').read_text()); a=json.loads((root/'runtime_acceptance.json').read_text()); selected=a.get('models',{}).get(model,{})
    if a.get('schema')!=1 or a.get('reviewer_model')!='gpt-6.1-sol' or a.get('reasoning_effort')!='medium' or a.get('manifest_hash')!=m['manifest_hash'] or a.get('runtime_code_hashes')!=m['code_hashes']: raise ValueError('Medium-effort runtime acceptance is missing or stale')
    if selected.get('accepted') is not True or selected.get('parity_identity')!=parity['identity'] or not selected.get('evidence'): raise ValueError(f'{model} has not received evidence-bound medium-effort execution acceptance')
    return selected
