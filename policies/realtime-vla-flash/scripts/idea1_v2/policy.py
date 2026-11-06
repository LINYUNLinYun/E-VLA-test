"""Single-backend D3 routing, with idempotent retry and synchronous pre-execution persistence."""
import hashlib
import json
import time
import numpy as np
from .core import Router, SKIP, WEIGHT, noise, write_json

def make_backend(metadata):
    """Instrument existing backend entrypoints without altering shared implementation."""
    from scripts.experiment10.backend import Backend
    class AuditedBackend(Backend):
        def __init__(self,metadata):
            self.forward_counts={'logical_calls':0,'record_dispatches':0,'ordinary_D10_forwards':0,'graph_replays':0}
            super().__init__(metadata)
        def record(self,*a,**kw):
            self.forward_counts['record_dispatches']+=1
            return super().record(*a,**kw)
        def prepare(self,*a,**kw):
            result=super().prepare(*a,**kw)
            if not getattr(self.runtime,'_v2_instrumented',False):
                original=self.runtime.forward
                def forward(*a,**kw):
                    self.forward_counts['ordinary_D10_forwards']+=1
                    return original(*a,**kw)
                self.runtime.forward=forward
                self.runtime._v2_instrumented=True
            return result
        def infer(self,*a,**kw):
            self.forward_counts['logical_calls']+=1
            result=super().infer(*a,**kw)
            self.forward_counts['graph_replays']+=1
            return result
    return AuditedBackend(metadata)

class Policy:
    def __init__(self,backend,suite,condition,output,mode='alternating'):
        self.backend,self.suite,self.condition,self.output,self.mode=backend,suite,condition,output,mode
        self.reset()

    def reset(self):
        self.router=Router(self.mode=='phase')
        self.cache={}
        self.next_id=0
        self.episode=None

    def infer(self,observation):
        identity=(int(observation['__trace_task_id__']),int(observation['__trace_episode_idx__']),str(observation.get('__v2_episode_attempt__','attempt00')))
        r=int(observation['__trace_infer_id__'])
        inputs={k:observation[k] for k in ('observation/image','observation/wrist_image','observation/state','prompt')}
        digest=hashlib.sha256()
        for key in sorted(inputs):
            value=inputs[key]
            if isinstance(value,str):
                digest.update(value.encode())
            else:
                array=np.asarray(value)
                digest.update(str(array.dtype).encode());digest.update(str(array.shape).encode());digest.update(array.tobytes())
        digest.update(json.dumps(observation.get('__v2_states__',[]),sort_keys=True).encode())
        input_hash=digest.hexdigest()
        if identity!=self.episode:
            if r!=0:
                raise ValueError('New episode must begin at replan zero')
            self.reset();self.episode=identity
        if r in self.cache:
            old_hash,result=self.cache[r]
            if old_hash!=input_hash:
                raise ValueError('Duplicate replan changed its input')
            return {**result,'actions':result['actions'].copy(),'policy_timing':dict(result['policy_timing'])}
        if r!=self.next_id:
            raise ValueError('Replan sequence gap')
        model,reason,slot=self.router.choose(observation.get('__v2_states__',[]))
        if self.mode in ('full','v14'):
            model='Full' if self.mode=='full' else 'V14'
        skip=() if model=='Full' else SKIP
        random=noise(identity[1],r)
        started=time.perf_counter()
        old_length=getattr(self.backend,'length',None)
        self.backend.prepare(inputs,{'suite':self.suite,'task_description':str(inputs['prompt'])},random)
        prepared=time.perf_counter()
        raw,command,timing=self.backend.infer(skip,3)
        command=np.asarray(command).copy()
        if command.shape!=(50,7) or not np.isfinite(command).all():
            raise ValueError('Nonfinite or wrong-shaped prediction')
        if self.backend.weight_hash!=WEIGHT:
            raise ValueError('Checkpoint mismatch')
        directory=self.output/self.suite/('task%02d'%identity[0])/('init%02d'%identity[1])/identity[2]
        directory.mkdir(parents=True,exist_ok=True)
        path=directory/('replan%04d.npz'%r)
        metadata={'model':model,'route_reason':reason,'cycle_slot':slot,'precision_latch':self.router.latch,
                  'candidate_skip_layers_1based':list(skip),'vlm_kv_shell_indices_1based':list(skip),
                  'denoise_steps':3,'denoise_times':[1,2/3,1/3],'action_layer_indices_1based':list(range(1,19)),
                  'condition':self.condition,'suite':self.suite,'checkpoint_sha256':WEIGHT,
                  'noise_sha256':hashlib.sha256(random.tobytes()).hexdigest(),'input_sha256':input_hash,
                  'full_input_path':str(path),'full_input_sha256':input_hash,'trace_infer_id':r,'trace_episode_idx':identity[1],
                  'frame_idx':int(observation.get('__v2_global_step__',0)), 'cuda_policy_ms':float(timing['cuda_policy_ms']),
                  'prepare_ms':(prepared-started)*1000,'sample_actions_ms':(time.perf_counter()-started)*1000,
                  'prompt_graph_rebuild':old_length!=getattr(self.backend,'length',None),
                  'backend_forward_counts_cumulative':dict(getattr(self.backend,'forward_counts',{})),
                  'is_full_pipeline_round':int(model=='Full'),'requested_horizon':5}
        # Original state dtype is preserved; predictions and inputs durable before env executes.
        with path.open('xb') as stream:
            np.savez_compressed(stream,agent_image=np.asarray(inputs['observation/image']).copy(),
                                wrist_image=np.asarray(inputs['observation/wrist_image']).copy(),
                                robot_state=np.asarray(inputs['observation/state']).copy(),prompt=np.asarray(inputs['prompt']),
                                noise=random,actions=command,raw_actions=np.asarray(raw).copy(),
                                metadata_json=np.asarray(json.dumps(metadata)),online_states_json=np.asarray(json.dumps(observation.get('__v2_states__',[]))))
            stream.flush()
            __import__('os').fsync(stream.fileno())
        write_json(path.with_suffix('.json'),dict(metadata,execution_status='predicted_not_yet_confirmed'),exclusive=True)
        result={'actions':command,'accepted_prefix_len':5,'policy_timing':metadata}
        self.cache[r]=(input_hash,result);self.next_id+=1
        return {**result,'actions':command.copy(),'policy_timing':dict(metadata)}

def serve(args):
    from .core import rows
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    import torch
    torch.cuda.init()
    metadata=[]
    for task in range(10):
        path=next((args.source/'full_observations'/args.suite/('task%02d'%task)).glob('init00/replan0000.npz'))
        with np.load(path,allow_pickle=False) as data:
            metadata.append({'suite':args.suite,'task_description':str(data['prompt'].item())})
    startup=time.perf_counter()
    backend=make_backend(metadata)
    # Equal graph preparation before measured requests; no environment/cycle advance.
    path=next((args.source/'full_observations'/args.suite/'task00/init00').glob('replan0000.npz'))
    with np.load(path,allow_pickle=False) as data:
        inputs={'observation/image':data['agent_image'],'observation/wrist_image':data['wrist_image'],
                'observation/state':data['robot_state'],'prompt':str(data['prompt'].item())}
        for mask in ((),SKIP):
            backend.prepare(inputs,{'suite':args.suite,'task_description':inputs['prompt']},noise(0,0))
            backend.infer(mask,3)
    policy=Policy(backend,args.suite,args.condition,args.output/'inputs'/args.condition,args.mode)
    startup_path=args.output/'logs'/('backend_%s_%s_%d.json'%(args.condition,args.suite,time.time_ns()))
    write_json(startup_path,{'initialization_seconds':time.perf_counter()-startup,'forward_counts':backend.forward_counts,
                             'gpu':__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),'condition':args.condition},exclusive=True)
    WebsocketPolicyServer(policy,host='127.0.0.1',port=args.port,
                          metadata={'protocol':{'schema':1},'idea1_v2':True,'checkpoint_sha256':backend.weight_hash},
                          reset_policy_on_connect=True).serve_forever()
