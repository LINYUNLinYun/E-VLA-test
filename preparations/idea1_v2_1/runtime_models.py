"""Native, persistent inference policies. No upstream or checkpoint edits."""
from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import uuid
import marshal
import os
import inspect
from importlib.metadata import version

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
def _class_code_image(cls):
    return {name:hashlib.sha256(marshal.dumps(method.__code__)).hexdigest()
            for name,method in vars(cls).items() if hasattr(method,'__code__')}


_previous_class=globals().get('NativePolicy')
LOADED_PREDECESSOR = (_class_code_image(_previous_class) if _previous_class else None)
LOADED_CODE_HASHES = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ('runtime_models.py','adapters.py')}
LOADED_EPOCH = {'id':uuid.uuid4().hex,'loaded_at_unix':time.time(),'source_sha256':LOADED_CODE_HASHES}
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from preparations.idea1_v2_1.adapters import (
    xiaomi_hidden_bypass, db_cogact_hidden_bypass, layer_probes)


def digest(value):
    h = hashlib.sha256()
    def add(v):
        if isinstance(v, dict):
            for k in sorted(v):
                h.update(k.encode()); add(v[k])
        elif isinstance(v, (list, tuple)):
            for item in v: add(item)
        elif isinstance(v, np.ndarray):
            h.update(str(v.dtype).encode()); h.update(str(v.shape).encode())
            h.update(np.ascontiguousarray(v).tobytes())
        else:
            h.update(json.dumps(v, sort_keys=True).encode())
    add(value)
    return h.hexdigest()


def quat_axisangle(quat):
    q = np.asarray(quat, dtype=np.float64).copy()
    q[3] = np.clip(q[3], -1, 1)
    den = np.sqrt(1 - q[3] ** 2)
    return np.zeros(3) if np.isclose(den, 0) else q[:3] * 2 * np.arccos(q[3]) / den


def native_observation(obs):
    """Raw LIBERO render → official 180-degree rotation and 8D state."""
    from PIL import Image
    images = [Image.fromarray(np.ascontiguousarray(np.asarray(obs[k], dtype=np.uint8)[::-1, ::-1]))
              for k in ('agentview_image', 'robot0_eye_in_hand_image')]
    state = np.concatenate([obs['robot0_eef_pos'], quat_axisangle(obs['robot0_eef_quat']),
                            obs['robot0_gripper_qpos']])
    if state.shape != (8,):
        raise ValueError(f'Expected native 8D proprioception, got {state.shape}')
    return images, state


@contextmanager
def seeded(torch, seed):
    py, npstate = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(py); np.random.set_state(npstate)


class NativePolicy:
    def __init__(self, model_id, output_dir=None):
        import torch
        self.torch, self.model_id = torch, model_id
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self._load()
        self.model.eval().requires_grad_(False)
        print('LAYER_FORWARD_OVERRIDES', [(i+1,repr(vars(layer).get('forward'))[:300])
              for i,layer in enumerate(self.layers) if 'forward' in vars(layer)], flush=True)
        self.protocol.update(model_id=model_id, dtype='bfloat16', device=str(self.model.device),
                             state_used=model_id == 'xiaomi-robotic0', image_rotation=180,
                             skip_semantics='hidden-update bypass with original-index KV shells')
        self.protocol['versions']={name:version(name) for name in ('torch','transformers','numpy')}
        self.protocol['cuda_runtime']=torch.version.cuda
        self.protocol['gpu_name']=torch.cuda.get_device_name()
        self.protocol['loaded_model_sources']={}
        for obj in [self.model,*self.modules.values(),self.layers[0],self.layers[0].self_attn]:
            source=inspect.getsourcefile(type(obj))
            self.protocol['loaded_model_sources'][type(obj).__name__]={
                'path':source,'sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest() if source else None}
        if self.output_dir:
            (self.output_dir / 'native_protocol.json').write_text(json.dumps(self.protocol, indent=2))
        self.record_runtime_epoch()

    def record_runtime_epoch(self):
        self.protocol['loaded_code_hashes']=LOADED_CODE_HASHES
        self.protocol['runtime_epoch']=LOADED_EPOCH['id']
        if getattr(self,'recorded_epoch',None)==LOADED_EPOCH['id']: return
        self.recorded_epoch=LOADED_EPOCH['id']
        if self.output_dir:
            previous_success_lines=(sum(1 for _ in (self.output_dir/'prediction_ledger.jsonl').open())
                                    if (self.output_dir/'prediction_ledger.jsonl').exists() else 0)
            main=sys.modules.get('__main__')
            mainfn=getattr(main,'main',None)
            legacy_class=getattr(main,'NativePolicy',None)
            snapshot=dict(LOADED_EPOCH,pid=os.getpid(),predecessor_class_code_image=LOADED_PREDECESSOR,
                          initial_server_class_code_image=_class_code_image(legacy_class) if legacy_class else None,
                          server_main_code_image=(hashlib.sha256(marshal.dumps(mainfn.__code__)).hexdigest()
                                                  if mainfn else None),
                          success_ledger_lines_before_epoch=previous_success_lines,
                          protocol=self.protocol)
            with (self.output_dir/'runtime_epochs.jsonl').open('a') as f:
                f.write(json.dumps(snapshot)+'\n')

    def predict(self, obs, instruction, seed, skip=(), reference=False, probe=False,
                cache_path=None, category='formal', native=False, **unused):
        self.record_runtime_epoch()
        attempt=uuid.uuid4().hex
        row=dict(attempt=attempt,category=category,seed=int(seed),skip=list(skip),
                 native=native,reference=reference,probe=probe,epoch=LOADED_EPOCH,
                 timestamp=time.time(),status='started')
        def record(update):
            if self.output_dir:
                with (self.output_dir/'prediction_attempts.jsonl').open('a') as f:
                    f.write(json.dumps(dict(row,**update))+'\n')
        record({})
        try:
            result=self._predict(obs,instruction,seed,skip,reference,probe,cache_path,category,native,**unused)
        except BaseException as exc:
            record(dict(status='failed',error=repr(exc),finished_at=time.time()))
            raise
        result.update(attempt=attempt,loaded_code_hashes=LOADED_CODE_HASHES,runtime_epoch=LOADED_EPOCH['id'])
        record(dict(status='success',finished_at=time.time(),input_hash=result['input_hash'],noise_hash=result['noise_hash']))
        return result

    def _load(self):
        import torch
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
        if self.model_id == 'xiaomi-robotic0':
            path = ROOT / 'shared/checkpoints/xiaomi-robotic0/fc04bbdcc53e498695cfad95d1199091396ef55b'
            self.model = AutoModel.from_pretrained(path, trust_remote_code=True,
                attn_implementation='flash_attention_2', dtype=torch.bfloat16,
                local_files_only=True).cuda()
            self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True,
                use_fast=False, local_files_only=True)
            self.layers = self.model.vlm.model.language_model.layers
            self.adapter = xiaomi_hidden_bypass
            self.modules = {'vision': self.model.vlm.model.visual,
                            'vlm': self.model.vlm.model.language_model, 'action_expert': self.model.dit}
            actual_h=int(self.processor.get_action_mask('libero_all').shape[1])
            self.protocol = dict(H=actual_h, configured_action_length=self.model.config.action_length,
                                 action_mask_shape=list(self.processor.get_action_mask('libero_all').shape),
                                 R=10, D=5, depth=36, CFG=None,
                                 attention='flash_attention_2', gripper='native continuous',
                                 readout='last 16 original VLM layers, all valid KV tokens',
                                 cameras=['agentview','wrist'],
                                 protocol_source='policies/xiaomi-robotic0/deploy/server.py; eval_libero/main.py')
        elif self.model_id == 'db-cogact':
            sys.path.insert(0, str(ROOT / 'policies/db-cogact'))
            from dexbotic.model.cogact.cogact_arch import CogActConfig, CogACTForCausalLM
            path = ROOT / 'shared/checkpoints/db-cogact/2cb20bcaa1e2e39d431b7e3d40a2d7a696388a10'
            hub = ROOT / 'shared/caches/huggingface/hub'
            config = CogActConfig.from_pretrained(path, local_files_only=True)
            config.llm_config = str(hub / 'models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796')
            config.mm_vision_tower = str(hub / 'models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1')
            self.model = CogACTForCausalLM.from_pretrained(path, config=config,
                torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True).cuda()
            vision = self.model.model.mm_vision_tower
            if not vision.is_loaded: vision.load_model()
            self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            stats = json.loads((path / 'norm_stats.json').read_text())
            self.norm = stats.get('norm_stats', stats)['default']
            self.layers, self.adapter = self.model.model.llm.layers, db_cogact_hidden_bypass
            self.modules = {'vision': vision, 'vlm': self.model.model.llm,
                            'action_expert': self.model.model.action_head.net}
            self.protocol = dict(H=16, R={'libero_spatial':12,'libero_object':16,'libero_goal':16,'libero_10':15},
                                 D=10, depth=28, CFG=1.5, attention=self.model.model.llm.config._attn_implementation,
                                 gripper='sign threshold at zero',readout='last normalized text hidden token',
                                 cameras=['agentview'],protocol_source=(
                                     'policies/dexbotic-benchmark/evaluation/configs/libero/libero_{spatial,object,goal,10}.yaml; '
                                     'evaluation/evaluator/libero_evaluator.py default send_image fallback; '
                                     'evaluation/policies/libero_vla_agent.py discrete_gripper default True'))
        else:
            raise ValueError(self.model_id)

    def reset(self):
        # Both native policies are stateless fresh-prefill chunk generators.
        self.record_runtime_epoch()
        return {'ok': True}

    def _clear_redundant_class_forwards(self):
        """Undo only an exact redundant bound-method assignment from HF output capture.

        transformers 4.57.6 utils/generic.py check_model_inputs restores
        module.forward = original_forward after output_hidden_states=True. This
        creates an instance entry even when the original was the class method.
        Identity checks reject every actual override; adapter source guards stay intact.
        """
        cleared=[]
        for number, layer in enumerate(self.layers,1):
            forward=vars(layer).get('forward')
            if forward is None: continue
            if (getattr(forward,'__self__',None) is not layer or
                    getattr(forward,'__func__',None) is not type(layer).forward):
                raise ValueError(f'Unknown instance forward override on L{number}')
            del layer.forward
            cleared.append(number)
        return cleared

    def parity(self, obs, instruction, seed=7, skip=None):
        """Five real predictions, all charged to the diagnostic ledger."""
        skip = skip or ([21] if self.model_id == 'xiaomi-robotic0' else [14])
        outputs = {}
        for label, kwargs in [('native',dict(native=True)), ('empty',{}), ('repeat',{}),
                              ('reference',dict(skip=skip,reference=True)), ('fast',dict(skip=skip))]:
            outputs[label]=self.predict(obs,instruction,seed,category='parity',**kwargs)
        comparisons = {}
        for a,b in [('native','empty'),('empty','repeat'),('reference','fast')]:
            diff=np.abs(outputs[a]['actions']-outputs[b]['actions'])
            comparisons[a+'_'+b]={'max_abs':float(diff.max()),'mean_abs':float(diff.mean()),
                                  'exact':bool(np.array_equal(outputs[a]['actions'],outputs[b]['actions']))}
        same_noise=len({row['noise_hash'] for row in outputs.values()})==1
        fast_counts=outputs['fast']['operator_calls']
        eliminated=all(v==0 for k,v in fast_counts.items() if k.endswith(('.q_proj','.o_proj','.mlp')))
        result={'ok':True,'model':self.model_id,'skip':skip,'same_noise':same_noise,
                'comparisons':comparisons,'operator_calls':{k:v['operator_calls'] for k,v in outputs.items()},
                'passed':same_noise and eliminated and all(v['exact'] for v in comparisons.values()),
                'predictions':5,'input_hash':outputs['native']['input_hash'],
                'noise_hash':outputs['native']['noise_hash'],'timing':{k:v['timing'] for k,v in outputs.items()}}
        if self.output_dir:
            (self.output_dir/'parity.json').write_text(json.dumps(result,indent=2))
            np.savez_compressed(self.output_dir/'parity_actions.npz',**{k:v['actions'] for k,v in outputs.items()})
        return result

    def _prepare(self, obs, instruction, seed):
        torch = self.torch
        images, state = native_observation(obs)
        if self.model_id == 'xiaomi-robotic0':
            text = ("<|im_start|>user\nThe following observations are captured from multiple views.\n"
                    "# Base View\n<|vision_start|><|image_pad|><|vision_end|>\n"
                    "# Left-Wrist View\n<|vision_start|><|image_pad|><|vision_end|>\n"
                    "Generate robot actions for the task:\n" + str(instruction).capitalize() +
                    ". /no_cot<|im_end|>\n<|im_start|>assistant\n<cot></cot><|im_end|>\n")
            data = self.processor(text=[text], images=images, videos=None, padding=True, return_tensors='pt').to(self.model.device)
            data.update(state=torch.as_tensor(np.r_[state, np.zeros(24)], device=self.model.device,
                        dtype=torch.bfloat16).view(1,1,-1),
                        action_mask=self.processor.get_action_mask('libero_all').to(self.model.device, torch.bfloat16),
                        seed=seed, num_steps=5)
            ids = data['input_ids']; valid = data.get('attention_mask', torch.ones_like(ids)).bool()
            vision = ids == self.processor.image_token_id
            masks = {'visual': valid & vision, 'text':valid & ~vision, 'readout':valid}
        else:
            from dexbotic.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
            from dexbotic.tokenization import conversation
            from dexbotic.tokenization.tokenization import tokenizer_image_token
            conv = conversation.conv_templates[self.model.config.chat_template].copy()
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + instruction)
            conv.append_message(conv.roles[1], ' ')
            ids = tokenizer_image_token(conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX,
                                       return_tensors='pt').unsqueeze(0).to(self.model.device)
            # The four published suite YAMLs omit send_image: native evaluator
            # falls back to agentview only. Do not add the wrist camera.
            image_tensor = self.model.process_images(images[:1]).to(self.model.device, torch.bfloat16)
            data = dict(input_ids=ids, image_tensor=image_tensor)
            # Multimodal expansion happens inside native model; hook its exact decoder input below.
            masks = {}
        return data, masks

    def _predict(self, obs, instruction, seed, skip=(), reference=False, probe=False,
                cache_path=None, category='formal', native=False, **unused):
        torch = self.torch
        start = time.perf_counter()
        input_hash = digest({'obs':obs, 'instruction':instruction})
        if cache_path:
            dest = Path(cache_path); dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, **{k:np.asarray(v) for k,v in obs.items()})
            dest.with_suffix('.json').write_text(json.dumps({'instruction':instruction,'seed':seed,'input_hash':input_hash}))
        data, masks = self._prepare(obs, instruction, seed)
        preparation_ms = (time.perf_counter()-start)*1000
        timing, events, handles, features, noise = {}, {}, [], [], []
        operators = {}
        def count(name):
            def hook(*args): operators[name] = operators.get(name, 0) + 1
            return hook
        for number in skip:
            layer = self.layers[number - 1]
            for name, module in [('q_proj',layer.self_attn.q_proj), ('k_proj',layer.self_attn.k_proj),
                                 ('v_proj',layer.self_attn.v_proj), ('o_proj',layer.self_attn.o_proj),
                                 ('mlp',layer.mlp)]:
                key=f'L{number}.{name}'; operators[key]=0
                handles.append(module.register_forward_pre_hook(count(key)))
        def before(name):
            def hook(*args):
                event=torch.cuda.Event(enable_timing=True); event.record()
                events.setdefault(name, []).append([event,None])
            return hook
        def after(name):
            def hook(*args):
                event=torch.cuda.Event(enable_timing=True); event.record(); events[name][-1][1]=event
            return hook
        for name, module in self.modules.items():
            handles += [module.register_forward_pre_hook(before(name)), module.register_forward_hook(after(name))]
        cfg_original=None
        if self.model_id == 'db-cogact':
            net=self.model.model.action_head.net
            cfg_original=net.forward_with_cfg
            def timed_cfg(*args,**kwargs):
                before('action_expert')()
                try: return cfg_original(*args,**kwargs)
                finally: after('action_expert')()
            net.forward_with_cfg=timed_cfg
        # Record actual sampling tensors without replacing the native sampling algorithm.
        original_randn, original_like = torch.randn, torch.randn_like
        def capture(fn):
            def sample(*args, **kwargs):
                out=fn(*args, **kwargs)
                noise.append(out.detach())
                return out
            return sample
        torch.randn, torch.randn_like = capture(original_randn), capture(original_like)
        if probe and self.model_id == 'db-cogact':
            def db_masks(module, args, kwargs):
                hidden=kwargs.get('inputs_embeds')
                if hidden is None: return
                length=hidden.shape[1]; visual_count=576
                # Native prepare_inputs_labels_for_multimodal expands one image marker in-place.
                marker=int((data['input_ids'][0] == -200).nonzero()[0].item())
                valid=torch.ones(hidden.shape[:2],dtype=torch.bool,device=hidden.device)
                visual=torch.zeros_like(valid); visual[:,marker:marker+visual_count]=True
                readout=torch.zeros_like(valid); readout[:,-1]=True
                masks.update(visual=visual,text=valid & ~visual,readout=readout)
            handles.append(self.model.model.llm.register_forward_pre_hook(db_masks,with_kwargs=True))
        rng_before = digest(torch.cuda.get_rng_state().cpu().numpy())
        redundant=self._clear_redundant_class_forwards()
        try:
            context = nullcontext({'native':True}) if native else self.adapter(self.model,skip,reference=reference)
            adapter_tick=time.perf_counter()
            with seeded(torch, int(seed)), torch.inference_mode(), context as audit, (
                    layer_probes(self.layers,masks,features) if probe else nullcontext()):
                timing['adapter_and_rng_setup_ms']=(time.perf_counter()-adapter_tick)*1000
                if self.model_id == 'xiaomi-robotic0':
                    output=self.model(**data)
                    actions=self.processor.decode_action(output.actions,'libero_all')[0,:,:7].cpu().numpy()
                else:
                    actions=np.asarray(self.model.inference_action(**data,inference_args={
                        'cfg_scale':1.5,'num_ddim_steps':10,'action_norms':self.norm}),dtype=np.float32)
                    actions[:,-1]=np.where(actions[:,-1]<0,-1,1)
            torch.cuda.synchronize()
        finally:
            torch.randn, torch.randn_like = original_randn, original_like
            if cfg_original is not None: del self.model.model.action_head.net.forward_with_cfg
            for handle in handles: handle.remove()
            self._clear_redundant_class_forwards()
        for name, pairs in events.items():
            timing[name+'_ms']=sum(a.elapsed_time(b) for a,b in pairs)
        noise_tick=time.perf_counter()
        noise_arrays=[x.cpu().float().numpy() for x in noise]
        noise_hash=digest(noise_arrays)
        timing['noise_recording_ms']=(time.perf_counter()-noise_tick)*1000
        timing.update(input_processing_ms=preparation_ms,server_total_ms=(time.perf_counter()-start)*1000)
        if actions.shape != (self.protocol['H'],7) or not np.isfinite(actions).all():
            raise ValueError(f'Invalid native action output {actions.shape}')
        self.calls += 1
        result=dict(ok=True,actions=actions.astype(np.float32),timing=timing,input_hash=input_hash,
                    noise_hash=noise_hash,noise_shapes=[list(x.shape) for x in noise],
                    rng_before_hash=rng_before,rng_after_hash=digest(torch.cuda.get_rng_state().cpu().numpy()),
                    features=features,audit=audit,operator_calls=operators,
                    redundant_class_forward_cleanup=redundant,seed=int(seed),category=category,call=self.calls,
                    loaded_code_hashes=LOADED_CODE_HASHES,runtime_epoch=LOADED_EPOCH['id'])
        if cache_path:
            dest=Path(cache_path)
            noise_path=dest.with_name(dest.stem+'_noise.npz')
            np.savez_compressed(noise_path,**{f'noise_{i:02d}':array for i,array in enumerate(noise_arrays)})
            prediction_path=dest.with_name(dest.stem+'_prediction.npz')
            np.savez_compressed(prediction_path,actions=result['actions'])
            result.update(noise_path=str(noise_path),prediction_path=str(prediction_path))
        if self.output_dir:
            with (self.output_dir/'prediction_ledger.jsonl').open('a') as f:
                f.write(json.dumps({k:v for k,v in result.items() if k not in ('actions','features')})+'\n')
        return result
