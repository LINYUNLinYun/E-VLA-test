"""Fail-closed native 2D PushT PI0 loading contract; CPU preparation only.

The checkpoint does not record its training LeRobot commit. The independently
pinned compatibility source is v0.4.3, not a claimed reconstruction of that
unknown training version. The saved SmolVLA text processor is never executed.
"""
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

CHECKPOINT_REVISION = '0b7c4aec769727469283937fd752c3d4d5acdbbc'
CHECKPOINT_SHA256 = 'dd232f08d73ed7954f4e60a9eb8d387bc94e3b94508e9ba76369729da1b8cdb7'
CONFIG_SHA256 = '6d50e42485314a64ff5f5c0bd4faebf238bc37b698137ba3788e0e4c9bd0952a'
STATS_SHA256 = 'bc174a7436ff5075efebe7be7cece5e5d8430658cc9b5314fb8a652eab912d96'
LEROBOT_REVISION = '0b067df57d21d3a02d6c511f1609172fa39ac29b'
TOKENIZER_REPOSITORY = 'google/paligemma-3b-pt-224'
TOKENIZER_REVISION = '35e4f46485b4d07967e7e9935bc3786aad50687c'
PRE_STATS = 'policy_preprocessor_step_5_normalizer_processor.safetensors'
POST_STATS = 'policy_postprocessor_step_0_unnormalizer_processor.safetensors'


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor_header(path):
    with Path(path).open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('Incomplete safetensors prefix')
        length = struct.unpack('<Q', prefix)[0]
        if not 1 <= length <= 4*1024*1024:
            raise ValueError('Safetensors header exceeds CPU audit cap')
        body = stream.read(length)
        if len(body) != length:
            raise ValueError('Incomplete safetensors header')
    return json.loads(body), 8+length


def read_xy_stats(path):
    """Read only four tiny float32 statistics, without loading model tensors."""
    header, offset = tensor_header(path); result = {}
    with Path(path).open('rb') as stream:
        for feature in ('observation.state', 'action'):
            result[feature] = {}
            for name in ('mean', 'std'):
                entry = header[feature+'.'+name]
                start, end = entry['data_offsets']
                if entry['dtype'] != 'F32' or entry['shape'] != [2] or end-start != 8:
                    raise ValueError('Native PushT statistics must be finite float32 XY vectors')
                stream.seek(offset+start); values = list(struct.unpack('<2f', stream.read(8)))
                if not all(math.isfinite(v) for v in values) or (name == 'std' and min(values) <= 0):
                    raise ValueError('Invalid native XY normalization statistics')
                result[feature][name] = values
    return result


def validate_config(config):
    required = dict(type='pi0', chunk_size=50, n_action_steps=50, num_inference_steps=10,
                    max_state_dim=32, max_action_dim=32, paligemma_variant='gemma_2b',
                    action_expert_variant='gemma_300m', tokenizer_max_length=48)
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError('Unexpected native PI0 contract: '+key)
    if config['input_features']['observation.state']['shape'] != [2] or config['output_features']['action']['shape'] != [2]:
        raise ValueError('Only actual native XY state/actions are supported')
    if config['input_features']['observation.image']['shape'] != [3,96,96]:
        raise ValueError('Released native PushT observation images are 96x96 RGB')
    if config['normalization_mapping'] != {'VISUAL':'IDENTITY','STATE':'MEAN_STD','ACTION':'MEAN_STD'}:
        raise ValueError('Normalization differs from the released native checkpoint')
    return required


def strict_key_gate(header, expected_shapes, tied_aliases=None):
    """Reject every unaccounted key/shape; only proven native tied aliases pass."""
    stored = {k:tuple(v['shape']) for k,v in header.items() if k != '__metadata__'}
    aliases = tied_aliases or {}
    metadata = header.get('__metadata__', {})
    missing = set(expected_shapes)-set(stored)
    for key in list(missing):
        target = metadata.get(key)
        if aliases.get(key) == target and target in stored and tuple(expected_shapes[key]) == stored[target]:
            missing.remove(key)
    unexpected = set(stored)-set(expected_shapes)
    wrong = {k:(stored[k],tuple(expected_shapes[k])) for k in set(stored)&set(expected_shapes)
             if stored[k] != tuple(expected_shapes[k])}
    if missing or unexpected or wrong:
        raise ValueError(f'Strict native weight gate failed: missing={sorted(missing)}, unexpected={sorted(unexpected)}, shapes={wrong}')
    return dict(missing=[], unexpected=[], wrong_shapes={}, stored_tensors=len(stored), tied_aliases=aliases)


def audit_checkpoint(directory, verify_weight_sha=True):
    directory = Path(directory)
    for filename, expected in [('config.json',CONFIG_SHA256),(PRE_STATS,STATS_SHA256),(POST_STATS,STATS_SHA256)]:
        if file_sha(directory/filename) != expected:
            raise ValueError('Pinned native metadata or normalization artifact changed: '+filename)
    config = json.loads((directory/'config.json').read_text())
    contract = validate_config(config); weight = directory/'model.safetensors'
    if verify_weight_sha and file_sha(weight) != CHECKPOINT_SHA256:
        raise ValueError('Native PushT weight SHA256 does not match the pinned release')
    header, _ = tensor_header(weight)
    if len(header)-('__metadata__' in header) != 777 or not any('paligemma_with_expert.gemma_expert' in k for k in header):
        raise ValueError('Checkpoint is not the audited released PI0 tensor structure')
    pre, post = read_xy_stats(directory/PRE_STATS), read_xy_stats(directory/POST_STATS)
    if pre != post:
        raise ValueError('Saved input/output normalization statistics disagree')
    return dict(status='checkpoint_contract_audited_not_model_loaded', checkpoint_revision=CHECKPOINT_REVISION,
                checkpoint_sha256=CHECKPOINT_SHA256, compatibility_source_revision=LEROBOT_REVISION,
                training_source_revision='not recorded in released config', contract=contract, stats=pre,
                saved_text_processor='SmolVLA metadata rejected; use native PI0 PaliGemma factory contract',
                strict_model_key_gate='required at load, not implied by header count',
                tokenizer_required=dict(repository=TOKENIZER_REPOSITORY,revision=TOKENIZER_REVISION), gpu_calls=0)


def normalize_xy(values, stats):
    if len(values) != 2 or not all(math.isfinite(float(x)) for x in values):
        raise ValueError('Expected finite native XY values')
    return [(float(x)-mean)/(std+1e-8) for x,mean,std in zip(values,stats['mean'],stats['std'])]


def denormalize_xy(values, stats):
    if len(values) != 2 or not all(math.isfinite(float(x)) for x in values):
        raise ValueError('Expected finite normalized XY values')
    return [float(x)*std+mean for x,mean,std in zip(values,stats['mean'],stats['std'])]


class NativePushTPolicy:
    """Native H50/R50/D10 XY policy, with explicit local-tokenizer and key gates."""
    def __init__(self, model, tokenizer, stats):
        self.model, self.tokenizer, self.stats = model, tokenizer, stats
        self.reset()

    def reset(self):
        self.model.reset()

    def predict(self, raw, instruction, seed):
        import torch
        image = torch.as_tensor(raw['image'])
        if tuple(image.shape) != (96,96,3) or image.dtype != torch.uint8:
            raise ValueError('Expected native uint8 HWC image; no camera bridge/projection')
        state = normalize_xy(raw['state2'], self.stats['observation.state'])
        text = instruction if instruction.endswith('\n') else instruction+'\n'
        self.tokenizer.padding_side = 'right'
        tokens = self.tokenizer([text], max_length=48, padding='max_length',
                                truncation=True, return_tensors='pt')
        batch = {'observation.image':image.permute(2,0,1).unsqueeze(0).float()/255.,
                 'observation.state':torch.tensor([state],dtype=torch.float32),
                 'observation.language.tokens':tokens['input_ids'],
                 'observation.language.attention_mask':tokens['attention_mask'].bool()}
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.manual_seed(seed)
            result = self.model.predict_action_chunk(batch)[0].cpu().float().tolist()
        if len(result) != 50:
            raise ValueError('Native policy returned an incorrect horizon')
        return dict(actions=[denormalize_xy(x,self.stats['action']) for x in result], execute=50,
                    family='pusht_native_pi0', projected_7d=False)

    @classmethod
    def load_cpu(cls, directory, tokenizer_directory, source_root='/policies/lerobot'):
        """No downloads, no GPU and no permissive state-dict remapping."""
        token_dir = Path(tokenizer_directory)
        if not token_dir.is_dir() or not (token_dir/'tokenizer_config.json').is_file():
            raise PermissionError('Provide normally authorized local PaliGemma tokenizer assets; no network fallback')
        source_root = Path(source_root).resolve()
        revision = subprocess.check_output(['git','-c','safe.directory='+str(source_root),'-C',str(source_root),'rev-parse','HEAD'],text=True).strip()
        if revision != LEROBOT_REVISION:
            raise ValueError('LeRobot compatibility source revision differs')
        audit = audit_checkpoint(directory)
        existing = sys.modules.get('lerobot')
        if existing is not None and (not getattr(existing,'__file__',None) or
                source_root/'src' not in Path(existing.__file__).resolve().parents):
            raise RuntimeError('A different LeRobot source is already imported; use an isolated native policy process')
        sys.path.insert(0,str(source_root/'src'))
        import torch
        from safetensors.torch import load_model
        from transformers import AutoTokenizer
        from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
        from lerobot.policies.pi0.configuration_pi0 import PI0Config
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        tokenizer = AutoTokenizer.from_pretrained(str(token_dir),local_files_only=True)
        if type(tokenizer).__name__ not in ('GemmaTokenizer','GemmaTokenizerFast'):
            raise ValueError('Native PI0 requires an actual Gemma/PaliGemma tokenizer, not the saved SmolVLM tokenizer')
        raw_config = json.loads((Path(directory)/'config.json').read_text())
        config = PI0Config()
        # Checkpoint UI/training metadata is not model configuration; all model
        # fields that exist in this pinned implementation are copied verbatim.
        fields = {f.name for f in dataclasses.fields(config)}
        for key,value in raw_config.items():
            if key in fields and key not in ('input_features','output_features','normalization_mapping','device'):
                setattr(config,key,value)
        config.input_features = {k:PolicyFeature(type=FeatureType(v['type']),shape=tuple(v['shape'])) for k,v in raw_config['input_features'].items()}
        config.output_features = {k:PolicyFeature(type=FeatureType(v['type']),shape=tuple(v['shape'])) for k,v in raw_config['output_features'].items()}
        config.normalization_mapping = {k:NormalizationMode(v) for k,v in raw_config['normalization_mapping'].items()}
        config.device='cpu'; config.gradient_checkpointing=False; config.compile_model=False
        model = PI0Policy(config)
        expected = {k:tuple(v.shape) for k,v in model.state_dict().items()}
        header,_ = tensor_header(Path(directory)/'model.safetensors')
        parameters = dict(model.named_parameters(remove_duplicate=False)); aliases={}
        for alias,target in header.get('__metadata__',{}).items():
            if alias in parameters and target in parameters and parameters[alias] is parameters[target]:
                aliases[alias]=target
        strict_key_gate(header,expected,aliases)
        missing,unexpected = load_model(model,str(Path(directory)/'model.safetensors'),strict=True,device='cpu')
        if missing or unexpected:
            raise ValueError('Strict safetensors loader reported unmatched weights')
        model.eval()
        return cls(model,tokenizer,audit['stats'])
