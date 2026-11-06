"""CPU action/state contracts; no import-time model loading or downloads.

The legacy LIBERO six-mode policy and target-trained native positive controls
are different checkpoint families. This module does not disguise one as another.
"""
from dataclasses import dataclass
from contextlib import contextmanager
import numpy as np


def native_checkpoint_header(path, *, key_prefix=""):
    """Bounded CPU-only pi0 header/shape gate, not full model-load parity."""
    import hashlib
    import json
    import struct
    from pathlib import Path
    path=Path(path)
    with path.open("rb") as stream:
        size=struct.unpack("<Q",stream.read(8))[0]
        if not 0 < size < 8*1024*1024:
            raise ValueError("Unbounded/invalid safetensors header")
        raw=stream.read(size)
    header=json.loads(raw)
    expected={"action_in_proj.weight":[1024,32],"state_proj.weight":[1024,32],
        "action_out_proj.weight":[32,1024]}
    for key,shape in expected.items():
        if header.get(key_prefix+key,{}).get("shape") != shape:
            raise ValueError("Native pi0 checkpoint key/shape mismatch: "+key_prefix+key)
    if not any(k.startswith(key_prefix+"paligemma_with_expert.") for k in header):
        raise ValueError("Native pi0 backbone missing")
    for key,row in header.items():
        if key=="__metadata__": continue
        start,end=row["data_offsets"]
        if not 0 <= start <= end <= path.stat().st_size-size-8:
            raise ValueError("Invalid/truncated tensor range: "+key)
    return dict(path=str(path),header_sha256=hashlib.sha256(raw).hexdigest(),
        header_bytes=size,tensor_count=len(header)-("__metadata__" in header),key_prefix=key_prefix,
        critical_shapes={key_prefix+k:v for k,v in expected.items()},
        scope="Header completeness/critical shapes only; constructor strict all-key load remains mandatory")


def install_native_transformers_overlay(root):
    """Load released OpenPI replacements into THIS process; never edit an environment.

    Must run before importing native PI0Pytorch. CPU import only, no constructor.
    """
    import importlib
    import importlib.util
    import sys
    base = root / "policies/realtime-vla-flash/src/openpi/models_pytorch/transformers_replace"
    names = ("models.gemma.configuration_gemma", "models.gemma.modeling_gemma",
        "models.siglip.modeling_siglip", "models.siglip.check", "models.paligemma.modeling_paligemma")
    for relative in names:
        name = "transformers." + relative
        parent_name, attribute = name.rsplit(".",1)
        parent = importlib.import_module(parent_name)
        path = base.joinpath(*relative.split(".")).with_suffix(".py")
        spec = importlib.util.spec_from_file_location(name,path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        setattr(parent,attribute,module)
        # HuggingFace lazy module caches resolved classes on both parent and
        # top-level package; update the declared replacements explicitly.
        for attr, value in vars(module).items():
            if attr.startswith(("Gemma", "Siglip", "PaliGemma")) and isinstance(value,type):
                setattr(parent,attr,value)
                setattr(importlib.import_module("transformers"),attr,value)
    return [str(base.joinpath(*name.split(".")).with_suffix(".py")) for name in names]


@contextmanager
def native_vlm_kv_shell(model, skip_1based):
    """Request-local VLM blocks become identity hidden updates + native K/V shell.

    AE layers remain untouched; every original layer index still has prefix K/V.
    No Q projection/attention/MLP runs in skipped blocks. Request-local cache is
    allocated by canonical sample_actions, not retained here across requests.
    """
    import types
    from transformers.models.gemma.modeling_gemma import apply_rotary_pos_emb
    saved, counts = [], {"kv_shell_calls":0}
    layers = model.paligemma_with_expert.paligemma.language_model.layers
    for onebased in skip_1based:
        layer = layers[onebased-1]
        original = layer.forward
        def shell(this, hidden_states, attention_mask=None, position_ids=None,
                  past_key_value=None, output_attentions=False, use_cache=False,
                  cache_position=None, position_embeddings=None, adarms_cond=None, **kwargs):
            if not use_cache or past_key_value is None or position_embeddings is None:
                raise RuntimeError("V14 shell is prefix-cache-only, never an AE/training substitute")
            normalized, _ = this.input_layernorm(hidden_states,adarms_cond)
            attn = this.self_attn
            shape = (*normalized.shape[:-1],-1,attn.head_dim)
            key = attn.k_proj(normalized).view(shape).transpose(1,2)
            value = attn.v_proj(normalized).view(shape).transpose(1,2)
            cos,sin = position_embeddings
            _,key = apply_rotary_pos_emb(key,key,cos,sin)
            past_key_value.update(key,value,attn.layer_idx,
                {"sin":sin,"cos":cos,"cache_position":cache_position})
            counts["kv_shell_calls"] += 1
            return (hidden_states,None) if output_attentions else (hidden_states,)
        layer.forward = types.MethodType(shell,layer)
        saved.append((layer,original))
    try:
        yield counts
    finally:
        for layer,original in saved:
            layer.forward = original


def _finite(value, shape, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite {shape}")
    return value


def unit_quaternion_xyzw(value):
    q = _finite(value, (4,), "quaternion")
    if not np.isclose(np.linalg.norm(q), 1., atol=1e-6, rtol=0):
        raise ValueError("Unit quaternion required; do not silently normalize provenance")
    return q


def native_libero_axisangle(quaternion_xyzw):
    """Exact native quaternion branch, including native near-identity behavior.

    q and -q encode the same SO(3) rotation but generally DIFFERENT model states.
    Never canonicalize w>=0 before this function.
    """
    q = unit_quaternion_xyzw(quaternion_xyzw).copy()
    q[3] = np.clip(q[3], -1, 1)
    den = np.sqrt(1 - q[3] ** 2)
    return np.zeros(3) if np.isclose(den, 0) else q[:3] * 2 * np.arccos(q[3]) / den


def lift_quaternion(quaternion_xyzw, anchor_xyzw):
    """Choose a continuous double-cover lift nearest a preregistered anchor.

    This is a representation convention, not a learned/corrected TCP offset.
    A rotation matrix alone cannot recover the simulator's original q sign.
    """
    q, anchor = unit_quaternion_xyzw(quaternion_xyzw), unit_quaternion_xyzw(anchor_xyzw)
    dot = float(q @ anchor)
    if abs(dot) < 1e-10:
        raise ValueError("Quaternion lift ambiguous at antipodal equator")
    return q.copy() if dot > 0 else -q


class ContinuousQuaternionLift:
    """Per-episode lift: reset to the source native q, never share across episodes."""
    def __init__(self, source_quaternion_xyzw):
        self.source = unit_quaternion_xyzw(source_quaternion_xyzw).copy()
        self.reset()

    def reset(self):
        self.previous = self.source.copy()

    def encode(self, aligned_quaternion_xyzw):
        self.previous = lift_quaternion(aligned_quaternion_xyzw, self.previous)
        return native_libero_axisangle(self.previous)


def native_libero_state(position, quaternion_xyzw, finger_qpos):
    return np.concatenate((_finite(position, (3,), "position"),
        native_libero_axisangle(quaternion_xyzw), _finite(finger_qpos, (2,), "fingers")))


def normalize(value, mean, std, eps=1e-6):
    value, mean, std = map(lambda x: np.asarray(x, np.float64), (value, mean, std))
    if mean.shape != std.shape or value.shape[-1] != mean.size or np.any(std < 0):
        raise ValueError("Invalid normalizer shape/std")
    result = (value - mean) / (std + eps)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite normalized value")
    return result


def decode_actions(raw, action_mean, action_std, state, *, extra_delta, eps=1e-6):
    """Independent mean/std decoder. extra_delta is checkpoint metadata, not OOD fix."""
    raw = np.asarray(raw, np.float64)
    mean, std = map(lambda x: np.asarray(x, np.float64), (action_mean, action_std))
    if raw.ndim != 2 or mean.shape != std.shape or raw.shape[1] < mean.size or np.any(std < 0):
        raise ValueError("Invalid action decoder dimensions/std")
    actions = raw[:, :mean.size] * (std + eps) + mean
    if extra_delta:
        if mean.size != 7:
            raise ValueError("Released LIBERO extraDelta contract requires action7")
        state = np.asarray(state, np.float64)
        if state.shape != (8,):
            raise ValueError("Released LIBERO extraDelta requires state8")
        actions[:, :6] += state[:6]
    if not np.isfinite(actions).all():
        raise ValueError("Nonfinite decoded action")
    return actions


def calvin_state7(tcp_xyz, tcp_euler_xyz, gripper_value, *, gripper_semantics):
    """Explicit checkpoint contract: official RLinf evaluator uses opening width.

    Do not infer state-field semantics from population statistics: the released
    stats' seventh quantile near +/-1 conflicts with official eval robot_obs[6].
    Record that discrepancy rather than silently replace width with action sign.
    """
    if gripper_semantics not in ("opening_width_m", "action_sign"):
        raise ValueError("Explicit gripper state convention required")
    if not np.isfinite(gripper_value):
        raise ValueError("Finite gripper state required")
    if gripper_semantics == "action_sign" and gripper_value not in (-1, 1):
        raise ValueError("Gripper action sign must be +/-1")
    if gripper_semantics == "opening_width_m" and not 0 <= gripper_value <= .1:
        raise ValueError("CALVIN opening width outside physical contract")
    return np.r_[_finite(tcp_xyz, (3,), "tcp_xyz"),
        _finite(tcp_euler_xyz, (3,), "tcp_euler_xyz"), float(gripper_value)]


def calvin_relative_target(action7, reference_xyz, reference_euler, *,
                           max_rel_pos, max_rel_orn, magic_scaling_pos, magic_scaling_orn):
    """Native relative controller adds Euler coordinates, not SO(3) rotvec composition.

    Caller must select current measured or accumulated target reference according
    to the pinned environment's use_target_pose flag.
    """
    action = _finite(action7, (7,), "CALVIN action")
    return (np.asarray(reference_xyz) + action[:3] * max_rel_pos * magic_scaling_pos,
        np.asarray(reference_euler) + action[3:6] * max_rel_orn * magic_scaling_orn,
        float(action[6]))


@dataclass(frozen=True)
class PolicyContract:
    checkpoint_sha256: str
    normalizer_sha256: str
    action_contract: str
    domain: str
    state_dim: int
    action_dim: int
    horizon: int
    execute: int
    denoising: int = 10

    def __post_init__(self):
        if not self.checkpoint_sha256 or not self.normalizer_sha256:
            raise ValueError("Pinned checkpoint and normalizer hashes required")
        if not 0 < self.execute <= self.horizon:
            raise ValueError("Invalid H/R")


def require_matched_control(full, candidate):
    """Target-tuned Full is not a matched control for LIBERO-trained draft weights."""
    if full != candidate:
        raise ValueError("Unmatched policy contracts: report as distinct positive/transfer controls")


class CorrectedTransferPolicy:
    """Lazy unchanged six-mode backend with explicit corrected client-state contract.

    Pose alignment/lift is client-owned before raw.state8 reaches this wrapper.
    No automatic state clipping, action decoder removal, or checkpoint swap.
    Construction allocates model resources: launch controller must grant GPU first.
    """
    def __init__(self, protocol, manifest_hash, backend=None):
        from preparations.idea1_v3_2.transfer_policy import TransferPolicy
        self.inner = TransferPolicy(protocol, manifest_hash, backend=backend)
        self.inner.protocol.update(state_convention="native LIBERO noncanonical quaternion axis-angle; episode-continuous lift anchored actual source quaternion",
            action_convention="unchanged released LIBERO extraDelta+AbsoluteActions",
            control_family="released LIBERO six-mode zero-shot transfer, not target-tuned native positive control")

    def __getattr__(self, name):
        return getattr(self.inner, name)


class NativeCalvinPolicy:
    """Strict local RLinf ABC→D native Full positive control, H5/R5/D10.

    Uses released native CalvinInputs and existing OpenPI PyTorch policy. V14
    retains K/V shells at skipped blocks; no target-domain draft is fabricated.
    """
    def __init__(self, checkpoint, norm_stats, *, launch=False):
        if not launch:
            raise RuntimeError("EXPLICIT_NATIVE_MODEL_LAUNCH_REQUIRED")
        import importlib.util
        from pathlib import Path
        import sys
        root = Path(__file__).resolve().parents[2]
        checkpoint, norm_stats = Path(checkpoint), Path(norm_stats)
        if not checkpoint.is_file() or not norm_stats.is_file():
            raise FileNotFoundError("Pinned local checkpoint and normalizer required")
        self.header_audit = native_checkpoint_header(checkpoint)
        # Same physical-device immediate idle guard used by the old service.
        from preparations.idea1_v3_2.residual import require_idle_gpu
        self.gpu_preflight = require_idle_gpu()
        sys.path.insert(0,str(root/"policies/realtime-vla-flash/src"))
        sys.path.insert(0,str(root/"policies/realtime-vla-flash/packages/openpi-client/src"))
        self.transformers_overlay = install_native_transformers_overlay(root)
        import safetensors.torch
        from openpi.models.pi0_config import Pi0Config
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from openpi.policies.policy import Policy
        from openpi import transforms
        from openpi.training.config import ModelTransformFactory
        from openpi.shared.normalize import deserialize_json
        from preparations.idea1_v3_2.transfer_policy import local_assets_only
        source = root/"policies/RLinf/rlinf/models/embodiment/openpi/policies/calvin_policy.py"
        name = "_e4_pinned_calvin_inputs"
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        config = Pi0Config(action_horizon=5)
        with local_assets_only():
            model = PI0Pytorch(config=config)
            # Native standalone uses strict=False. We gate every key instead:
            # no silently random/missing target-trained parameters.
            safetensors.torch.load_model(model,str(checkpoint),strict=True,device="cpu")
            model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
            model_transforms = ModelTransformFactory()(config)
            stats = deserialize_json(norm_stats.read_text())
            self.policy = Policy(model,transforms=[module.CalvinInputs(config.model_type),
                transforms.Normalize(stats,use_quantiles=False),*model_transforms.inputs],
                output_transforms=[*model_transforms.outputs,transforms.Unnormalize(stats,use_quantiles=False),
                    module.CalvinOutputs()],sample_kwargs={"num_steps":10},is_pytorch=True,pytorch_device="cuda")
        self.protocol = dict(family="target-trained RLinf CALVIN ABC-D SFT positive control",H=5,R=5,D=10,
            state="XYZ+EulerXYZ+actual robot_obs[-1] stored controller sign; training dataset state.gripper +/-1",
            diagnostic_state="Official standalone evaluator instead used robot_obs[6] opening width; retained only explicit preflight diagnostic",
            extra_delta=False,gripper="decoded native action threshold >0:+1 else:-1 in environment client",
            policy_parity_gpu_verified=False)
        self.legacy_gate_passed = False
        self.counters = {}

    def observation(self,raw,instruction,category):
        state=_finite(raw["state7"],(7,),"native CALVIN state7").astype(np.float32)
        convention=raw.get("gripper_semantics","action_sign")
        if convention=="action_sign":
            if state[6] not in (-1,1): raise ValueError("Training-matched state requires actual stored controller +/-1")
        elif convention=="opening_width_m" and category=="preflight":
            if not 0<=state[6]<=.1: raise ValueError("Invalid diagnostic width")
        else: raise ValueError("Official eval width allowed only as explicit preflight diagnostic")
        return {"observation/image":raw["base_rgb"],"observation/wrist_image":raw["wrist_rgb"],
            "observation/state_ee_pos":state[:3],"observation/state_ee_rot":state[3:6],
            "observation/state_gripper":state[6:7],"prompt":instruction}

    def predict(self, raw, instruction, seed, mode="full_d10", category="quality"):
        if category not in ("preflight","quality","warmup"):
            raise ValueError("Unknown accounting category")
        if category != "preflight" and not self.legacy_gate_passed:
            raise RuntimeError("NATIVE_REAL_INPUT_PARITY_GATE_REQUIRED")
        if mode not in ("full_d10","v14_d10"):
            raise ValueError("Native target-trained draft unavailable; no LIBERO draft substitution/fallback")
        obs=self.observation(raw,instruction,category)
        noise=np.random.default_rng(int(seed)).standard_normal((5,32),dtype=np.float32)
        skip = (3,4,9,11) if mode == "v14_d10" else ()
        with native_vlm_kv_shell(self.policy._model,skip) as observed:
            result=self.policy.infer(obs,noise=noise)
        if observed["kv_shell_calls"] != len(skip):
            raise RuntimeError("Unexpected native V14 prefix shell call count")
        actions=np.asarray(result["actions"],np.float32)
        if actions.shape != (5,7) or not np.isfinite(actions).all():
            raise ValueError("Native CALVIN output must be finite 5x7")
        count=self.counters.setdefault(category,{}).setdefault(mode,{"requests":0,"state_gripper_positive":0,
            "state_gripper_negative":0,"output_elements":0,"output_outside_unit_by_axis":[0]*7})
        count["requests"]+=1
        count["state_gripper_positive"]+=int(obs["observation/state_gripper"][0]>0)
        count["state_gripper_negative"]+=int(obs["observation/state_gripper"][0]<0)
        count["output_elements"]+=len(actions)
        count["output_outside_unit_by_axis"]=(np.asarray(count["output_outside_unit_by_axis"])+np.sum(np.abs(actions)>1,axis=0)).tolist()
        return dict(actions=actions.tolist(),execute_length=5,protocol=self.protocol,
            timing=result.get("policy_timing",{}),mode=mode,seed=int(seed),observed_calls=observed,
            action_contract="native_CALVIN_relative7_Euler_no_extraDelta",skip_vlm_1based=list(skip),
            action_diagnostics=dict(outside_unit_fraction_by_axis=np.mean(np.abs(actions)>1,axis=0).tolist(),
                min_by_axis=actions.min(axis=0).tolist(),max_by_axis=actions.max(axis=0).tolist(),
                gripper_positive_fraction=float(np.mean(actions[:,6]>0)),not_executed_clipping=True))

    def preflight(self, raw, instruction, seed):
        obs=self.observation(raw,instruction,"preflight")
        noise=np.random.default_rng(int(seed)).standard_normal((5,32),dtype=np.float32)
        expected=np.asarray(self.policy.infer(obs,noise=noise)["actions"],np.float32)
        candidate=np.asarray(self.predict(raw,instruction,seed,category="preflight")["actions"],np.float32)
        error=float(np.max(np.abs(expected-candidate)))
        # Approximate V14 is NOT required to equal Full. It must run only four
        # shells, preserve shape/finite and retain untouched native Full route.
        masked=self.predict(raw,instruction,seed,mode="v14_d10",category="preflight")
        self.legacy_gate_passed=bool(np.allclose(expected,candidate,atol=1e-5,rtol=0))
        self.protocol.update(policy_parity_gpu_verified=self.legacy_gate_passed,legacy_gate_passed=self.legacy_gate_passed)
        return dict(passed=self.legacy_gate_passed,full_max_abs=error,full_bitwise=bool(np.array_equal(expected,candidate)),
            v14_observed_calls=masked["observed_calls"],v14_is_approximate=True,
            scope="Same pinned loaded native model and real input/noise, canonical Full vs request wrapper; no target success-rate claim")
