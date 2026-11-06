"""Native checkpoint adapters; constructors are called only by launched servers."""
import hashlib
from pathlib import Path
import time
from types import MethodType

import numpy as np

from .common import ROOT


def raw_observation(obs, instruction):
    from preparations.idea1_v2_1.runtime_models import native_observation
    images, state = native_observation(obs)
    return {"observation/image":np.asarray(images[0]), "observation/wrist_image":np.asarray(images[1]),
            "observation/state":state, "prompt":instruction}


class PiPolicy:
    def __init__(self, condition):
        from .vision import install
        install(condition["skip"],manual_full=condition.get("manual_vision_full",False))
        import jax
        import jax.numpy as jnp
        from openpi.training import config
        from openpi.policies import policy_config
        from openpi.shared import nnx_utils
        from openpi.models import model as model_api
        self.jax, self.jnp, self.model_api = jax, jnp, model_api
        self.condition = condition
        name = "pi05_libero" if condition["model"]=="pi05" else "pi0_libero"
        checkpoint = ROOT/f"shared/checkpoints/openpi/openpi-assets/checkpoints/{name}"
        self.policy = policy_config.create_trained_policy(config.get_config(name), checkpoint, sample_kwargs={"num_steps":10})
        self.model = self.policy._model
        if self.model.action_horizon != condition["horizon"]:
            raise ValueError("Manifest horizon differs from native checkpoint")
        self.vision = nnx_utils.module_jit(MethodType(self._vision_only,self.model))
        self.protocol = dict(condition, checkpoint=str(checkpoint), backend="native JAX OpenPI",
                             H=self.model.action_horizon, R=5, D=10, vision_depth=27, vlm_depth=18,
                             action_expert="original unmodified scanned Gemma module and original sample_actions",
                             vision_timing_boundary="GPU-resident transformed observation ready before timer; all policy cameras through original SigLIP projector, result complete before stop",
                             vision_compression="off", vision_tokens=256,
                             openpi_sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                             for p in [ROOT/"policies/openpi/src/openpi/models/pi0.py",ROOT/"policies/openpi/src/openpi/models/siglip.py",ROOT/"policies/openpi/src/openpi/models/gemma.py"]})

    @staticmethod
    def _vision_only(model, observation):
        from openpi.models import model as model_api
        observation = model_api.preprocess_observation(None, observation, train=False)
        return tuple(model.PaliGemma.img(observation.images[name], train=False)[0] for name in observation.images)

    def reset(self):
        return {"ok":True}

    def prepare_vision(self, obs, instruction):
        data = self.policy._input_transform(raw_observation(obs,instruction))
        data = self.jax.tree.map(lambda value:self.jnp.asarray(value)[None],data)
        return self.model_api.Observation.from_dict(data)

    def vision_benchmark(self, obs, instruction):
        observation = self.prepare_vision(obs,instruction)
        self.jax.block_until_ready(observation)
        begin = time.perf_counter()
        result = self.vision(observation)
        self.jax.block_until_ready(result)
        return dict(vision_ms=(time.perf_counter()-begin)*1000, cameras=len(result),
                    shapes=[list(value.shape) for value in result], total_tokens=sum(value.shape[1] for value in result))

    def predict(self, obs, instruction, seed, **unused):
        from preparations.idea1_v2_1.runtime_client import input_digest, value_digest
        noise = np.random.default_rng(seed).standard_normal((self.model.action_horizon,32),dtype=np.float32)
        begin = time.perf_counter()
        output = self.policy.infer(raw_observation(obs,instruction),noise=noise)
        actions = np.asarray(output["actions"],np.float32)
        return dict(actions=actions, input_hash=input_digest(obs,instruction), noise_hash=value_digest(noise),
                    seed=seed, condition=self.condition["id"], execute_length=5,
                    timing=dict(request_wall_ms=(time.perf_counter()-begin)*1000),
                    audit=dict(NFE=10, vision_skip=self.condition["skip"], original_scanned_ae=True))

    def parity(self, obs, instruction, seed, **unused):
        first = self.predict(obs,instruction,seed)
        second = self.predict(obs,instruction,seed)
        a,b = first["actions"],second["actions"]
        return dict(passed=bool(np.allclose(a,b,atol=1e-5,rtol=1e-5)),max_abs=float(np.abs(a-b).max()),
                    check="same explicit noise repeatability; selected-scan/native Full parity is a separate benchmark gate")


class TorchPolicy:
    def __init__(self, condition):
        from preparations.idea1_v2_2.runtime_models import NativePolicy
        self.condition = condition
        self.policy = NativePolicy(condition["model"],output_dir=None)
        if self.policy.protocol["depth"] != condition["depth"] or self.policy.protocol["H"] != condition["horizon"]:
            raise ValueError("Loaded original model depth/horizon differs from frozen manifest")
        self.protocol = dict(self.policy.protocol,condition=condition,D=condition["denoising_steps"],
                             persistence="disabled in adapter; rollout controls optional artifact I/O")

    def reset(self):
        return self.policy.reset()

    def predict(self, obs, instruction, seed, **unused):
        return self.policy.predict(obs,instruction,seed,skip=self.condition["skip"],schedule=self.condition["schedule"],
                                   category="measurement",cache_path=None)

    def vision_benchmark(self, obs, instruction):
        prepared = self.policy.prepare_condition(obs,instruction,self.condition["skip"])
        return dict(prepared["timing"], scope="vision stage CUDA events measured within a complete fresh prefill")

    def parity(self,obs,instruction,seed,**unused):
        first = self.policy.predict(obs,instruction,seed,skip=self.condition["skip"],schedule=self.condition["schedule"],
                                    category="measurement",reference=True)
        second = self.predict(obs,instruction,seed)
        a,b = first["actions"],second["actions"]
        return dict(passed=bool(np.allclose(a,b,atol=1e-5,rtol=1e-5)),max_abs=float(np.abs(a-b).max()),
                    noise_paired=first["noise_hash"]==second["noise_hash"],
                    check="reference original-block/discard-hidden vs fast original-index KV shell, same complete sampler")


def load(condition):
    return PiPolicy(condition) if condition["model"] in ("pi0","pi05") else TorchPolicy(condition)
