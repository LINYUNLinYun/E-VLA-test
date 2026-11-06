"""Isolated real-model vision parity diagnostic; does not alter frozen modules."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
from types import MethodType

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--model",choices=("pi0","pi05"),default="pi0")
    parser.add_argument("--launch",action="store_true")
    args = parser.parse_args()
    if not args.launch:
        print("Dry run; explicit --launch required")
        return
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    from openpi.models import siglip, model as model_api
    from openpi.policies import policy_config
    from openpi.training import config
    from openpi.shared import nnx_utils
    from .models import raw_observation
    from .common import ROOT, atomic_json
    name = "pi0_libero" if args.model=="pi0" else "pi05_libero"
    policy = policy_config.create_trained_policy(config.get_config(name),ROOT/f"shared/checkpoints/openpi/openpi-assets/checkpoints/{name}",sample_kwargs={"num_steps":10})
    model = policy._model
    record = pickle.loads(Path(args.bank).read_bytes())[0]
    data = policy._input_transform(raw_observation(record["obs"],record["instruction"]))
    data = jax.tree.map(lambda x:jnp.asarray(x)[None],data)
    obs = model_api.Observation.from_dict(data)
    noise = jnp.asarray(np.random.default_rng(int(record["seed"])).standard_normal((1,model.action_horizon,32),dtype=np.float32))
    original = siglip.Encoder.__call__
    def wrapper(gather):
        @nn.compact
        def call(self,x,deterministic=True):
            if self.is_initializing():
                return original(self,x,deterministic)
            params = self.variables["params"]
            stacked = params["encoderblock"]
            selected = jax.tree.map(lambda v:jnp.take(v,jnp.arange(self.depth),axis=0),stacked) if gather else stacked
            block = nn.remat(siglip.Encoder1DBlock,prevent_cse=False,static_argnums=(2,),policy=getattr(jax.checkpoint_policies,self.remat_policy,None))
            scan = nn.scan(block,variable_axes={"params":0},split_rngs={"params":True,"dropout":True},in_axes=nn.broadcast,length=self.depth)(
                mlp_dim=self.mlp_dim,num_heads=self.num_heads,dropout=self.dropout,dtype_mm=self.dtype_mm,parent=None)
            x,scan_out = scan.apply({"params":selected},x,deterministic)
            outputs = {f"block{i:02d}":jax.tree.map(lambda v,i=i:v[i],scan_out) for i in range(self.depth)}
            norm = nn.LayerNorm(dtype=self.dtype_mm,parent=None)
            return norm.apply({"params":params["encoder_norm"]},x),outputs
        return call
    reference = {}
    report = dict(model=args.model,gpu_devices=[str(d) for d in jax.devices()],bank=args.bank,
        diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),modes={})
    def compare(a,b):
        a,b = np.asarray(a,dtype=np.float32),np.asarray(b,dtype=np.float32)
        return dict(max_abs=float(np.max(np.abs(a-b))),rmse=float(np.sqrt(np.mean((a-b)**2))),exact=bool(np.array_equal(a,b)),passed=bool(np.allclose(a,b,atol=1e-5,rtol=1e-5)))
    for label,call in (("native",original),("native_retrace",original),("selected_v2",wrapper(True)),("detached_no_gather",wrapper(False))):
        siglip.Encoder.__call__ = call
        # Fresh Python functions force separately traced variants of one model.
        def image_fn(m,image):
            return m.PaliGemma.img(image,train=False)
        image = nnx_utils.module_jit(MethodType(image_fn,model))
        per_camera = {}
        for camera,pixels in obs.images.items():
            outputs = jax.device_get(image(pixels))
            if label == "native":
                reference[camera] = outputs
            base = reference[camera]
            per_camera[camera] = dict(projected=compare(base[0],outputs[0]),encoded=compare(base[1]["encoded"],outputs[1]["encoded"]),
                blocks={k:{component:compare(base[1]["encoder"][k][component],value) for component,value in row.items()}
                        for k,row in outputs[1]["encoder"].items() if k.startswith("block")})
        def action_fn(m,observation,initial_noise):
            return m.sample_actions(jax.random.key(0),observation,num_steps=10,noise=initial_noise)
        sample = nnx_utils.module_jit(MethodType(action_fn,model))
        actions = np.asarray(sample(obs,noise))
        if label == "native":
            reference["actions"] = actions
        report["modes"][label] = dict(cameras=per_camera,raw_actions=compare(reference["actions"],actions))
        atomic_json(args.output,report)
        print(json.dumps(dict(mode=label,raw_actions=report["modes"][label]["raw_actions"],
            encoded={camera:r["encoded"] for camera,r in per_camera.items()})),flush=True)
    siglip.Encoder.__call__ = original


if __name__ == "__main__":
    main()
