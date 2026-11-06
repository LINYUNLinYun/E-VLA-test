"""Same-loaded-model native/retrace/exact-v2 all27 vision E0; not a quality run."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from types import MethodType

import numpy as np

from .common import ROOT, SUITES, atomic_json, read_manifest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def representatives(records):
    if len(records)<2:
        raise ValueError("At least two distinct benchmark inputs required")
    indices = [0,1]
    covered = {records[i]["suite"] for i in indices}
    for suite in SUITES:
        if suite not in covered:
            index = next((i for i,r in enumerate(records) if r["suite"]==suite),None)
            if index is None:
                raise ValueError(f"No actual input for {suite}")
            indices.append(index)
            covered.add(suite)
    return indices


def compare(left,right):
    a,b = np.asarray(left,dtype=np.float32),np.asarray(right,dtype=np.float32)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    return dict(shape=list(a.shape),max_abs=float(np.max(np.abs(a-b))),rmse=float(np.sqrt(np.mean((a-b)**2))),
                exact=bool(np.array_equal(a,b)),passed=bool(np.allclose(a,b,atol=1e-5,rtol=1e-5)))


def cross_process_controls(root,model):
    """Read observations only; cross-process equality is never silently assumed."""
    controls = dict(native_vs_selected=[],native_vs_native=[])
    a,b = root/(model+"_full")/"benchmark.json",root/(model+"_full_vision_scan_reference")/"benchmark.json"
    if a.exists() and b.exists():
        left,right = json.loads(a.read_text()),json.loads(b.read_text())
        for x,y in zip(left["rows"][:2],right["rows"][:2]):
            controls["native_vs_selected"].append(dict(input_paired=x["input_hash"]==y["input_hash"],noise_paired=x["noise_hash"]==y["noise_hash"],
                actions=compare(x["parity_actions"],y["parity_actions"]),sources=[dict(path=str(p.resolve()),sha256=sha(p)) for p in (a,b)]))
    for p in (root/(model+"_full")/"full_io").glob("*/task00/init*/infer.jsonl"):
        q = root/(model+"_full")/"lean_io"/p.relative_to(root/(model+"_full")/"full_io")
        if q.exists():
            x,y = (json.loads(z.read_text().splitlines()[0]) for z in (p,q))
            controls["native_vs_native"].append(dict(input_paired=x["input_hash"]==y["input_hash"],noise_paired=x["noise_hash"]==y["noise_hash"],
                actions=compare(x["actions"],y["actions"]),sources=[dict(path=str(z.resolve()),sha256=sha(z)) for z in (p,q)]))
    return controls


def run(args,manifest):
    import jax
    import jax.numpy as jnp
    from openpi.models import siglip, model as model_api
    from openpi.policies import policy_config
    from openpi.training import config
    from openpi.shared import nnx_utils
    from .benchmark import load_banks
    from .models import raw_observation
    from .vision import install
    from preparations.idea1_v2_1.runtime_client import input_digest,value_digest
    records,banks = load_banks([args.bank])
    indices = representatives(records)
    name = "pi0_libero" if args.model=="pi0" else "pi05_libero"
    checkpoint = ROOT/f"shared/checkpoints/openpi/openpi-assets/checkpoints/{name}"
    policy = policy_config.create_trained_policy(config.get_config(name),checkpoint,sample_kwargs={"num_steps":10})
    model = policy._model
    original = siglip.Encoder.__call__
    prepared,checks = [],[]
    for index in indices:
        record = records[index]
        data = policy._input_transform(raw_observation(record["obs"],record["instruction"]))
        observation = model_api.Observation.from_dict(jax.tree.map(lambda v:jnp.asarray(v)[None],data))
        noise_np = np.random.default_rng(int(record["seed"])).standard_normal((1,model.action_horizon,32),dtype=np.float32)
        prepared.append((observation,jnp.asarray(noise_np)))
        checks.append(dict(bank_index=index,suite=record["suite"],seed=int(record["seed"]),input_hash=input_digest(record["obs"],record["instruction"]),
            noise_hash=value_digest(noise_np),prepared_hash=value_digest({"data":jax.tree.map(np.asarray,data)}),modes={}))
    def times():
        dt = -1.0/10
        def step(carry):
            t,n,values = carry
            return t+dt,n+1,values.at[n].set(t)
        return jax.lax.while_loop(lambda c:c[0]>=-dt/2,step,(1.0,0,jnp.zeros((11,),jnp.float32)))
    final_time,nfe,time_values = jax.device_get(jax.jit(times)())
    assert int(nfe)==10
    sources = [Path(__file__),Path(__file__).with_name("vision.py"),Path(__file__).with_name("models.py"),
        ROOT/"policies/openpi/src/openpi/models/pi0.py",ROOT/"policies/openpi/src/openpi/models/siglip.py",
        ROOT/"policies/openpi/src/openpi/models/gemma.py"]
    report = dict(passed=False,model=args.model,manifest_hash=manifest["manifest_hash"],atol=1e-5,rtol=1e-5,
        method="same-loaded-model original native / native retrace / exact v2 selected-scan all27",
        source_hashes={str(p.resolve()):sha(p) for p in sources},banks=banks,checkpoint=manifest["checkpoints"][args.model],
        gpu_devices=[str(d) for d in jax.devices()],representative_indices=indices,checks=checks,
        sampler=dict(NFE=int(nfe),num_steps=10,time_values=np.asarray(time_values)[:int(nfe)].tolist(),final_time=float(final_time),
            scope="Original sample_actions invoked unchanged with num_steps10. NFE/time inferred from its pinned FP32 scalar while recurrence; no per-step NN instrumentation."),
        cross_process_controls=cross_process_controls(Path(args.output).parent,args.model),
        scope="All27 vision blocks; one model/weights/input/noise, original scanned VLM/AE and KV/state computation unchanged. This clarifies gate pairing, not tolerances. Cross-process reproducibility observations remain separate.")
    reference = [{} for _ in indices]
    atomic_json(args.output,report)
    for mode in ("native","native_retrace","selected_v2"):
        if mode=="selected_v2":
            install([],manual_full=True)
        else:
            siglip.Encoder.__call__ = original
        def image_fn(m,image):
            return m.PaliGemma.img(image,train=False)
        image = nnx_utils.module_jit(MethodType(image_fn,model))
        def sample_fn(m,observation,noise):
            return m.sample_actions(jax.random.key(0),observation,num_steps=10,noise=noise)
        sample = nnx_utils.module_jit(MethodType(sample_fn,model))
        for index,(obs,noise) in enumerate(prepared):
            cameras = {}
            for camera,pixels in obs.images.items():
                actual = jax.device_get(image(pixels))
                if mode=="native":
                    reference[index][camera] = actual
                base = reference[index][camera]
                blocks = {k:{node:compare(base[1]["encoder"][k][node],v) for node,v in values.items()}
                          for k,values in actual[1]["encoder"].items() if k.startswith("block")}
                assert set(blocks)=={f"block{i:02d}" for i in range(27)}
                assert all(set(v)=={"sa","+sa","mlp","+mlp"} for v in blocks.values())
                cameras[camera] = dict(blocks=blocks,encoded=compare(base[1]["encoded"],actual[1]["encoded"]),projected=compare(base[0],actual[0]))
            raw = np.asarray(sample(obs,noise))
            decoded = policy._output_transform(dict(state=np.asarray(obs.state)[0],actions=raw[0]))["actions"]
            if mode=="native":
                reference[index].update(raw=raw,decoded=decoded)
            checks[index]["modes"][mode] = dict(raw=compare(reference[index]["raw"],raw),decoded=compare(reference[index]["decoded"],decoded),cameras=cameras)
            atomic_json(args.output,report)
            print(json.dumps(dict(mode=mode,bank_index=checks[index]["bank_index"],suite=checks[index]["suite"],raw=checks[index]["modes"][mode]["raw"],decoded=checks[index]["modes"][mode]["decoded"])),flush=True)
    def passed(value):
        if isinstance(value,dict):
            return value["passed"] if "passed" in value else all(passed(v) for v in value.values())
        return all(passed(v) for v in value) if isinstance(value,list) else True
    report["passed"] = all(set(c["modes"])=={"native","native_retrace","selected_v2"} and passed(c["modes"]) for c in checks)
    siglip.Encoder.__call__ = original
    atomic_json(args.output,report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",required=True,choices=("pi0","pi05"))
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--bank",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--launch",action="store_true")
    args = parser.parse_args()
    manifest = read_manifest(args.manifest,verify_sources=args.launch)
    if not args.launch:
        print(json.dumps(dict(dry_run=True,model=args.model,manifest_hash=manifest["manifest_hash"])))
        return
    if Path(args.output).exists():
        raise FileExistsError("Parity artifacts are immutable across invocations")
    old = Path(args.output).with_name(args.model+"_vision_parity.json")
    preserved = old.with_name(old.stem+"_cross_process_observation.json")
    if old.exists() and not preserved.exists():
        shutil.copy2(old,preserved)
    report = run(args,manifest)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
