"""Freeze exact masks, relative floors and rollout budgets before execution."""
import argparse
import json
import random
import hashlib
from pathlib import Path

from .common import ROOT, SUITES, atomic_json, digest


DEPTHS = {"pi0": 27, "pi05": 27, "xiaomi-robotic0": 36, "db-cogact": 28}
REFERENCES = {"xiaomi-robotic0": [4,5,6,8,9,10,29,30,31,33,34,35],
              "db-cogact": [5,6,8,9,21,22,23,24,25]}
LAYOUTS = ("uniform", "front_middle", "back_middle", "short_runs", "seeded_random")


def validate_mask(skip, depth, protected=()):
    if any(type(value) is not int for value in skip):
        raise ValueError("Original one-based integer layer numbers required")
    if len(skip) != len(set(skip)) or any(value < 1 or value > depth or value in protected for value in skip):
        raise ValueError("Duplicate, out-of-range or protected original layer")
    return sorted(skip)


def masks(depth, count, protected=(), seed=3101):
    eligible = [value for value in range(1, depth+1) if value not in protected]
    if not 0 < count < len(eligible):
        raise ValueError("Skip budget must preserve at least one eligible block")
    uniform = [eligible[((2*index+1)*len(eligible))//(2*count)] for index in range(count)]
    # Center-distance rankings implement explicit front/back middle biases.
    front = sorted(eligible, key=lambda x:(abs(x-depth*.35), x))[:count]
    back = sorted(eligible, key=lambda x:(abs(x-depth*.65), -x))[:count]
    # Adjacent short segments ordered round-robin across three regions.
    runs = []
    chunks = [eligible[index:index+3] for index in range(0, len(eligible), 3)]
    for offset in range(3):
        for chunk in chunks[offset::3]:
            runs.extend(chunk)
    candidates = [uniform, front, back, runs[:count], random.Random(seed+depth*100+count).sample(eligible,count)]
    result, seen = [], set()
    for layout, candidate in zip(LAYOUTS, candidates, strict=True):
        attempt = 0
        while tuple(sorted(candidate)) in seen:
            attempt += 1
            candidate = random.Random(seed+depth*100+count+attempt*100003).sample(eligible,count)
        candidate = validate_mask(candidate, depth, protected)
        seen.add(tuple(candidate))
        result.append(dict(layout=layout, collision_redraws=attempt, skip=candidate,
                           keep=[x for x in range(1,depth+1) if x not in candidate]))
    return result


def build():
    conditions = []
    for model, depth in DEPTHS.items():
        stage = "E3" if model in ("pi0", "pi05") else "E1"
        budgets = (3,6,9,12) if stage == "E3" else ((12,15,18,21,24) if model == "xiaomi-robotic0" else (9,12,15,18))
        steps = 2 if model == "xiaomi-robotic0" else 10
        base = dict(model=model, experiment=stage, depth=depth, module="vision" if stage=="E3" else "vlm",
                    protected_layers=[1,2,3] if model=="xiaomi-robotic0" else ([1,2] if model=="db-cogact" else []),
                    denoising_steps=steps, schedule=dict(kind="native", m=None, num_steps=steps),
                    horizon={"pi0":50,"pi05":10,"xiaomi-robotic0":10,"db-cogact":16}[model],
                    execute_length=5 if stage=="E3" else (10 if model=="xiaomi-robotic0" else dict(zip(SUITES,(12,16,16,15)))))
        conditions.append(dict(base, id=f"{model}_full", layout="full", skip=[], keep=list(range(1,depth+1)), reference=False))
        if stage == "E1":
            skip = REFERENCES[model]
            conditions.append(dict(base, id=f"{model}_reference_skip{len(skip)}", layout="historical_reference",
                                   skip=skip, keep=[x for x in range(1,depth+1) if x not in skip], reference=True))
        for budget in budgets:
            protected = (1,2,3) if model=="xiaomi-robotic0" else ((1,2) if model=="db-cogact" else ())
            for index, mask in enumerate(masks(depth,budget,protected=protected)):
                label = f"keep{depth-budget}" if stage=="E3" else f"skip{budget}"
                conditions.append(dict(base, id=f"{model}_{label}_{index+1}_{mask['layout']}", reference=False, **mask))
    references = [dict(next(c for c in conditions if c["id"]==f"{model}_full"),
                       id=f"{model}_full_vision_scan_reference",manual_vision_full=True,
                       benchmark_only=True,quality_baseline=f"{model}_full") for model in ("pi0","pi05")]
    checkpoints = {}
    for model in DEPTHS:
        if model in ("pi0","pi05"):
            checkpoint = ROOT/f"shared/checkpoints/openpi/openpi-assets/checkpoints/{'pi05_libero' if model=='pi05' else 'pi0_libero'}"
            artifacts = [checkpoint/"params/_METADATA",checkpoint/"params/manifest.ocdbt",
                         checkpoint/"assets/physical-intelligence/libero/norm_stats.json"]
        else:
            receipt = ROOT/f"preparations/idea1_v2_1/metadata/{model}/weights_verified.json"
            metadata = json.loads(receipt.read_text())
            checkpoint = Path(metadata["checkpoint"])
            artifacts = [receipt,checkpoint/"config.json"]
        checkpoints[model] = dict(path=str(checkpoint),identity_artifacts={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                                                        for p in artifacts if p.exists()},
                                 full_weight_rehash="Native torch adapter rehashes all verified shards at service startup; JAX identity is original checkpoint manifest and assets")
    manifest = dict(revision="idea1-v3.1-preparation-v2", parent_manifest_hash="759fdea316c8125254ec3f1bab94ba1b9dd7a7675bd5f973884ea53526facd16",
                    quality_environment_reuse=False, conditions=conditions,benchmark_references=references,checkpoints=checkpoints,
                    vision_performance_comparisons="Report both native scanned Full and matched static-selected scan all27 Full; matched Full must pass real-input parity before interpretation",
                    suites=list(SUITES), tasks=list(range(10)), screen_inits=[0], confirm_inits=list(range(5)),
                    relative_success_floor=.70, confirmation_near_floor_margin=.10,
                    technical_max_attempts=3,
                    confirmation_min_successes_screen=4,
                    promotion_policy="Full confirms; one best per budget confirms if screen/full>=0.60 for E1, or >=4/40 successes for E3; exploratory shared screen data",
                    mask_numbering="original one-based block IDs; complete residual blocks only",
                    vision_tokens=256, vision_compression="off", action_expert="original full scanned JAX path for pi; no AE layer pruning",
                    performance=dict(warmup=20, samples=100, real_observations=100, timing="completed outputs, all cameras summed for vision"),
                    e2=dict(episodes=[dict(suite=s,task=0,init=i) for s in SUITES for i in (0,1)],
                            variants=[dict(id="full_io",video="all",save_observations=True,environment_reuse=False,workers=1),
                                      dict(id="lean_io",video="none",save_observations=False,environment_reuse=False,workers=1),
                                      dict(id="reuse_env",video="none",save_observations=False,environment_reuse=True,workers=1),
                                      dict(id="two_workers",video="none",save_observations=False,environment_reuse=True,workers=2)],
                            policy_camera_rendering="always enabled", rpc_batching=False,
                            compile_graph="Not enabled by this preparation; require separate parity and timing evidence"),
                    source_files={str(path.relative_to(ROOT)): __import__('hashlib').sha256(path.read_bytes()).hexdigest()
                                  for path in Path(__file__).parent.glob("*.py")})
    manifest["manifest_hash"] = digest(manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    manifest = build()
    counts = {model:sum(c["model"]==model for c in manifest["conditions"]) for model in DEPTHS}
    if args.write:
        if not args.output:
            parser.error("--write requires --output")
        path = Path(args.output)
        if path.exists():
            raise FileExistsError("Frozen manifests are never overwritten")
        atomic_json(path, manifest)
    print(json.dumps(dict(dry_run=not args.write, manifest_hash=manifest["manifest_hash"], conditions=counts,
                          screen_episodes={model:count*40 for model,count in counts.items()}), indent=2))


if __name__ == "__main__":
    main()
