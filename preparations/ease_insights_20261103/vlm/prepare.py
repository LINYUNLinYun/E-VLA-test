"""Freeze discovery and reserve independent validation, without launching a GPU."""
import argparse
import json
from pathlib import Path
import subprocess

from preparations.idea1_v3_1.common import environment
from .common import ROOT, SUITES, ENV_PYTHON, MODULE, atomic_json, digest, read_manifest, sha, conditions


def inspect_inits():
    from libero.libero import benchmark
    from preparations.idea1_v2_1.runtime_client import official_init_states
    import numpy as np
    import hashlib
    rows = []
    for suite in SUITES:
        selected = benchmark.get_benchmark_dict()[suite]()
        for task in range(10):
            states = official_init_states(selected, task)
            if len(states) <= 35:
                raise ValueError(f"Insufficient official inits: {suite}/{task}")
            rows.append(dict(suite=suite, task=task, available=len(states),
                init_sha256={str(i): hashlib.sha256(np.asarray(states[i]).tobytes()).hexdigest() for i in range(30, 36)}))
    return dict(passed=True, cases=rows, discovery_init=30, reserved_validation_inits=list(range(31, 36)))


def prepare(output):
    output = Path(output).resolve()
    path = output / "manifest.json"
    if path.exists():
        manifest = read_manifest(path, verify_sources=True)
        return dict(reused=True, path=str(path), manifest_hash=manifest["manifest_hash"])
    probe = subprocess.check_output([str(ENV_PYTHON), "-m", MODULE + ".prepare", "--inspect-inits"],
                                   cwd=ROOT, env=environment(), text=True)
    inits = json.loads(probe.splitlines()[-1])
    provenance = ROOT / "policies/realtime-vla-flash/results/experiment11_layer_skip_prefix_20261010/provenance/preflight.json"
    checkpoint = json.loads(provenance.read_text())["checkpoint"]
    source_manifest = ROOT / "policies/realtime-vla-flash/results/experiment11_layer_skip_prefix_20261010/protocol/candidate_manifest.json"
    bank = ROOT / "results/idea1_v3_1/input_bank.pkl"
    proposal = ROOT / "preparations/ease_insights_20261103/history/vlm_fallback_proposal.json"
    source = list(Path(__file__).parent.glob("*.py"))
    source += [ROOT / p for p in (
        "preparations/ease_insights_20261103/fastpath/gpu_binding.py",
        "preparations/idea1_v3_1/common.py", "preparations/idea1_v3_1/rollout.py", "preparations/idea1_v3_1/benchmark.py",
        "preparations/idea1_v3_4/rollout.py", "preparations/idea1_v3_4_v12_compensation/backend.py",
        "preparations/idea1_v3_4_v12_compensation/common.py", "preparations/idea1_v3_4_E2_E4_single/torch_model.py",
        "preparations/idea1_v3/policy.py", "preparations/idea1_v3/safe_qk.py", "preparations/idea1_v3/io.py",
        "preparations/idea1_v2_2/pi0_adapter.py", "preparations/idea1_v2_2/common.py", "preparations/idea1_v2_2/provenance.py",
        "preparations/idea1_v2_1/runtime_client.py", "preparations/idea1_v2_1/runtime_models.py",
        "policies/realtime-vla-flash/scripts/spec/triton/pi0_infer.py",
        "policies/realtime-vla-flash/scripts/experiment10/backend.py")]
    manifest = dict(revision="ease-vlm-fixed-mask-discovery-v2-cost-probe", conditions=conditions(), suites=list(SUITES),
        tasks=list(range(10)), discovery_inits=[30], independent_validation=dict(status="locked_pending_discovery_selection",
            inits=list(range(31, 36)), selected_pair=None, preregistered_category_directions=None,
            rule="Freeze pair and directions before launching validation; a new manifest is required"),
        cases_per_condition=40, total_discovery_episodes=160, quality_environment_reuse=False,
        technical_max_attempts=3, official_init_availability=inits,
        environment=dict(seed=7, seed_before_construction=True, settling_steps=10, control_hz=20, image_size=256,
            limits=dict(zip(SUITES, (220, 280, 300, 520)))),
        science=dict(module="vlm", retained_layers=12, depth=18, H=50, R=5, draft_D=2, refresh_D=10,
            schedule="Full-D10 followed by two Fast-D2 requests", compensation="own-path residual extrapolation",
            layer_numbering="original one-based", fixed_mask=True, selector_calls=0, gradient_selector=False,
            interpretation="Periodic Full/fast system; does not establish standalone fast-path or adaptive-selector benefit"),
        noise=dict(function="preparations.idea1_v2_2.common.seed", namespace="idea1-v22", seed=1701,
            key=["suite", "task", "init", "request"], condition_independent=True, shape=[50, 32], dtype="float32"),
        checkpoints={"pi0": dict(path=checkpoint["path"], sha256=checkpoint["sha256"],
            identity_artifacts={str(provenance): sha(provenance), str(source_manifest): sha(source_manifest)})},
        bank=dict(path=str(bank), sha256=sha(bank)),
        proposal=dict(path=str(proposal), sha256=sha(proposal)),
        preflight=dict(status="required_before_rollout", atol=1e-5, rtol=1e-5,
            requirements=["native Full and uncompensated fixed-mask parity", "same-observation residual reconstruction of Full-D2",
                "exact residual layer identities and hidden-update execution", "reference vs Triton extrapolation",
                "no-grad fixed selector-free execution", "policy Full/Fast/Fast schedule and paired noise",
                "paired initial observation before any action"]),
        fixed_input_cost=dict(real_observations_per_suite=1, suites=list(SUITES), conditions="Full and all three fixed masks",
            warmup_rounds_per_condition_per_suite=6, measured_rounds_per_condition_per_suite=12,
            primary="Completed policy_wall_ms; Full/Fast stages and three-request cycles reported separately",
            excluded="Service loading, numerical gate, and six probe warmup rounds are separate preparation costs",
            scope="Cost only; fixed observations do not establish quality, acceptance, or closed-loop speed"),
        analysis=dict(scope="exploratory 40 episodes/condition, not official 50-init evaluation",
            selection="Seek crossing suite differences at usable aggregate success; universal winner supports a fixed path",
            validation="Independent init31-35 remain locked; do not relabel directions after validation"),
        source_files={str(p.relative_to(ROOT)): sha(p) for p in sorted(set(source))})
    manifest["manifest_hash"] = digest(manifest)
    atomic_json(path, manifest)
    return dict(reused=False, path=str(path), manifest_hash=manifest["manifest_hash"],
                discovery_episodes=160, validation_locked=True, gpu_launched=False)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output")
    parser.add_argument("--inspect-inits", action="store_true")
    args = parser.parse_args()
    if not args.inspect_inits and not args.output:
        parser.error("--output is required")
    print(json.dumps(inspect_inits() if args.inspect_inits else prepare(args.output)))


if __name__ == "__main__":
    main()
