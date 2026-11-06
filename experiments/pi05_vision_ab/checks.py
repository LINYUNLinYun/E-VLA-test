"""Correctness checks for the deterministic A/B implementation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


# ---------------------------------------------------------------------------
# CPU checks: algorithm definition, independent of model/checkpoint
# ---------------------------------------------------------------------------

def _reference_mapping(x, keep_tokens):
    b = x.shape[0]
    coords = np.arange(256).reshape(16, 16)
    mask = np.zeros((16, 16), bool)
    mask[::2, ::2] = True
    dst = coords[mask]
    src = coords[~mask]
    out = np.zeros((b, 256), np.int32)
    for row in range(b):
        z = x[row].astype(np.float32)
        z /= np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-6)
        sim = z[src] @ z[dst].T
        best = sim.argmax(axis=1)
        score = sim.max(axis=1)
        order = sorted(range(len(src)), key=lambda i: (-score[i], i))
        merged = set(order[: 256 - keep_tokens])
        kept = [i for i in range(len(src)) if i not in merged]
        for rank, i in enumerate(kept):
            out[row, src[i]] = rank
        for i in range(len(src)):
            if i in merged:
                out[row, src[i]] = len(kept) + best[i]
        for rank, i in enumerate(dst):
            out[row, i] = len(kept) + rank
    return out


def cpu_checks():
    import jax.numpy as jnp

    from experiments.pi05_vision_ab import compression as C

    rng = np.random.default_rng(20261020)
    report = {}

    # 1) independent numpy reference for the deterministic mapping
    mapping = {}
    for keep in (128, 64):
        x = rng.standard_normal((4, 256, 32)).astype(np.float32)
        got = np.asarray(C.make_mapping(jnp.asarray(x), keep))
        want = _reference_mapping(x, keep)
        mapping[str(keep)] = {
            "exact_match": bool(np.array_equal(got, want)),
            "groups_per_sample": int(len(np.unique(got[0]))),
            "contiguous": bool(np.array_equal(np.unique(got[0]), np.arange(keep))),
        }
    report["mapping_vs_numpy_reference"] = mapping

    # 2) merge includes the destination, unmerge restores original indices
    groups = np.array([[0, 0, 0, 1, 1, 2, 2, 2]], dtype=np.int32)
    z = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],
                   [10.0, 20.0], [30.0, 40.0],
                   [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]], dtype=np.float32)
    merged = np.asarray(C.merge(jnp.asarray(z), jnp.asarray(groups), 3))
    unmerged = np.asarray(C.unmerge(jnp.asarray(merged), jnp.asarray(groups)))
    hand_means = np.array([[[3.0, 4.0], [20.0, 30.0], [9.0, 10.0]]], dtype=np.float32)
    # Singleton groups prove that tokens outside a merge are untouched.
    singleton_groups = np.array([[0, 1, 1, 2, 2, 2, 3, 4]], dtype=np.int32)
    z2 = rng.standard_normal((1, 8, 4)).astype(np.float32)
    unmerged2 = np.asarray(C.unmerge(
        C.merge(jnp.asarray(z2), jnp.asarray(singleton_groups), 5),
        jnp.asarray(singleton_groups)))
    report["hand_tensor"] = {
        "merge_exact": bool(np.allclose(merged, hand_means, atol=1e-6)),
        "unmerge_exact": bool(np.array_equal(
            unmerged, np.stack([hand_means[0, g] for g in groups[0]])[None])),
        "destination_included": bool(np.allclose(merged[0, 0], [3.0, 4.0], atol=1e-6)),
        "group_values_restored": bool(
            np.allclose(unmerged[0, 3], [20.0, 30.0], atol=1e-6)
            and np.allclose(unmerged[0, 7], [9.0, 10.0], atol=1e-6)),
        "singleton_independent": bool(np.array_equal(unmerged2[0, [0, 6, 7]], z2[0, [0, 6, 7]])),
    }

    # 3) dense residual: same-group branch update, original differences kept
    for keep in (128, 64):
        x = rng.standard_normal((2, 256, 16)).astype(np.float32)
        g = np.asarray(C.make_mapping(jnp.asarray(x), keep))
        branch = rng.standard_normal((2, keep, 16)).astype(np.float32)
        dense = np.asarray(C.unmerge(jnp.asarray(branch), jnp.asarray(g)))
        same_group_same_update = all(
            np.array_equal(dense[b][g[b] == g[b, i]], np.broadcast_to(dense[b, i], dense[b][g[b] == g[b, i]].shape))
            for b in range(2) for i in range(256)
        )
        residual = x + dense
        diff_kept = all(
            np.allclose(residual[b, i] - residual[b, j], x[b, i] - x[b, j], atol=1e-6, rtol=1e-5)
            for b in range(2)
            for i in range(256)
            for j in np.nonzero(g[b] == g[b, i])[0][:3]
        )
        report[f"dense_residual_{keep}"] = {
            "same_group_same_update": bool(same_group_same_update),
            "original_differences_preserved": bool(diff_kept),
        }

    # 4) K=N identity: g_i=i, merge/unmerge bit-exact
    x = rng.standard_normal((2, 256, 16)).astype(np.float32)
    g = np.asarray(C.make_mapping(jnp.asarray(x), 256))
    identity = np.asarray(C.unmerge(C.merge(jnp.asarray(x), jnp.asarray(g), 256), jnp.asarray(g)))
    report["kN_identity"] = {
        "g_is_arange": bool(np.array_equal(g, np.broadcast_to(np.arange(256), g.shape))),
        "bit_exact": bool(np.array_equal(identity, x)),
    }
    report["all_passed"] = all(
        value.get("exact_match", True) and value.get("bit_exact", True)
        and value.get("merge_exact", True) and value.get("unmerge_exact", True)
        and value.get("same_group_same_update", True)
        and value.get("original_differences_preserved", True)
        and value.get("singleton_independent", True)
        and value.get("group_values_restored", True)
        and value.get("g_is_arange", True)
        for value in report.values() if isinstance(value, dict)
    )
    return report


# ---------------------------------------------------------------------------
# GPU parity: real checkpoint, mode off vs K=256 modified path
# ---------------------------------------------------------------------------

def _probe_observations(limit_per_suite=2):
    """Read the already-collected P0 inputs without requiring LIBERO."""
    pattern = "policies/realtime-vla-flash/results/idea1_v3/p0_*/episodes/E1/*/*/*/task*/init*/profile_inputs.pkl"
    paths = sorted(ROOT.glob(pattern))
    import pickle

    from preparations.idea1_v2_1.runtime_models import native_observation

    probes = []
    for suite in SUITES:
        count = 0
        for path in paths:
            if f"/{suite}/" not in str(path):
                continue
            for item in pickle.loads(path.read_bytes()):
                images, state = native_observation(item["obs"])
                probes.append({
                    "observation_id": f"{path.parent.relative_to(ROOT)}#{count}",
                    "suite": suite,
                    "instruction": item["instruction"],
                    "agentview_image": np.asarray(images[0]),
                    "wrist_image": np.asarray(images[1]),
                    "state": state,
                })
                count += 1
                if count >= limit_per_suite:
                    break
            if count >= limit_per_suite:
                break
    return probes


def run_parity_condition(mode, keep_tokens, output):
    """Run in the model environment for one static condition."""
    import jax
    import jax.numpy as jnp

    from experiments.pi05_vision_ab.policy_adapter import Pi05VisionPolicy

    probes = _probe_observations()
    policy = Pi05VisionPolicy(mode, keep_tokens, f"check_{mode}{keep_tokens}")
    single, all_slots = policy.make_vision_fns()
    noise = np.random.default_rng(7).standard_normal((10, 32), dtype=np.float32)

    import time
    # Compile the vision call on the first probe; compile time is reported
    # separately and never mixed into steady-state latency.
    _, slots = policy._prepared_inputs(probes[0])
    tick = time.perf_counter()
    jax.block_until_ready(all_slots(slots))
    compile_s = time.perf_counter() - tick

    features = []
    observations = []
    actions = []
    features_repeat = []
    actions_repeat = []
    for probe in probes:
        _, slots = policy._prepared_inputs(probe)
        observations.append(policy.make_observation(probe))
        out = all_slots(slots)
        jax.block_until_ready(out)
        features.append([np.asarray(o, dtype=np.float32) for o in out])
    for observation in observations:
        result = policy.sample_actions(observation, noise)
        jax.block_until_ready(result)
        actions.append(np.asarray(result))
    # Same compiled functions, same inputs: repeat-run variability of the
    # unchanged model (used as the floating-point comparison noise floor).
    for probe in probes:
        _, slots = policy._prepared_inputs(probe)
        out = all_slots(slots)
        jax.block_until_ready(out)
        features_repeat.append([np.asarray(o, dtype=np.float32) for o in out])
    for observation in observations:
        result = policy.sample_actions(observation, noise)
        jax.block_until_ready(result)
        actions_repeat.append(np.asarray(result))

    # Parameter metadata: paths, shapes, dtypes.  Values come from the same
    # checkpoint for every condition by construction; K=256 identity proves
    # they are used unchanged.
    from flax import nnx as _nnx

    _, state = _nnx.split(policy.policy._model)
    param_meta = sorted(
        (str(path), tuple(np.shape(leaf)), str(getattr(leaf, "dtype", "")))
        for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]
    )
    import hashlib
    hasher = hashlib.sha256()
    for path, shape, dtype in param_meta:
        hasher.update(path.encode())
        hasher.update(str(shape).encode())
        hasher.update(dtype.encode())
    for leaf in jax.tree_util.tree_flatten(state)[0]:
        hasher.update(np.ascontiguousarray(np.asarray(leaf)).tobytes())
    param_sha256 = hasher.hexdigest()
    np.savez(
        output,
        features=np.stack(features),
        actions=np.stack(actions),
        features_repeat=np.stack(features_repeat),
        actions_repeat=np.stack(actions_repeat),
        param_meta=np.asarray(json.dumps(param_meta)),
        compile_s=np.asarray(compile_s),
        load_s=np.asarray(policy.load_seconds),
        protocol=np.asarray(json.dumps(policy.protocol)),
        param_sha256=np.asarray(param_sha256),
    )


def run_sanity_condition(mode, keep_tokens, output):
    """Compressed-config sanity: finite outputs, correct shapes, actual K compute."""
    import jax

    from experiments.pi05_vision_ab.policy_adapter import Pi05VisionPolicy

    probes = _probe_observations(limit_per_suite=1)
    policy = Pi05VisionPolicy(mode, keep_tokens, f"sanity_{mode}{keep_tokens}")
    _, all_slots = policy.make_vision_fns()
    noise = np.random.default_rng(11).standard_normal((10, 32), dtype=np.float32)

    import time
    _, slots = policy._prepared_inputs(probes[0])
    tick = time.perf_counter()
    jax.block_until_ready(all_slots(slots))
    compile_vision_s = time.perf_counter() - tick
    observation0 = policy.make_observation(probes[0])
    tick = time.perf_counter()
    jax.block_until_ready(policy.sample_actions(observation0, noise))
    compile_model_s = time.perf_counter() - tick
    feature_shapes = []
    actions = []
    for probe in probes:
        _, slots = policy._prepared_inputs(probe)
        out = all_slots(slots)
        jax.block_until_ready(out)
        feature_shapes.append([list(np.shape(o)) for o in out])
        observation = policy.make_observation(probe)
        result = policy.sample_actions(observation, noise)
        jax.block_until_ready(result)
        actions.append(np.asarray(result, dtype=np.float32))

    from openpi.models import vision_compression as impl
    trace = impl.TRACE
    actions = np.stack(actions)
    result = {
        "config_id": f"sanity_{mode}{keep_tokens}",
        "mode": mode,
        "keep_tokens": keep_tokens,
        "feature_shapes": feature_shapes,
        "action_shape": list(actions.shape),
        "finite": bool(np.isfinite(actions).all()),
        "trace": trace,
        "branch_compute_shapes": [t for t in trace if t[0] == "branch_compute"],
        "unmerge_shapes": [t for t in trace if t[0] == "unmerge_output"],
        "compile_vision_s": compile_vision_s,
        "compile_model_s": compile_model_s,
        "load_s": policy.load_seconds,
        "protocol": policy.protocol,
    }
    result["k_length_compute_observed"] = bool(
        result["branch_compute_shapes"]
        and all(t[1][1] == keep_tokens for t in result["branch_compute_shapes"])
    )
    result["unmerge_restores_256"] = bool(
        result["unmerge_shapes"] and all(t[1][1] == 256 for t in result["unmerge_shapes"])
    )
    Path(output).write_text(json.dumps(result, indent=2))
    np.savez(str(output) + ".npz", actions=actions)
    return result


def _as_float(a):
    """Numpy cannot natively load bfloat16 stored as |V2 without ml_dtypes."""
    if a.dtype.kind == "V":
        import ml_dtypes

        a = a.view(ml_dtypes.bfloat16)
    return a.astype(np.float64)


def compare_parity(off_npz, k256_npz, atol=1e-2):
    off = np.load(off_npz)
    k = np.load(k256_npz)
    feats_off, feats_k = off["features"], k["features"]
    acts_off, acts_k = off["actions"], k["actions"]
    if feats_off.shape != feats_k.shape:
        raise AssertionError(f"feature shapes differ {feats_off.shape} vs {feats_k.shape}")
    if acts_off.shape != acts_k.shape:
        raise AssertionError(f"action shapes differ {acts_off.shape} vs {acts_k.shape}")
    feats_off_f, feats_k_f = _as_float(feats_off), _as_float(feats_k)
    acts_off_f, acts_k_f = _as_float(acts_off), _as_float(acts_k)
    fdiff = feats_k_f - feats_off_f
    adiff = acts_k_f - acts_off_f
    denom = np.linalg.norm(feats_off_f)
    report = {
        "feature_shape": list(feats_off.shape),
        "action_shape": list(acts_off.shape),
        "feature_max_abs": float(np.max(np.abs(fdiff))),
        "feature_mean_abs": float(np.mean(np.abs(fdiff))),
        "feature_relative_l2": float(np.linalg.norm(fdiff) / denom),
        "action_max_abs": float(np.max(np.abs(adiff))),
        "action_mean_abs": float(np.mean(np.abs(adiff))),
        "action_relative_l2": float(np.linalg.norm(adiff) / np.linalg.norm(acts_off_f)),
        "params_identical": bool(
            off["param_meta"].shape == k["param_meta"].shape
            and (off["param_meta"] == k["param_meta"]).all()
            and str(off["param_sha256"]) == str(k["param_sha256"])),
        "param_sha256_off": str(off["param_sha256"]),
        "param_sha256_k256": str(k["param_sha256"]),
        "bitwise_feature": bool(np.array_equal(feats_off, feats_k)),
        "bitwise_action": bool(np.array_equal(acts_off, acts_k)),
        "repeat_feature_max_abs": float(np.max(np.abs(
            _as_float(off["features_repeat"]) - feats_off_f))),
        "repeat_action_max_abs": float(np.max(np.abs(
            _as_float(off["actions_repeat"]) - acts_off_f))),
        "finite_off": bool(np.isfinite(feats_off_f).all() and np.isfinite(acts_off_f).all()),
        "finite_k256": bool(np.isfinite(feats_k_f).all() and np.isfinite(acts_k_f).all()),
        "load_s_off": float(off["load_s"]),
        "load_s_k256": float(k["load_s"]),
        "compile_s_off": float(off["compile_s"]),
        "compile_s_k256": float(k["compile_s"]),
    }
    report["atol"] = atol
    report["within_tolerance"] = bool(
        report["feature_max_abs"] <= atol and report["action_max_abs"] <= atol)
    return report
