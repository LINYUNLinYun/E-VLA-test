"""Bounded real-input GPU checks; invoked only by an explicitly assigned service."""
import numpy as np
from .common import SUITES, atomic_json, validate_condition


def run(backend, manifest, output):
    from preparations.idea1_v3_1.benchmark import load_banks
    from .policy import Policy
    checks, arrays = [], {}
    records, identities = load_banks([manifest["bank"]["path"]])
    if identities[0]["sha256"] != manifest["bank"]["sha256"]:
        raise ValueError("Gate input bank changed")
    selected = [next(row for row in records if row.get("suite") == suite) for suite in SUITES]
    torch = backend.torch
    def require(name, passed, **detail):
        checks.append(dict(name=name, passed=bool(passed), **detail))
        if not passed:
            atomic_json(output / "gate.json", dict(passed=False, checks=checks))
            raise RuntimeError("VLM preflight failed: " + name)
    def same(name, a, r, aa, rr):
        require(name, np.allclose(a, aa, atol=1e-5, rtol=1e-5) and torch.allclose(r.float(), rr.float(), atol=1e-5, rtol=1e-5),
                decoded_max_abs=float(np.abs(a-aa).max()), raw_max_abs=float((r.float()-rr.float()).abs().max().cpu()))
    for row in selected:
        obs, prompt, seed, suite = row["obs"], row["instruction"], int(row["seed"]), row["suite"]
        native, _ = backend.prepare(obs, prompt)
        native_a, native_r, _ = backend.sample(native, seed, 10)
        full, _ = backend.prepare_residual(obs, prompt)
        a, r, _ = backend.sample(full, seed, 10)
        same(suite + "/full_native", native_a, native_r, a, r)
        arrays[suite + "_full_raw"] = r.float().cpu().numpy()
        for cfg in manifest["conditions"]:
            mask = validate_condition(cfg)
            if not mask:
                continue
            cid = cfg["id"]
            tag = suite + "/" + cid
            full, _ = backend.prepare_residual(obs, prompt, capture_layers=mask, traced=mask)
            capture_audit = full["execution_audit"]
            require(tag + "/actual_capture", capture_audit["captured_layers"] == list(mask) and
                    capture_audit["kv_only_layers"] == [] and capture_audit["patched_layers"] == [] and
                    capture_audit["qkv_layers"] == list(range(1, 19)) and
                    capture_audit["hidden_update_layers"] == list(range(1, 18)) and
                    not capture_audit["grad_enabled"] and capture_audit["selector_calls"] == 0)
            expected = {l: full["traces"][l]["output"].float() - full["traces"][l]["input"].float() for l in mask}
            require(tag + "/residual_identity", set(full["residuals"]) == set(mask) and
                all(torch.equal(full["residuals"][l], expected[l]) for l in mask))
            a, r, _ = backend.sample(full, seed, 2)
            rebuilt, _ = backend.prepare_residual(obs, prompt, skip=mask, history=full["residuals"])
            aa, rr, _ = backend.sample(rebuilt, seed, 2)
            same(tag + "/same_obs_reconstruct_full_d2", a, r, aa, rr)
            require(tag + "/reconstructed_kv", all(torch.equal(full[key], rebuilt[key]) for key in ("K", "V")))
            audit = rebuilt["execution_audit"]
            require(tag + "/actual_execution", audit["kv_only_layers"] == list(mask) and
                audit["qkv_layers"] == [l for l in range(1, 19) if l not in mask] and
                audit["hidden_update_layers"] == [l for l in range(1, 18) if l not in mask] and
                audit["patched_layers"] == list(mask) and audit["captured_layers"] == [] and not audit["grad_enabled"] and audit["selector_calls"] == 0,
                execution_audit=audit)
            pure_native, _ = backend.prepare(obs, prompt, mask)
            a, r, _ = backend.sample(pure_native, seed, 2)
            pure, _ = backend.prepare_residual(obs, prompt, skip=mask)
            aa, rr, _ = backend.sample(pure, seed, 2)
            same(tag + "/pure_native", a, r, aa, rr)
            # A different real observation gives a nontrivial second history.
            earlier = next(x for x in records if x["instruction"] == prompt and x is not row)
            old, _ = backend.prepare_residual(earlier["obs"], prompt, capture_layers=mask)
            ref, _ = backend.prepare_residual(obs, prompt, skip=mask, history=full["residuals"], older=old["residuals"], ratio=2/3, reference=True)
            a, r, _ = backend.sample(ref, seed, 2)
            actual, _ = backend.prepare_residual(obs, prompt, skip=mask, history=full["residuals"], older=old["residuals"], ratio=2/3)
            aa, rr, _ = backend.sample(actual, seed, 2)
            same(tag + "/extrapolation_reference", a, r, aa, rr)
            aaa, rrr, _ = backend.sample(actual, seed, 2)
            same(tag + "/explicit_noise_repeat", aa, rr, aaa, rrr)
            arrays[suite + "_" + cid + "_raw"] = rr.float().cpu().numpy()
        # Exercise actual policy state transitions and seeds after numerical checks.
        for row_cfg in manifest["conditions"]:
            policy = Policy(backend); policy.configure(row_cfg)
            routes = []
            for i in range(6):
                response = policy.predict(obs, prompt, suite, seed+i)
                routes.append(response["route"])
                require(suite + "/" + row_cfg["id"] + f"/policy_round{i}", response["round"] == i and response["step"] == i*5)
                policy.acknowledge(response["actions"][:5])
            if row_cfg["id"] == "full_d10":
                require(suite + "/full_d10/schedule", all(r["refresh"] for r in routes))
            else:
                require(suite + "/" + row_cfg["id"] + "/schedule", [r["refresh"] for r in routes] == [True, False, False, True, False, False] and
                        routes[4]["extrapolation_ratio"] == 1/3 and routes[5]["extrapolation_ratio"] == 2/3)
    np.savez(output / "gate_actions.npz", **arrays)
    result = dict(passed=True, checks=checks, atol=1e-5, rtol=1e-5, bank=identities,
                  manifest_hash=manifest["manifest_hash"], scope="four real suite inputs, every manifest candidate mask and schedule, no closed-loop quality claim",
                  condition_ids=[c["id"] for c in manifest["conditions"]])
    atomic_json(output / "gate.json", result)
    return result
