# Versioned evaluation entrypoints

The original `evaluation.py`/`native_envs.py` sign3 pilot is preserved, including
its added first-six-channel clipping. It is not relabeled an exact native-action
benchmark. The new `native_exact.py` changes only that interface: decoded first6
are passed through, gripper alone is thresholded. CPU actual-controller evidence
is `native_preflight/calvin_action_exact_v3/receipt.json` under supplement results.

## Native CALVIN formal prefix

Use a separately approved native-checkpoint service/protocol, never the Plus
LIBERO service. After its real-input preflight gate, the new driver supports the
fixed 100-sequence target with ten-new-sequence launches. Repeat the exact same
command/output/plan to resume. Completed cases require matching source, policy,
sequence and mode identities. Technical failures stay in immutable attempts and
are not scored as policy zeros; the next attempt resets the whole sequence.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
PYTHONPATH="$PWD:$PWD/calvin/calvin_env" \
calvin/.venv/bin/python -m preparations.idea1_v3_2_e4_supplement.evaluation_v2 \
  --protocol APPROVED_NATIVE_MANIFEST.json \
  --sequence-file results/idea1_v3_2_e4_supplement/native_preflight/calvin_formal1000/sequences.json \
  --sequence-count 100 --max-new-sequences 10 --modes full_d10 v14_d10 \
  --preflight-input results/idea1_v3_2_e4_supplement/native_preflight/calvin_captured_formalD0/native_input_gripper_sign.npz \
  --port 19143 --output APPROVED_NATIVE_RESULT_DIRECTORY \
  --deadline-unix APPROVED_DEADLINE --video --launch
```

Every request and native step is append-persisted. Each task also has a separate
checkpoint. The complete actual initial robot15, scene24, and RGB are saved, with
cross-mode numeric differences rather than treating differing float hashes as
proof of a physically different initial scene. There is no scene reset between
successful tasks; native oracle success interrupts and discards remaining R5.

## Native LIBERO-Plus and original controls

`run_plus.py` requires the new Plus manifest carrying the actual frozen
`case_selection`. Each task gets Full, V14, and only its matched mother-suite
draft. Original controls use the original repository/interpreter; Plus uses
`/policies/LIBERO-plus/.venv`. Each child gets private LIBERO_CONFIG_PATH and
explicit assigned-device renderer settings. No old environment is uninstalled.

The initial approved segment is four original tasks, one per suite, 12 episodes.
The complete40-task plan is frozen in round-robin order so the same output can
later resume its original120-condition denominator without importing results
from another plan. Further segments still require the coordinator's grant.

```sh
PYTHONPATH="$PWD" /policies/LIBERO-plus/.venv/bin/python \
  -m preparations.idea1_v3_2_e4_supplement.run_plus \
  --protocol APPROVED_PLUS_MANIFEST.json --output APPROVED_ORIGINAL40_OUTPUT \
  --baseline --indices 0 10 20 30 1 11 21 31 2 12 22 32 3 13 23 33 \
    4 14 24 34 5 15 25 35 6 16 26 36 7 17 27 37 8 18 28 38 9 19 29 39 \
  --max-new-cases 4 --port 19143 --gpu 3 \
  --gpu-uuid GPU-555cde36-5e41-bc44-0c69-ea47358fd703 \
  --deadline-unix APPROVED_DEADLINE --video --launch
```

Without `--launch`, both entrypoints only validate arguments/identity and print
their bounded launch scope. They do not construct models or environments.
