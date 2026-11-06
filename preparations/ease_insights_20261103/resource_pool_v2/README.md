The previous scientific branch completed432 cases and selected no candidate.
Its GPU leases and small keepers remain alive. This separately versioned CPU
pool accepts new, independently frozen local cohorts while retaining the same
two GPUs. No old scientific or resource source is edited.

Handoff is sequential and uses actual allocations. The new helper first keeps
6GiB while the old16GiB helper remains alive. Only after the new nonce/UUID/PID
receipt, assigned-process memory check and a fresh childless-old-owner check
does the new guardian write the old guardian's explicit release file. It waits
on inherited pidfds for both old guardian and old helper to exit, obtains the
existing CPU borrower singleton, then grows the new context to16GiB. Original
scheduler GPU lease descriptors are never duplicated or closed. The6GiB staging
floor is transient; this is not a continuous16GiB-total-memory guarantee during
handoff. On this24GiB hardware, old16 plus new6 and known context overhead fit;
the actual allocation must succeed before release is authorized.

Between jobs the pool keeps16GiB, while scientific execution uses an8GiB
placeholder plus model memory. The exact existing model command builder and
Science.execute lifecycle are reused. Task keys/noise come from each frozen
cohort, not scheduling order. One or two fixed task shards are supported.
Every cohort root, executable module and source file must be pinned in its
manifest before import. A failed scientific request is recorded once and never
automatically rerun. Attention retains memory; a separately recorded operator
acknowledgement bound to PID/start/plan/attention can permit new requests after
children are reaped and source/lease/context checks pass. Existing outcomes
remain unchanged.

The launcher supplies pidfds using system Python because the model interpreter
may not expose os.pidfd_open. Only these observation descriptors are inherited;
the CUDA helper receives only its control/receipt pipes and closes unrelated
descriptors. Only --launch allocates; --launch is used after CPU tests and a
source-pinned plan. Explicit slot release or fatal parent/driver loss cleans
up only this pool's children and allocations.
