This resource-only revision follows the accepted module discovery:320 new
cases plus160 preserved controls, frozen no_candidate and independent ranking
agreement. It permits separately frozen camera, functional AE-search or
denoising/depth cohorts while retaining exactly the same two original GPU
leases. No old scientific or resource implementation is edited.

The same tested staged handoff applies: establish and verify a real6GiB CUDA
allocation before asking the old resource_pool_v2 guardian to release only its
own large helper. Wait on identity-bound pidfds, take the existing CPU borrower
lock, and grow the new context to16GiB. Original scheduler/session/64MiB helpers
and GPU lease descriptors remain untouched. Handoff is sequential across the
two cards. This is not a continuous16GiB-total guarantee during the brief6GiB
staging interval. Between jobs retain16GiB; during models retain8GiB plus model.

Only --launch performs a GPU allocation. Frozen manifest, allowed cohort root,
all executable sources, owned-process identities, nonce/sequence receipts,
single-child handoff and assigned-process memory checks remain mandatory.
CPU contracts include a real child/pipe test with a fake CUDA implementation;
no GPU numerical evidence is claimed by them. The new keeper shim preserves
the tested implementation and exit status. Unknown scientific failures retain
memory and require a new recorded request/resume, never a policy-failure retry.
