This is a resource-only amendment for the already frozen within-suite study.
The user requested a larger reservation, finally specifying **16 GiB**.

The original scheduler, two remote GPU leases and two 64-MiB CUDA helpers stay
alive. A new helper on each already assigned UUID allocates **16 GiB between
jobs and during CPU preparation**. It keeps **8 GiB** while the unchanged
scientific server and environment worker run, then restores 16 GiB after both
children are reaped. The model's memory is additional. This leaves room for the
model on a 24-GiB card; it does not promise a continuous 16-GiB floor during cold
loading, exclusive access, or that smaller outside jobs cannot enter.

Only a fresh **15-second window with 2-second cached polls and nine distinct
samples** permits the first real allocation. Only the original small keeper
may already be present. Failed allocation races return to a new complete window
before another allocation, and do not create scientific attempts. The single
existing observer remains the all-card monitor. Once allocation succeeds,
ending a condition or assignment never releases the reservoir. A slot-specific
`release` file explicitly ends it; parent loss or fatal ownership/driver failure
also releases only this helper's allocations. Other processes are never killed.

The original scientific manifest, server, runner, worker, backend, numerical
gates, tasks, initial states, conditions and selection criteria remain unchanged.
The new guardian calls the original `Science.execute` with the original command
builder. Each new run truthfully records the new controller hash, resource
amendment and reservoir identity. Completed 288 records are preserved. There is
no scientific retry after a policy failure or an incomplete scientific attempt;
an execution exception enters attention while retaining the reservoir.

Resource sources are frozen in a prospective append-only amendment before
launch. The guardian holds the existing borrower CPU singleton and shared
publisher lock, without taking or duplicating either original GPU lease. The
old automatic CPU launch waiters must be retired at a verified childless boundary
before these guardians start. Per-condition claims still prevent duplicate work.

CPU tests use fake CUDA memory accounting, not a real driver, and cover 16/8-GiB
transitions, failed growth, preserved context, commands, the availability window,
outside contexts and restoring memory after successful/failed child cleanup.
Actual allocation and frozen numerical gates are verified after a card becomes
available; CPU tests alone are not evidence that GPU retention succeeded.
