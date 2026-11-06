# Future path asset and split feasibility

**Status:** metadata feasibility only; not protocol approval, freeze, or launch authorization.

Source digests matched for the candidate (`478b637d0e688ca8178551ef4c18e548c70492045844b1f5374923aeeada47bc`), planning-only parent plan (`3c50de72fbde814ce96f7b3839f8e97ce195e7d50b01c73689b9eb2bde996a9d`), and existing official pool metadata audit (`c7863aa701e1caa5069a9582020bfc40ffb62a6acfcbd7da64aef7185a616144`). For all 24 candidate LIBERO90 tasks, the official BDDL and configured official init-file hashes matched the candidate and existing metadata audit. All state arrays matched metadata shapes; zero-based rows 44–49 were present, numeric, and finite for every task. Per-task BDDL/init bindings and per-row statevector hashes are in [`feasibility.json`](feasibility.json). Hash framing includes source dtype and row shape before contiguous row bytes; state values are not included.

Within each category, exact instruction strings are indivisible groups. The specified SHA-256 rule finds six discovery and six complementary heldout tasks per category without crossing any exact prompt group. Exact groups, task IDs, feasible assignment counts, and digests are in [`feasibility.json`](feasibility.json).

A bounded filename search did not find a clearly named metadata-only historical prior-use audit outside the excluded active `vlm_within_suite_v1` subtree. This report makes no claim that rows 44–49 are unused, especially outside known EASE metadata.

No active outcomes were read; no environment or model was started. Finite array rows establish file-level integrity only, not simulator reset or rendered-scene validity. The 288-episode budget remains planning only; no performance or complementarity claim is made.
