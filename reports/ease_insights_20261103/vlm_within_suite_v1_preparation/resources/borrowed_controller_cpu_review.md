# Borrowed controller CPU review

Reviewed `preparations/ease_insights_20261103/vlm_within_suite_v1/borrowed_controller.py` read-only at SHA-256 `10b6a2cbb31e5cb7ebf654d985f0a215ae6d46a1564f865b6a76be87dc697e5b`. The Root-owned NFS fix opens shared lock FDs readably with `a+`; exclusive locks retain append mode.

Root's latest changes close the dispatch gaps found in the initial review: launch now holds a shared lock on the legacy broker singleton for its whole lifetime; the frozen publisher takes the same lock exclusively before writing its receipt or dispatching. The guard checks the pinned broker receipt, the original queue's non-holder jobs are complete, the two holder rows match their PIDs/UUIDs/argv, and it rechecks the queue hash and old task filenames while borrowing. Attention resolution now rejects any retry suffix beyond the single same-science `__port_retry1`.

A read-only queue snapshot during review showed 33 jobs: 31 complete and the two original holder sessions running as PIDs 631907 and 631915. The borrowed controller was not launched, so the live broker/holder path was not exercised.

The CPU-only tests use temporary files, fake `/proc` FD links and `fdinfo`, fake holder guards, fake Popen handles, fake science modules, and temporary broker locks. They cover remote lease inode/device plus write-FLOCK verification, exact retry completion/tamper guards, borrowed lifecycle restrictions, owner-loss cleanup of borrower-owned children, preservation of existing completion markers, optional-lock behavior, and two shared borrowers excluding an exclusive publisher. Importing the controller did not load `torch`, `libero`, or `mujoco`.

The targeted Python 3.11 borrowed-resource CPU suite ran with:

```bash
PYTHONDONTWRITEBYTECODE=1 policies/realtime-vla-flash/.venv/bin/python -m unittest preparations.ease_insights_20261103.vlm_within_suite_v1.test_borrowed_controller -v
```

Result: 13 tests passed, including the readable-FD regression and two-SH/EX-exclusion contract. The preceding combined Python 3.11 science/runtime/resource suite passed 30 tests before this one-test addition. Current test SHA-256: `4e80f7dfcff571a001a0b4003cbd20e6e350ae64b393f4921806aab0c99579ce`.

No `--launch`, GPU query, model load, signal, broker stop, holder change, plan/queue/case write, or resource release was performed. An early fake run omitted patching the campaign path and attempted a `flock` on the repository broker lock; it returned `EBADF` before acquiring a lock. The final suite redirects all broker and output paths to temporary directories.
