# One-shot port retry helper

The helper is pinned to the v3 campaign and its two original owned sessions. It defaults to a read-only dry run. A future eligible first port-attention can be published once with `--apply`; the helper stops at `published_wait_ack`, and the root operator must inspect each fresh Session receipt.

Run from the repository root:

```bash
python3 preparations/ease_insights_20261103/retry_resource_session_port.py
python3 preparations/ease_insights_20261103/retry_resource_session_port.py --apply
```

`--apply` requires the exact address-in-use attention, an intact activated task with no committed episodes in that target condition, an absent completion marker, unchanged plan/source pins, the original live session and keeper identities, fresh watcher evidence, and a bind-ready port for every slot currently able to claim a descriptor. A running peer is accepted only with its known keeper/server/optional-worker identity; if it becomes idle or attention during validation, its port is checked again and the audit fails closed on a changed tuple. The new descriptor changes only `id` and `activation_file`, and any retry-of-retry is refused.

The helper does not query GPUs, signal processes, start services, alter `jobs_state`, or modify source, plan, queue, case, status, or outcome files. It makes no claim that the historical bind error was caused by `TIME_WAIT`.
