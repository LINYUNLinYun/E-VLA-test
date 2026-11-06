"""Benchmark-only v2: masked prefix KV plus the actual original scanned AE.

The existing scientific pi05 adapter and its native total reference remain
unchanged. Flax method dispatch selects the original method for suffix calls
only; prefix image/VLM stages still use their declared VLM skip and KV shells.
"""


def ae_stage_scanned_suffix(self,observation,prefix_mask,cache,noise):
    from preparations.idea1_v3.pi05_bench import _ae_stage
    return _ae_stage(self,observation,prefix_mask,cache,noise,native_scanned_suffix=True)
