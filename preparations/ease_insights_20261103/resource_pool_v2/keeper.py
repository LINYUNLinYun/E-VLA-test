"""Bootstrap6GiB while the previous16GiB helper still holds its context.

After the old helper exits, the same new context grows to16GiB and uses the
existing8/16GiB command protocol. No model, kernels or GPU lease are loaded.
"""
import argparse
import ctypes

from preparations.ease_insights_20261103.reservoir_v1 import keeper as prior

STAGING_MIB = 6144


class StagedReservoir(prior.Reservoir):
    def acquire(self):
        # Keep the audited64MiB driver bootstrap and its exact UUID check.
        prior.base.DriverAllocation.acquire(self)
        if prior.IDLE_MIB * prior.base.MIB * 4 > self.total_memory_bytes * 3:
            raise ValueError('16GiB exceeds75% of assigned capacity')
        size = (STAGING_MIB - prior.ANCHOR_MIB) * prior.base.MIB
        pointer = prior.base.DEVICE_POINTER()
        self._call('cuMemAlloc_v2', ctypes.byref(pointer), size)
        if not pointer.value:
            raise RuntimeError('CUDA returned a null staging allocation')
        self.extra.append((pointer, size))
        self.allocated_bytes = STAGING_MIB * prior.base.MIB


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('gpu-uuid','control-fd','ready-fd','nonce','parent-pid','parent-start-ticks'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    return prior.run(prior.base.Config(**vars(args)), allocation_factory=StagedReservoir)


if __name__ == '__main__':
    raise SystemExit(main())
