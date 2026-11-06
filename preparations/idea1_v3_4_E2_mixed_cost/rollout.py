"""Reuse the frozen episode protocol, forwarding this process's client RPC."""
from preparations.idea1_v2_1.runtime_client import rpc
from preparations.idea1_v3_4_E2_module_skip import rollout as _reference


def episode(directory, port, case, config, pool, manifest_hash):
    previous = _reference.rpc
    _reference.rpc = rpc
    try:
        return _reference.episode(directory, port, case, config, pool, manifest_hash)
    finally:
        _reference.rpc = previous
