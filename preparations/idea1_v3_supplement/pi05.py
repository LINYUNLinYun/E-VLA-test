"""Supplement-only explicit VLM masks with native scanned AE suffix."""
from preparations.idea1_v3 import pi05 as base

REVISION='supplement-pi05-prefix-mask-native-suffix-v1'
_install_prefix=base.install_vlm_mask


def validate_skip(skip,allow_first_two=False):
    skip=tuple(skip)
    if len(skip) not in (4,5) or len(set(skip))!=len(skip) or any(type(i) is not int or not 1<=i<=18 for i in skip):
        raise ValueError('Supply exactly four or five unique original 1-based layers')
    if not allow_first_two and set(skip)&{1,2}: raise ValueError('First two layers retained by default')
    return tuple(sorted(skip))


def install(skip):
    from openpi.models import gemma
    original=_install_prefix(skip)
    prefix=gemma.Module.__call__
    def call(self,embedded,positions,mask,adarms_cond=None,*,kv_cache=None,deterministic=True):
        prefix_only=embedded[0] is not None and all(x is None for x in embedded[1:])
        method=prefix if prefix_only else original
        return method(self,embedded,positions,mask,adarms_cond,kv_cache=kv_cache,deterministic=deterministic)
    gemma.Module.__call__=call
    return original


class Pi05Policy(base.Pi05Policy):
    def __init__(self,config):
        skip=validate_skip(config['skip'],config.get('allow_first_two',False))
        name=config['mask']
        # Fresh supplement model process only; no scientific/live adapter patch.
        if name in base.MASKS and tuple(base.MASKS[name])!=skip: raise ValueError('Mask name collision')
        base.MASKS[name]=skip
        saved=base.install_vlm_mask; base.install_vlm_mask=install
        try: super().__init__(config)
        finally: base.install_vlm_mask=saved
        if self.H!=10: raise ValueError('Supplement requires native H10')
        self.protocol.update(revision=REVISION,ae_structure='original native scanned suffix',
            mask_semantics='prefix-only hidden skip with original-index KV shell; contiguous skips allowed')
