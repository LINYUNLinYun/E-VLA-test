"""Static SigLIP block bypass; leave the VLM and original scanned AE unchanged."""
from .manifest import validate_mask


def install(skip, manual_full=False):
    skip = tuple(validate_mask(skip, 27))
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    from openpi.models import siglip, vision_compression
    if vision_compression.CONFIG["mode"] != "off":
        raise ValueError("Independent vision skipping requires existing compression mode off")
    if hasattr(siglip.Encoder, "_v31_original_call"):
        raise RuntimeError("One frozen vision mask per isolated process; start a new service")
    original = siglip.Encoder.__call__
    siglip.Encoder._v31_original_call = original

    @nn.compact
    def call(self, x, deterministic=True):
        if self.is_initializing() or (not skip and not manual_full):
            return original(self, x, deterministic)
        if self.depth != 27 or not self.scan or not deterministic or x.shape[1] != 256:
            raise ValueError("Require original scanned 27-block, 256-token inference tower")
        params = self.variables["params"]
        stacked = params["encoderblock"]
        keep = [index for index in range(self.depth) if index+1 not in skip]
        selected = jax.tree.map(lambda value:jnp.take(value,jnp.asarray(keep),axis=0),stacked)
        # Preserve the original scan/remat operator structure. Only the static
        # parameter-axis selection and loop length change; original block IDs
        # remain recorded in the manifest, and no AE code is patched.
        block = nn.remat(siglip.Encoder1DBlock,prevent_cse=False,static_argnums=(2,),
                         policy=getattr(jax.checkpoint_policies,self.remat_policy,None))
        scanned = nn.scan(block,variable_axes={"params":0},split_rngs={"params":True,"dropout":True},
                          in_axes=nn.broadcast,length=len(keep))(
                              mlp_dim=self.mlp_dim,num_heads=self.num_heads,dropout=self.dropout,
                              dtype_mm=self.dtype_mm,parent=None)
        x,scan_out = scanned.apply({"params":selected},x,deterministic)
        outputs = {f"block{index:02d}":jax.tree.map(lambda value,position=position:value[position],scan_out)
                   for position,index in enumerate(keep)}
        outputs["pre_ln"] = x
        norm = nn.LayerNorm(dtype=self.dtype_mm, parent=None)
        return norm.apply({"params":params["encoder_norm"]}, x), outputs

    siglip.Encoder.__call__ = call
    return original
