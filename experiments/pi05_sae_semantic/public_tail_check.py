"""Check whether released full-sequence L5 states can be replayed by our tail.

This reuses NativeProbe's validated tail. Agreement is measured against the
published L16 from exactly the same episode/time, before any skipping.
"""
import json
from pathlib import Path
import time

import numpy as np
import torch


def check_public_tail(probe, public_check, native_receipt, output):
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
    from preparations.idea2.experiments import native_raw
    check = json.loads(Path(public_check).read_text())
    records = {r["layer"]: r for r in check["layers"]}
    data = {}
    for layer in (5, 16):
        with np.load(records[layer]["activation_path"], allow_pickle=False) as archive:
            data[layer] = torch.from_numpy(archive["activations"][0].copy())
    if data[5].shape != data[16].shape:
        raise ValueError("Public layer arrays are not aligned")
    prompt = records[5]["auxiliary_metadata"]["prompt"]
    if prompt != records[16]["auxiliary_metadata"]["prompt"]:
        raise ValueError("Public layer prompts are not aligned")
    # Only native preprocessing/masks are used; image embedding is not recomputed.
    with np.load(native_receipt["input"], allow_pickle=False) as archive:
        raw = {k: archive[k].copy() for k in archive.files if k not in ("instruction", "seed")}
    prepared = probe.backend.prepare(native_raw(dict(obs=raw, instruction=prompt)))
    per_camera = native_receipt["layout"]["tokens_per_camera"]
    image_n = len(prepared.images) * per_camera
    if image_n + prepared.tokens.shape[1] != data[5].shape[0]:
        raise ValueError("Published and native sequence layouts differ")
    image_mask = torch.cat([m[:, None].expand(1, per_camera) for m in prepared.image_masks], dim=1)
    native_mask = torch.cat([image_mask, prepared.token_masks], dim=1)
    masks = {"native_prompt_token_mask": native_mask}
    # Repeated trailing text residuals identify a candidate padding suffix.
    text = data[5][image_n:]
    repeated = (text == text[-1]).all(-1)
    trailing = 0
    for equal in reversed(repeated.tolist()):
        if not equal:
            break
        trailing += 1
    if trailing >= 2:
        inferred = native_mask.clone()
        inferred[:, image_n:] = False
        inferred[:, image_n:image_n + len(text) - trailing] = True
        if not torch.equal(inferred, native_mask):
            masks = {"padding_suffix_inferred_from_public_L5": inferred, **masks}
    results = []
    hidden5 = data[5][None].to(probe.backend.device, probe.lm.layers[6].self_attn.q_proj.weight.dtype)
    expected = data[16].float()
    for name, mask in masks.items():
        positions = torch.cumsum(mask, dim=1) - 1
        attention = probe.model._prepare_attention_masks_4d(make_att_2d_masks(mask, torch.zeros_like(mask)))
        started = time.perf_counter()
        actual = probe.tail(hidden5, attention, positions, list(range(data[5].shape[0])))[0].float()
        delta = actual - expected
        row = dict(mask_source=name, valid_text_tokens=int(mask[:, image_n:].sum()),
                   exact_equal=bool(torch.equal(actual, expected)), max_abs=float(delta.abs().max()),
                   rmse=float(delta.square().mean().sqrt()),
                   relative_rmse=float(delta.square().mean().sqrt() / expected.square().mean().sqrt()),
                   elapsed_s=time.perf_counter() - started)
        results.append(row)
        print(json.dumps(dict(event="public_tail_check", **row)), flush=True)
        if row["exact_equal"]:
            break
    receipt = dict(scope="One published episode/time; source GPU vs native CPU may differ numerically",
                   public_prompt=prompt, public_paths={str(k):v["activation_path"] for k,v in records.items()},
                   public_shape=list(data[5].shape), inferred_trailing_text_padding=trailing,
                   device=str(probe.backend.device), results=results,
                   exact_reuse_established=any(r["exact_equal"] for r in results),
                   interpretation="A failed equality check does not by itself distinguish weights, masks, and numerical backend differences")
    Path(output).write_text(json.dumps(receipt, indent=2))
    return receipt
