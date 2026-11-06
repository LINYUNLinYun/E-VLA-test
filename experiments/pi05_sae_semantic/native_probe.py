"""Thin OpenPI adapter: native preprocessing, read-only layer captures, FFN switches.

Uses the existing project loader and the release's own SAE implementation.
Only the bounded VLM tail intervention is new research code.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def sampling_layout(prepared,instruction,sequence_length,tokenizer,discrete_state_input=False):
    text_len=prepared.tokens.shape[1];image_len=sequence_length-text_len
    per_camera=image_len//len(prepared.images)
    if per_camera*len(prepared.images)!=image_len:
        raise ValueError('Cannot derive equal camera spans from runtime shapes')
    cleaned=instruction.strip().replace('_',' ').replace('\n',' ')
    task_ids=tokenizer.encode(cleaned,add_bos=True)
    expected=(task_ids+tokenizer.encode('\n'))[:text_len]
    if prepared.tokens[0,:len(expected)].cpu().tolist()!=expected or discrete_state_input:
        raise ValueError('Instruction format differs from the audited nondiscrete prefix')
    text_positions=np.arange(1,min(len(task_ids),text_len))+image_len
    if len(text_positions)>32:text_positions=text_positions[np.linspace(0,len(text_positions)-1,32).astype(int)]
    active=[i for i,m in enumerate(prepared.image_masks) if bool(m[0])]
    remaining=64-len(text_positions);selected=text_positions.tolist();kinds=['instruction']*len(selected)
    for j,camera in enumerate(active):
        n=remaining//len(active)+int(j<remaining%len(active))
        selected.extend((camera*per_camera+np.linspace(0,per_camera-1,n).astype(int)).tolist());kinds.extend([f'camera_{camera}']*n)
    for position in text_positions:
        if not bool(prepared.token_masks[0,int(position-image_len)]):raise ValueError('Selected padded instruction token')
    return dict(instruction=instruction,sequence_length=sequence_length,tokens_per_camera=per_camera,
        camera_masks=[bool(m[0]) for m in prepared.image_masks],instruction_tokens=len(task_ids)-1,
        selected_positions=selected,selected_kinds=kinds)


class NativeProbe:
    def __init__(self, checkpoint, device="cpu", threads=8):
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        torch.set_num_threads(threads)
        sys.path[:0] = [str(ROOT / "policies/openpi/src"), str(ROOT / "policies/openpi/packages/openpi-client/src")]
        try:
            from transformers.models.siglip.check import check_whether_transformers_replace_is_installed_correctly
            if not check_whether_transformers_replace_is_installed_correctly():
                raise ImportError("OpenPI Transformers patch/version check failed")
        except ImportError as exc:
            raise RuntimeError("Run the native probe with policies/openpi/.venv/bin/python; shared environments are not modified") from exc
        from preparations.idea2.experiments import load_backend
        started = time.perf_counter()
        self.backend = load_backend(str(checkpoint), device=device, model_config="pi05_libero")
        self.model = self.backend.model.eval().requires_grad_(False)
        self.lm = self.model.paligemma_with_expert.paligemma.language_model
        self.load_seconds = time.perf_counter() - started
        transforms = self.backend.policy._input_transform.transforms
        self.tokenizer = next(t.tokenizer._tokenizer for t in transforms if type(t).__name__ == "TokenizePrompt")

    def prepare(self, observation_path):
        from preparations.idea2.experiments import native_raw
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
        with np.load(observation_path, allow_pickle=False) as archive:
            instruction = str(archive["instruction"].item())
            raw = {k: archive[k].copy() for k in archive.files if k not in ("instruction", "seed")}
        prepared = self.backend.prepare(native_raw(dict(obs=raw, instruction=instruction)))
        with torch.inference_mode():
            embedded, valid, att = self.model.embed_prefix(prepared.images, prepared.image_masks,
                                                         prepared.tokens, prepared.token_masks)
            positions = torch.cumsum(valid, dim=1) - 1
            attention = self.model._prepare_attention_masks_4d(make_att_2d_masks(valid, att))
        layout=sampling_layout(prepared,instruction,embedded.shape[1],self.tokenizer,self.model.config.discrete_state_input)
        if not bool(valid[0, layout['selected_positions']].all()):
            raise ValueError("Selected an invalid/padded token")
        return embedded, attention, positions, layout

    @torch.inference_mode()
    def full(self, embedded, attention, positions, selected=None):
        captures, handles = {}, []
        if selected is not None:
            def hook(index):
                def capture(module, args, output):
                    hidden = output[0]
                    captures[index] = hidden[:, selected].detach().cpu().clone()
                    if index == 5:
                        captures["full_L5"] = hidden.detach().cpu().clone()
                return capture
            handles = [self.lm.layers[index].register_forward_hook(hook(index)) for index in (5, 16)]
        try:
            result = self.lm(inputs_embeds=embedded, attention_mask=attention, position_ids=positions,
                             use_cache=False, return_dict=True)
            final = result.last_hidden_state.detach().cpu().clone()
        finally:
            for handle in handles:
                handle.remove()
        return final, captures

    @contextmanager
    def skip_ffn(self, skipped):
        original = {}
        try:
            for index in skipped:
                module = self.lm.layers[index].mlp
                original[index] = module.forward
                module.forward = lambda x: torch.zeros_like(x)
            yield
        finally:
            for index, forward in original.items():
                self.lm.layers[index].mlp.forward = forward

    @torch.inference_mode()
    def tail(self, hidden5, attention, positions, selected, skipped=()):
        hidden = hidden5.to(self.backend.device)
        rotary = self.lm.rotary_emb(hidden, positions)
        cache_positions = torch.arange(hidden.shape[1], device=hidden.device)
        with self.skip_ffn(skipped):
            for index in range(6, 17):
                hidden = self.lm.layers[index](hidden, attention_mask=attention, position_ids=positions,
                                             past_key_value=None, use_cache=False, cache_position=cache_positions,
                                             position_embeddings=rotary, adarms_cond=None)[0]
        return hidden[:, selected].detach().cpu().clone()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="shared/checkpoints/openpi/pi05_libero_pytorch_idea2")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(json.dumps(dict(event="loading_native_pi05", device=args.device)), flush=True)
    probe = NativeProbe(Path(args.checkpoint).resolve(), args.device, args.threads)
    print(json.dumps(dict(event="loaded", seconds=probe.load_seconds)), flush=True)
    embedded, attention, positions, layout = probe.prepare(args.input)
    # Validate metadata serialization before expensive native forwards.
    json.dumps(layout)
    timings = {}
    for name, selected in [("original", None), ("read_only_capture", layout["selected_positions"])]:
        start = time.perf_counter()
        final, captures = probe.full(embedded, attention, positions, selected)
        timings[name] = time.perf_counter() - start
        if selected is None:
            original = final
        else:
            hooked = final
        print(json.dumps(dict(event=name, seconds=timings[name], shape=list(final.shape))), flush=True)
    # Preserve completed captures even if a subsequent stage is interrupted.
    torch.save(dict(hidden5=captures["full_L5"], attention=attention.cpu(), positions=positions.cpu(),
                    selected=layout["selected_positions"], kinds=layout["selected_kinds"], L5=captures[5], L16=captures[16]),
               output / "native_preflight_tensors.pt")
    (output / "native_capture_check.json").write_text(json.dumps(dict(
        noop_equal=bool(torch.equal(original, hooked)), noop_max_abs=float((original-hooked).abs().max()),
        layout=layout, timings=timings, input=str(Path(args.input).resolve()),
        checkpoint=str(Path(args.checkpoint).resolve())), indent=2))
    start = time.perf_counter()
    tail = probe.tail(captures["full_L5"], attention, positions, layout["selected_positions"])
    timings["full_tail"] = time.perf_counter() - start
    receipt = dict(device=args.device, checkpoint=str(Path(args.checkpoint).resolve()), input=str(Path(args.input).resolve()),
                   input_sha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
                   torch=torch.__version__, model_config=repr(probe.model.config),
                   layout=layout, load_s=probe.load_seconds, timings=timings,
                   noop_equal=bool(torch.equal(original, hooked)), noop_max_abs=float((original-hooked).abs().max()),
                   tail_equal=bool(torch.equal(captures[16], tail)), tail_max_abs=float((captures[16]-tail).abs().max()),
                   scope="CPU/source-path and split-tail check; no quality or GPU timing claim")
    (output / "native_preflight.json").write_text(json.dumps(receipt, indent=2))
    if not receipt["noop_equal"] or not receipt["tail_equal"]:
        raise ValueError("Native capture or split-tail mismatch; diagnose before continuing")
    print(json.dumps(dict(event="native_preflight_complete", **receipt)), flush=True)


if __name__ == "__main__":
    main()
