"""Target text-capability smoke tests; these are measurements, not asserted accuracy."""
__test__ = False

from pathlib import Path

from .common import (append_jsonl, cli_parser, control_observation, load_config, make_backends,
                     prepare_run, select_rows, target_prompt, write_json)
from .data import load_observation
from .workflow import decode


def main():
    parser = cli_parser(__doc__)
    parser.add_argument("--hf-reference", help="Optional local/HF PaliGemma task-positive checkpoint")
    parser.add_argument("--reference-device", default="cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    rows = select_rows(config, args.limit_groups)
    source, target = make_backends(config)
    out, run = prepare_run(config, rows, source, target, command="text_smoke")
    path = out / "text_decode_smoke.jsonl"
    if path.exists():
        raise FileExistsError(f"Preserve previous measurements; choose a new output_dir: {path}")
    reference = None
    if args.hf_reference:
        import torch
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
        processor = AutoProcessor.from_pretrained(args.hf_reference)
        model = PaliGemmaForConditionalGeneration.from_pretrained(args.hf_reference).to(args.reference_device).eval()
        reference = (torch, processor, model)
    for row in rows:
        obs = load_observation(row)
        for probe in config["probes"]:
            if probe["task_type"] != row["task_type"]:
                continue
            prompt = target_prompt(row, probe)
            for condition, image_obs in [("direct", obs)] + [
                (c["id"], control_observation(obs, c)) for c in config["controls"]
            ]:
                prefix = target.prepare_target(image_obs, prompt)
                record = {"sample_id": row["sample_id"], "probe_id": probe["id"], "split": row["split"],
                          "task_type": row["task_type"], "label": row["label"], "labels": row["labels"],
                          "condition": condition, "question": prompt, "config_hash": run["config_hash"],
                          **decode(target, prefix, row, config)}
                append_jsonl(path, record)
            if reference:
                from PIL import Image
                torch, processor, model = reference
                inputs = processor(text=prompt, images=Image.fromarray(obs["observation/image"]), return_tensors="pt").to(args.reference_device)
                with torch.inference_mode():
                    output = model.generate(**inputs, max_new_tokens=config["max_new_tokens"], do_sample=False)
                token_ids = output[0, inputs["input_ids"].shape[1]:].tolist()
                append_jsonl(path, {"sample_id": row["sample_id"], "probe_id": probe["id"],
                    "condition": "external_task_positive", "checkpoint": args.hf_reference,
                    "camera": "observation/image", "question": prompt, "token_ids": token_ids,
                    "raw_generation": processor.decode(token_ids, skip_special_tokens=True),
                    "note": "Single-agent-view task reference; not verification of the custom decoder."})
            print(f"decoded {row['sample_id']} / {probe['id']}", flush=True)
    write_json(out / "text_decode_smoke_identity.json", run)
    print(path)


if __name__ == "__main__":
    main()
