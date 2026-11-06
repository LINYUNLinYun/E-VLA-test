"""Read-only shape/key audit using the actual official conversion functions.

Loads Orbax metadata only and evaluates conversions on symbolic array shapes;
does not load either full model, allocate model tensors, or access a GPU.
This establishes mapping coverage and shapes, not value parity with JAX.
"""
import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace


class Shape:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def transpose(self, *axes):
        axes = axes or tuple(reversed(range(len(self.shape))))
        return Shape(tuple(self.shape[i] for i in axes))

    def reshape(self, *shape):
        shape = list(shape)
        if -1 in shape:
            missing = shape.index(-1)
            shape[missing] = math.prod(self.shape) // math.prod(x for x in shape if x != -1)
        if math.prod(shape) != math.prod(self.shape):
            raise ValueError("Invalid symbolic reshape")
        return Shape(shape)

    def __getitem__(self, selection):
        selection = selection if isinstance(selection, tuple) else (selection,)
        return Shape([size for index, size in enumerate(self.shape)
                      if index >= len(selection) or not isinstance(selection[index], int)])


def main():
    os.environ["JAX_PLATFORMS"] = "cpu"
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Original checkpoint root containing params/")
    parser.add_argument("--converted", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import orbax.checkpoint as ocp
    from safetensors import safe_open
    import openpi.models.gemma as gemma
    root = Path(__file__).resolve().parents[2]
    source = root / "policies/openpi/examples/convert_jax_model_to_pytorch.py"
    spec = importlib.util.spec_from_file_location("official_idea2_conversion_audit", source)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    # Local module binding only; does not monkeypatch global torch functions.
    converter.torch = SimpleNamespace(from_numpy=lambda x: x, Tensor=Shape)
    metadata = ocp.PyTreeCheckpointer().metadata(str(Path(args.source).resolve() / "params"))
    tree = metadata["params"] if "params" in metadata else metadata
    def flatten(value, prefix=""):
        result = {}
        for key, item in value.items():
            path = f"{prefix}/{key}" if prefix else key
            if isinstance(item, dict):
                result.update(flatten(item, path))
            else:
                result[path] = Shape(item.shape)
        return result
    flat = flatten(tree)
    pali = {key.removeprefix("PaliGemma/"): value for key, value in flat.items() if key.startswith("PaliGemma/")}
    cfg = SimpleNamespace(vision_config=SimpleNamespace(hidden_size=1152, num_hidden_layers=27),
                          text_config=SimpleNamespace(hidden_size=2048, num_hidden_layers=18,
                                                      num_attention_heads=8, head_dim=256))
    vlm, expert = converter.slice_paligemma_state_dict(pali, cfg)
    expert_cfg = SimpleNamespace(**dataclasses.asdict(gemma.get_config("gemma_300m")))
    ae = converter.slice_gemma_state_dict(expert, expert_cfg, num_expert=1,
                                          checkpoint_dir=str(args.source), pi05=False)
    expected = {**vlm, **ae}
    for group in ("state_proj", "action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out"):
        for kind in ("kernel", "bias"):
            found = flat.get(f"{group}/{kind}/value", flat.get(f"{group}/{kind}"))
            if found is None:
                raise ValueError(f"Missing source parameter {group}/{kind}")
            expected[f"{group}.{'weight' if kind == 'kernel' else 'bias'}"] = found.transpose() if kind == "kernel" else found
    with safe_open(str(Path(args.converted) / "model.safetensors"), framework="pt", device="cpu") as weights:
        actual = {key: tuple(weights.get_slice(key).get_shape()) for key in weights.keys()}
        aliases = weights.metadata() or {}
    missing = [key for key in expected if key not in actual and aliases.get(key) not in actual]
    mismatches = {key: dict(source=list(value.shape), converted=list(actual.get(key, actual.get(aliases.get(key), ()))))
                  for key, value in expected.items() if key not in missing and
                  value.shape != actual.get(key, actual.get(aliases.get(key), ()))}
    unused = [key for key in actual if key not in expected]
    unexpected = [key for key in unused if ".lm_head." not in key]
    result = dict(passed=not missing and not mismatches and not unexpected,
                  expected_trained_tensors=len(expected), saved_tensors=len(actual), missing_trained_keys=missing,
                  shape_mismatches=mismatches, saved_unmapped_keys=unused, unexpected_keys=unexpected,
                  saved_aliases=aliases, official_converter_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  scope="Metadata proves trained tensor mapping coverage/shapes; does not establish numeric JAX/PyTorch parity")
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
