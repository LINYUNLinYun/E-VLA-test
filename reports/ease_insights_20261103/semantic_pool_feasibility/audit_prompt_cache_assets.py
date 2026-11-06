"""Read-only prompt/cache audit; no new embedding, model, or environment execution.

Only the JSON report beside this script is written. Run with
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 and the FLASH interpreter.
"""
import ast
from collections import Counter, defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import pickletools


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/"shared/checkpoints/realtime-vla-flash/converted"
SUITE_ARTIFACTS = [("libero_spatial", "draft_spatial"), ("libero_object", "draft_object"),
                   ("libero_goal", "draft"), ("libero_10", "draft_10")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SkippedBytes(bytes):
    def __new__(cls, size, header):
        value = super().__new__(cls, b"")
        value.size, value.header = size, header
        return value

    def __len__(self):
        return self.size


class MetadataReader:
    """Let pickletools read opcodes while seeking over large binary payloads."""
    def __init__(self, raw):
        self.raw, self.skipped = raw, 0

    def read(self, size=-1):
        if size >= 1024*1024:
            header = self.raw.read(2048)
            self.raw.seek(size-len(header), 1)
            self.skipped += size-len(header)
            return SkippedBytes(size, header)
        return self.raw.read(size)

    def readline(self):
        return self.raw.readline()

    def tell(self):
        return self.raw.tell()


def checkpoint_metadata(path):
    blocks, strings, embedding_ops = [], [], []
    recording = False
    with path.open("rb") as raw:
        reader = MetadataReader(raw)
        for opcode, arg, position in pickletools.genops(reader):
            if isinstance(arg, str):
                strings.append(arg)
            if arg == "language_embedding_weight":
                recording = True
            if recording:
                value = {"binary_bytes":len(arg)} if isinstance(arg, bytes) else arg
                embedding_ops.append(dict(position=position, opcode=opcode.name, argument=value))
            if opcode.name in ("BINBYTES", "BINBYTES8", "SHORT_BINBYTES"):
                stream = io.BytesIO(arg.header if isinstance(arg, SkippedBytes) else arg[:2048])
                header_values = []
                for _ in range(5):
                    for _, value, _ in pickletools.genops(stream):
                        if isinstance(value, str):
                            header_values.append(value)
                blocks.append(dict(position=position, bytes=len(arg),
                    storage_types=[x for x in header_values if x.startswith("torch ") and x.endswith("Storage")],
                    locations=[x for x in header_values if x == "cpu" or x.startswith("cuda")]))
        stopped_at, skipped = reader.tell(), reader.skipped
    assert stopped_at == path.stat().st_size and "language_embedding_weight" in strings
    assert blocks and all(b["locations"] == ["cpu"] and b["storage_types"] == ["torch BFloat16Storage"] for b in blocks)
    # Verify the shape tuple in the serialized final language embedding entry.
    shape_ops = [(x["opcode"], x["argument"]) for x in embedding_ops]
    assert any(shape_ops[i:i+3] == [("BININT",257152),("BININT2",2048),("TUPLE2",None)]
               for i in range(len(shape_ops)-2))
    return dict(path=str(path), file_size_bytes=path.stat().st_size, storage_blocks=blocks,
        all_storage_locations_cpu=True, all_storage_types_bfloat16=True,
        language_embedding_weight=dict(present=True, shape=[257152,2048], dtype="torch.bfloat16", device="cpu",
                                       serialized_entry_opcodes=embedding_ops),
        skipped_binary_payload_bytes=skipped, actual_weight_deserialization=False)


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Explicit CPU-only audit required"
    import torch
    import sentencepiece
    official = ROOT/"LIBERO/libero/libero/benchmark"
    task_map_path, benchmark_path = official/"libero_suite_task_map.py", official/"__init__.py"
    task_map = ast.literal_eval(ast.parse(task_map_path.read_text()).body[0].value)
    language_function = next(n for n in ast.parse(benchmark_path.read_text()).body
                             if isinstance(n, ast.FunctionDef) and n.name == "grab_language_from_filename")
    scope = {}
    exec(compile(ast.Module(body=[language_function], type_ignores=[]), "official-language-function", "exec"), scope)
    tasks = [dict(task=i, name=name, prompt=scope["grab_language_from_filename"](name+".bddl"))
             for i, name in enumerate(task_map["libero_90"])]
    prompts = {r["prompt"] for r in tasks}
    tokenizers = [ROOT/"shared/checkpoints/openpi/big_vision/paligemma_tokenizer.model",
                  Path("/root/.cache/openpi/big_vision/paligemma_tokenizer.model")]
    assert len({sha(p) for p in tokenizers}) == 1
    processor = sentencepiece.SentencePieceProcessor(model_proto=tokenizers[0].read_bytes())
    groups, routing, artifacts = defaultdict(list), {}, []
    for suite, artifact in SUITE_ARTIFACTS:
        manifest_path = BASE/artifact/"manifest.json"
        manifest = json.loads(manifest_path.read_text())["prompts"]
        artifacts.append(dict(suite_label=suite, artifact=artifact, manifest_path=str(manifest_path),
            manifest_sha256=sha(manifest_path), prompt_count=len(manifest), libero90_matching=len(set(manifest)&prompts)))
        for prompt, entry in manifest.items():
            path = BASE/artifact/entry["embed_path"]
            tensor = torch.load(path, map_location="cpu", weights_only=True)
            assert isinstance(tensor, torch.Tensor) and tensor.ndim == 2 and tensor.shape[1] == 2048
            assert tensor.shape[0] == entry["prompt_len"] and tensor.dtype == torch.bfloat16 and torch.isfinite(tensor).all()
            cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
            tokens = processor.encode(cleaned, add_bos=True) + processor.encode("\n")
            if prompt in prompts:
                assert len(tokens) < 48 and len(tokens) == len(tensor)
            record = dict(suite_label=suite, artifact=artifact, path=str(path), sha256=sha(path),
                shape=list(tensor.shape), dtype=str(tensor.dtype), device=str(tensor.device), all_finite=True,
                tokenizer_length=len(tokens), local_tokenizer_length_matches=len(tokens[:48]) == len(tensor))
            groups[prompt].append((record, tensor))
            routing.setdefault(prompt, suite)
    assert prompts <= set(groups) and {routing[p] for p in prompts} == {"libero_goal"}
    comparisons = []
    for prompt, entries in groups.items():
        for index, (left, a) in enumerate(entries):
            for right, b in entries[index+1:]:
                same_shape = a.shape == b.shape
                comparisons.append(dict(prompt=prompt, artifacts=[left["artifact"],right["artifact"]],
                    same_shape=same_shape, exactly_equal=torch.equal(a,b),
                    max_abs=float((a.float()-b.float()).abs().max()) if same_shape else None))
    assert all(row["exactly_equal"] for row in comparisons)
    checkpoint = checkpoint_metadata(BASE/"base/base_weights.pkl")
    assert processor.get_piece_size() == checkpoint["language_embedding_weight"]["shape"][0]
    report = dict(schema="ease-libero90-prompt-cache-readonly-audit-v1",
        outcomes_read=False, cache_generated=False, checkpoint_or_frozen_sources_modified=False,
        environment_constructed=False, gpu_launched=False,
        assets=artifacts, official_tasks=tasks, libero90_distinct_prompts=len(prompts),
        existing_cache_coverage=len(prompts & set(groups)), missing_prompts=sorted(prompts-set(groups)),
        native_prompt_routing_counts=dict(Counter(routing[p] for p in prompts)),
        cache_entries=sum(len(v) for v in groups.values()), unique_cached_prompts=len(groups),
        cache_entries_all_valid=True, cross_cache_shared_prompt_groups=sum(len(v)>1 for v in groups.values()),
        cross_cache_pairwise_comparisons=comparisons,
        cross_cache_all_exactly_equal=True, cross_cache_max_abs=max(c["max_abs"] for c in comparisons),
        tokenizer=dict(files=[dict(path=str(p), sha256=sha(p), size_bytes=p.stat().st_size) for p in tokenizers],
            piece_count=processor.get_piece_size(), libero90_token_lengths=sorted({entries[0][0]["tokenizer_length"] for p,entries in groups.items() if p in prompts}),
            truncation_required_for_libero90=False), checkpoint_metadata=checkpoint,
        cache_inventory={prompt:[r for r,_ in entries] for prompt,entries in groups.items()},
        interpretation=[
            "Current official LIBERO-90 has 90 tasks but 74 unique language strings; all74 already have valid existing cache entries. No new embedding cache is needed for these exact strings.",
            "Pi0Adapter loads every manifest prompt and uses first suite label in Spatial/Object/Goal/Long order; all74 map to libero_goal only as a cache lookup label.",
            "EASE fixed VLM paths use the shared Full checkpoint and residual/native sampling, not the learned Goal draft head.",
            "Cache availability and the107 prompts in draft/manifest.json do not establish the Goal head training task distribution, and do not change any released-head source identity or prior transfer comparison.",
            "For genuinely missing text, local tokenizer plus CPU embedding lookup, FP32 sqrt(2048) scaling and BF16 conversion are sufficient; no transformer/action-expert forward is needed.",
            "The existing builder uses pickle.load on the entire 6.48 GB base file, so CPU RAM is needed; serialized storage metadata confirms all 42 storages are CPU BF16.",
            "Do not infer full tokenizer/base provenance parity solely from matching lengths; future new-cache generation should pin the local tokenizer and verify overlap tensors before use.",
            "This audit does not remove frozen suite/task contracts, rollout limits or the need for a separately rendered real-input bank and GPU numerical gates.",
        ], sources=[dict(path=str(p),sha256=sha(p)) for p in (task_map_path,benchmark_path)],
        audit_script=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)))
    destination = Path(__file__).with_name("prompt_cache_asset_audit.json")
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(dict(report=str(destination), unique_libero90_prompts=len(prompts), coverage=len(prompts&set(groups)),
        cache_entries=report["cache_entries"], cross_cache_pairs=len(comparisons), all_pairs_equal=True,
        base_storages=len(checkpoint["storage_blocks"]), all_base_storages_cpu=True, cache_generated=False,gpu_launched=False)))


if __name__ == "__main__":
    main()
