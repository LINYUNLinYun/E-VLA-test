"""CPU-only native instruction tokenization and legacy BF16 embedding lookup.

This builds language input rows, not a model or control policy. Unsupported
lengths fail explicitly; instructions are never substituted or truncated.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import time
import traceback

import numpy as np
import sentencepiece
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
TOKENIZER = ROOT/"shared/checkpoints/openpi/big_vision/paligemma_tokenizer.model"
SAFETENSORS = ROOT/"shared/checkpoints/openpi/pi0_libero_pytorch_idea2/model.safetensors"
TABLE_KEY = "paligemma_with_expert.paligemma.lm_head.weight"
MANIFEST = ROOT/"shared/checkpoints/realtime-vla-flash/converted/draft/manifest.json"
ANNOTATIONS = ROOT/"calvin/dataset/calvin_debug_dataset/validation/lang_annotations/auto_lang_ann.npy"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while data := stream.read(8*1024*1024):
            digest.update(data)
    return digest.hexdigest()


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def instruction_tokens(instruction, *, tokenizer_path=TOKENIZER, max_tokens=48):
    if not isinstance(instruction,str) or not instruction.strip():
        raise ValueError("EMPTY_INSTRUCTION: a nonempty instruction is required")
    processor = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_path))
    cleaned = instruction.strip().replace("_"," ").replace("\n"," ")
    ids = processor.encode(cleaned,add_bos=True)+processor.encode("\n")
    if len(ids)>max_tokens:
        raise ValueError(f"TOKEN_LIMIT_EXCEEDED: {len(ids)} > {max_tokens}; no truncation permitted")
    return cleaned,ids


def instruction_embedding(instruction, *, embedding_weight=None, row_lookup=None,
                          tokenizer_path=TOKENIZER, max_tokens=48):
    """Return all valid BF16 rows; accept a CPU table or sparse row callback.

    Mirrors released legacy converted prompts: cast table rows to BF16, scale
    in FP32 by sqrt(2048), then round once to BF16. No padding rows are emitted.
    """
    if (embedding_weight is None)==(row_lookup is None):
        raise ValueError("Provide exactly one CPU embedding table or row_lookup")
    cleaned,ids = instruction_tokens(instruction,tokenizer_path=tokenizer_path,max_tokens=max_tokens)
    if embedding_weight is not None:
        table = torch.as_tensor(embedding_weight)
        if table.device.type!="cpu":
            raise ValueError("CPU_ONLY: embedding table must already reside on CPU")
        if table.ndim!=2 or table.shape[1]!=2048:
            raise ValueError("Language table must have shape [vocabulary,2048]")
        if min(ids)<0 or max(ids)>=table.shape[0]:
            raise ValueError("TOKEN_OUT_OF_RANGE: no fallback row or substitution permitted")
        rows = table[torch.tensor(ids,dtype=torch.long)]
    else:
        rows = row_lookup(ids)
    if rows.device.type!="cpu" or tuple(rows.shape)!=(len(ids),2048):
        raise ValueError("CPU_ONLY: row lookup must return every token row with width2048")
    rows = rows.detach().to(torch.bfloat16).contiguous()
    if not torch.isfinite(rows.float()).all():
        raise ValueError("NONFINITE_TOKEN_ROW")
    embeds = (rows.float()*math.sqrt(2048)).to(torch.bfloat16).contiguous()
    if not torch.isfinite(embeds.float()).all():
        raise ValueError("NONFINITE_EMBEDDING")
    return dict(instruction=instruction,cleaned=cleaned,token_ids=ids,
                embeddings=embeds,rows_sha256=tensor_hash(rows))


def instruction_embedding_from_base(instruction, base, **kwargs):
    """Use an existing CPU base dictionary without loading a model or moving GPU data."""
    return instruction_embedding(instruction,embedding_weight=base["language_embedding_weight"],**kwargs)


class SparseSafeTensorRows:
    """Only requested rows are materialized; the 6.6GB checkpoint is never loaded."""
    def __init__(self,path=SAFETENSORS,key=TABLE_KEY):
        self.path,self.key = Path(path),key
        with self.path.open("rb") as stream:
            prefix = stream.read(8)
            length = struct.unpack("<Q",prefix)[0]
            raw = stream.read(length)
        self.header = json.loads(raw)
        self.entry = self.header[key]
        self.data_start = 8+length
        self.header_sha256 = hashlib.sha256(prefix+raw).hexdigest()
        if self.entry["dtype"]!="BF16" or self.entry["shape"][1]!=2048:
            raise ValueError("Expected BF16 language table width2048")
        self.cache = {}

    def __call__(self,ids):
        if min(ids)<0 or max(ids)>=self.entry["shape"][0]:
            raise ValueError("TOKEN_OUT_OF_RANGE")
        with safe_open(str(self.path),framework="pt",device="cpu") as handle:
            table = handle.get_slice(self.key)
            for index in set(ids)-self.cache.keys():
                self.cache[index] = table[index:index+1,:].clone()
        return torch.cat([self.cache[index] for index in ids],dim=0)

    def identity(self):
        # Hash just the complete embedding-table BF16 byte range in streaming
        # chunks; this does not materialize the table or load other model data.
        start,end = self.entry["data_offsets"]
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            stream.seek(self.data_start+start)
            remaining = end-start
            while remaining:
                data = stream.read(min(8*1024*1024,remaining))
                if not data:
                    raise EOFError("Incomplete safetensors table")
                digest.update(data)
                remaining -= len(data)
        return dict(path=str(self.path),file_size_bytes=self.path.stat().st_size,
                    key=self.key,header_sha256=self.header_sha256,table=self.entry,
                    table_payload_sha256=digest.hexdigest(),
                    hash_scope="Exact complete table BF16 payload range, streamed; not a full checkpoint hash")


def summary(result):
    embeds = result["embeddings"]
    return {k:v for k,v in result.items() if k!="embeddings"}|dict(
        shape=list(embeds.shape),dtype=str(embeds.dtype),device=str(embeds.device),
        all_token_rows_finite=bool(torch.isfinite(embeds.float()).all()),
        embedding_sha256=tensor_hash(embeds),token_ids_sha256=hashlib.sha256(
            np.asarray(result["token_ids"],dtype="<i8").tobytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--table",default=str(SAFETENSORS))
    parser.add_argument("--key",default=TABLE_KEY)
    parser.add_argument("--tokenizer",default=str(TOKENIZER))
    parser.add_argument("--manifest",default=str(MANIFEST))
    parser.add_argument("--annotations",default=str(ANNOTATIONS))
    parser.add_argument("--released-count",type=int,default=10)
    parser.add_argument("--launch",action="store_true")
    args = parser.parse_args()
    if not args.launch:
        print(json.dumps(dict(dry_run=True,cpu_only=True,**vars(args))))
        return
    if args.released_count<5:
        raise ValueError("At least five released prompt comparisons are required")
    directory = Path(args.output)
    directory.mkdir(parents=True,exist_ok=False)
    receipt = directory/"preflight.json"
    report = dict(status="running",started_unix=time.time(),cpu_only=True,
        scope="Instruction token rows and converted BF16 prompt embeddings only; not native model/policy parity, action semantics or environment compatibility",
        normalization="Native pi0 strip, underscores/newlines to spaces; SentencePiece add_bos then separately encode newline; full valid rows only",
        no_truncation=True,no_instruction_substitution=True,model_constructed=False,gpu_used=False,
        source_sha256={str(Path(__file__)):sha(__file__)},
        tokenizer=dict(path=args.tokenizer,sha256=sha(args.tokenizer)),
        inputs={p:sha(p) for p in (args.manifest,args.annotations)},released_checks=[],calvin_checks=[])
    def save():
        receipt.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    save()
    try:
        table = SparseSafeTensorRows(args.table,args.key)
        report["table_identity"] = table.identity()
        manifest = json.loads(Path(args.manifest).read_text())
        for prompt,entry in list(manifest["prompts"].items())[:args.released_count]:
            generated = instruction_embedding(prompt,row_lookup=table,tokenizer_path=args.tokenizer)
            path = Path(args.manifest).parent/entry["embed_path"]
            expected = torch.load(path,map_location="cpu",weights_only=True)
            if not isinstance(expected,torch.Tensor):
                raise TypeError("Released prompt file must contain a tensor")
            actual = generated["embeddings"]
            shapes_match = actual.shape==expected.shape
            row = summary(generated)|dict(released_path=str(path),released_sha256=sha(path),
                released_shape=list(expected.shape),manifest_prompt_len=entry["prompt_len"],
                bitwise_equal=bool(shapes_match and actual.dtype==expected.dtype and torch.equal(actual,expected)),
                max_abs=float((actual.float()-expected.float()).abs().max()) if shapes_match else None)
            report["released_checks"].append(row)
            save()
        annotations = np.load(args.annotations,allow_pickle=True).item()
        prompts = annotations["language"]["ann"]
        if len(prompts)!=8:
            raise ValueError("Expected all8 actual CALVIN debug validation annotations")
        for index,(task,prompt) in enumerate(zip(annotations["language"]["task"],prompts)):
            generated = instruction_embedding(str(prompt),row_lookup=table,tokenizer_path=args.tokenizer)
            path = directory/f"calvin_{index:02d}.pt"
            torch.save(generated["embeddings"],path)
            report["calvin_checks"].append(summary(generated)|dict(annotation_index=index,
                task=str(task),annotation_frame_range=np.asarray(annotations["info"]["indx"][index]).tolist(),
                output_path=str(path),output_sha256=sha(path)))
            save()
        report["passed"] = all(r["bitwise_equal"] for r in report["released_checks"]) and all(
            r["all_token_rows_finite"] for r in report["calvin_checks"])
        report.update(status="complete" if report["passed"] else "mismatch_preserved",
            completed_unix=time.time(),failure_code=None if report["passed"] else "RELEASED_EMBEDDING_MISMATCH",
            distinct_table_rows_materialized=len(table.cache))
        if not report["passed"]:
            raise ArithmeticError("Released embedding mismatch: receipt preserved; do not infer policy compatibility")
    except Exception:
        report.setdefault("failure_code","CPU_PREFLIGHT_TECHNICAL")
        report.update(status="mismatch_preserved" if report.get("failure_code")=="RELEASED_EMBEDDING_MISMATCH" else "failed",
                      completed_unix=time.time(),error=traceback.format_exc())
        raise
    finally:
        save()
    print(json.dumps(dict(passed=report["passed"],released_prompts=len(report["released_checks"]),
        calvin_instructions=len(report["calvin_checks"]),distinct_table_rows_materialized=len(table.cache),
        receipt=str(receipt)),indent=2))


if __name__ == "__main__":
    main()
