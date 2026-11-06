"""CPU full-row no-truncation audit for actual parsed Plus instruction corpus."""
import argparse
import json
from pathlib import Path
import torch
from preparations.idea1_v3_2.transfer_prompts import (SparseSafeTensorRows,instruction_embedding,
    instruction_tokens,tensor_hash,TOKENIZER)
from preparations.idea1_v3_2.transfer_protocol import sha


def audit(corpus):
    instructions=json.loads(Path(corpus).read_text())
    if not isinstance(instructions,list) or not all(isinstance(s,str) for s in instructions):
        raise ValueError("Actual instruction string list required")
    lookup=SparseSafeTensorRows();rows=[]
    for index,instruction in enumerate(instructions):
        try:
            result=instruction_embedding(instruction,row_lookup=lookup)
            rows.append(dict(index=index,instruction=instruction,passed=True,token_ids=result["token_ids"],
                token_count=len(result["token_ids"]),embedding_shape=list(result["embeddings"].shape),
                all_rows_finite=bool(torch.isfinite(result["embeddings"].float()).all()),
                rows_sha256=result["rows_sha256"],embedding_sha256=tensor_hash(result["embeddings"])))
        except Exception as exc:
            rows.append(dict(index=index,instruction=instruction,passed=False,error=repr(exc)))
    return dict(status="complete",passed=all(row["passed"] for row in rows),cpu_only=True,model_constructed=False,gpu_used=False,
        instruction_count=len(rows),distinct_instruction_count=len(set(instructions)),
        max_token_count=max((row.get("token_count",0) for row in rows),default=0),max_allowed_tokens=48,
        source_sha256={str(Path(__file__)):sha(__file__),"preparations/idea1_v3_2/transfer_prompts.py":sha("preparations/idea1_v3_2/transfer_prompts.py")},
        corpus=dict(path=str(corpus),sha256=sha(corpus)),tokenizer=dict(path=str(TOKENIZER),sha256=sha(TOKENIZER)),
        table=dict(path=str(lookup.path),key=lookup.key,header_sha256=lookup.header_sha256,shape=lookup.entry["shape"]),
        rows=rows,scope="Every actual instruction token and BF16 row finite; no nearest instruction or truncation. Sparse CPU table proof alone is not model-policy parity.")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--corpus",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();output=Path(a.output)
    if output.exists(): raise FileExistsError(output)
    result=audit(a.corpus);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x") as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({key:result[key] for key in ("passed","instruction_count","distinct_instruction_count","max_token_count")}))


if __name__=="__main__":main()
