"""Recompute cached prompt tensors from the trusted converted CPU checkpoint.

No model forward, environment, GPU context, cache write or checkpoint change.
Compare both the20 LIBERO90 instructions and previously evaluated suite caches.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import hashlib
import json
import math
from pathlib import Path
import pickle
import time

import sentencepiece
import torch

ROOT = Path('/root/nfs/embodied/benchmark')
BASE = ROOT / 'shared/checkpoints/realtime-vla-flash/converted'
OUTPUT = Path(__file__).resolve().parent
ARTIFACTS = (('libero_spatial','draft_spatial'), ('libero_object','draft_object'),
             ('libero_goal','draft'), ('libero_10','draft_10'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.time()
    manifest = json.loads((ROOT/'results/ease_insights_20261103/vlm_within_suite_v1/discovery/manifest.json').read_text())
    wanted = {x['benchmark_instruction'] for x in manifest['task_assets']}
    checkpoint_path = BASE/'base/base_weights.pkl'
    print(json.dumps(dict(phase='loading_trusted_CPU_checkpoint', path=str(checkpoint_path))), flush=True)
    with checkpoint_path.open('rb') as stream:
        checkpoint = pickle.load(stream)
    weight = checkpoint['language_embedding_weight']
    assert weight.device.type == 'cpu' and weight.shape == (257152,2048) and weight.dtype == torch.bfloat16
    tokenizer_path = ROOT/'shared/checkpoints/openpi/big_vision/paligemma_tokenizer.model'
    tokenizer = sentencepiece.SentencePieceProcessor(model_proto=tokenizer_path.read_bytes())
    assert tokenizer.get_piece_size() == weight.shape[0]
    rows, manifests = [], []
    with torch.no_grad():
        for suite, artifact in ARTIFACTS:
            path = BASE/artifact/'manifest.json'
            saved = json.loads(path.read_text())['prompts']
            manifests.append(dict(suite=suite,path=str(path),sha256=sha(path),prompts=len(saved)))
            for prompt, entry in sorted(saved.items()):
                clean = prompt.strip().replace('_',' ').replace('\n',' ')
                tokens = (tokenizer.encode(clean,add_bos=True)+tokenizer.encode('\n'))[:48]
                expected = (weight[torch.tensor(tokens,dtype=torch.long)].float()*math.sqrt(2048)).to(torch.bfloat16)
                cache = BASE/artifact/entry['embed_path']
                actual = torch.load(cache,map_location='cpu',weights_only=True)
                same_shape = expected.shape == actual.shape
                equal = same_shape and torch.equal(expected,actual)
                rows.append(dict(suite_label=suite,prompt=prompt,selected_LIBERO90_prompt=prompt in wanted,
                    path=str(cache),sha256=sha(cache),shape=list(actual.shape),token_count=len(tokens),
                    token_ids_sha256=hashlib.sha256(torch.tensor(tokens,dtype=torch.int64).numpy().tobytes()).hexdigest(),
                    exactly_equal=equal,max_abs=float((expected.float()-actual.float()).abs().max()) if same_shape else None))
    selected = [x for x in rows if x['selected_LIBERO90_prompt']]
    report = dict(schema='base-embedding-cache-CPU-parity-v1', started=started,finished=time.time(),
        checkpoint=dict(path=str(checkpoint_path),expected_sha256=manifest['checkpoints']['pi0']['sha256'],
                        binary_rehashed=False,all_loaded_storage_device_types=sorted({str(x.device.type) for x in checkpoint.values() if isinstance(x,torch.Tensor)})),
        tokenizer=dict(path=str(tokenizer_path),sha256=sha(tokenizer_path)),
        method='LocalSentencePiece: clean, BOS+prompt+newline, truncate48; CPU BF16 weight lookup -> FP32 sqrt2048 -> BF16',
        scope='Embedding-input correctness only; does not validate training coverage, vision/action arithmetic, normalization provenance or closed-loop success',
        cache_manifests=manifests,cache_entries=len(rows),all_entries_exact=all(x['exactly_equal'] for x in rows),
        selected_LIBERO90_unique_prompts=len(wanted),selected_entries_checked=len(selected),
        selected_all_exact=bool(selected) and all(x['exactly_equal'] for x in selected),
        selected_prompts_fully_covered={x['prompt'] for x in selected}==wanted,
        comparisons=rows,cache_or_weights_modified=False,model_forward_or_GPU_context_started=False,
        current432_scientific_results_modified=False,script_sha256=sha(__file__))
    OUTPUT.mkdir(parents=True,exist_ok=True)
    target=OUTPUT/'prompt_embedding_parity.json';target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(report=str(target),cache_entries=report['cache_entries'],all_entries_exact=report['all_entries_exact'],
                         selected_unique_prompts=len(wanted),selected_all_exact=report['selected_all_exact'],
                         selected_prompts_fully_covered=report['selected_prompts_fully_covered'])),flush=True)


if __name__ == '__main__':
    main()
