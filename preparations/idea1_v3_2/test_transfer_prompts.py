"""CPU-only instruction contract tests; no model or environment startup."""
import math
import unittest

import torch

from .transfer_prompts import instruction_embedding,instruction_embedding_from_base,instruction_tokens


class InstructionEmbedding(unittest.TestCase):
    def test_native_sentencepiece_bos_cleanup_and_newline(self):
        cleaned,ids = instruction_tokens("  Open_the middle\nlayer of the drawer  ")
        self.assertEqual(cleaned,"Open the middle layer of the drawer")
        self.assertEqual(ids,[2,6047,573,7185,8651,576,573,39635,108])

    def test_overlength_rejected_not_truncated(self):
        with self.assertRaisesRegex(ValueError,"TOKEN_LIMIT_EXCEEDED"):
            instruction_tokens("Open the middle layer of the drawer",max_tokens=3)

    def test_actual_rows_and_bf16_scale(self):
        def lookup(ids):
            return torch.tensor(ids,dtype=torch.bfloat16)[:,None].expand(-1,2048).contiguous()
        result = instruction_embedding("turn on the light bulb",row_lookup=lookup)
        expected = (torch.tensor(result["token_ids"],dtype=torch.bfloat16).float()*math.sqrt(2048)).bfloat16()
        torch.testing.assert_close(result["embeddings"][:,0],expected,atol=0,rtol=0)
        self.assertEqual(result["embeddings"].shape[0],len(result["token_ids"]))

    def test_selected_nonfinite_rows_rejected(self):
        with self.assertRaisesRegex(ValueError,"NONFINITE_TOKEN_ROW"):
            instruction_embedding("test",row_lookup=lambda ids:torch.full((len(ids),2048),float("nan")))

    def test_existing_base_lookup_no_model_loading(self):
        # Minimal real tokenizer IDs include high vocabulary indices; a small
        # full-width CPU table suffices for this one short instruction.
        _,ids = instruction_tokens("a")
        base = {"language_embedding_weight":torch.zeros((max(ids)+1,2048),dtype=torch.bfloat16)}
        result = instruction_embedding_from_base("a",base)
        self.assertEqual(result["embeddings"].device.type,"cpu")
        self.assertEqual(torch.count_nonzero(result["embeddings"]).item(),0)


if __name__ == "__main__":
    unittest.main()
