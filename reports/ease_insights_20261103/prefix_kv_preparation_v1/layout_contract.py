"""CPU-only buffer/index contracts; no model, CUDA or executable policy."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Layout:
    pattern: str
    original_prefix: int
    packed_prefix: int
    keep_indices: tuple
    decoder_absolute_positions: tuple
    decoder_storage_positions: tuple
    kv_shape: tuple
    attention_shape: tuple


def plan(pattern, original_prefix, horizon=50):
    if type(original_prefix) is not int or not 512 <= original_prefix <= 973 or horizon != 50 or type(horizon) is not int:
        raise ValueError('Only the inspected two-camera Pi0 prefix and H50/51 decoder-token contract is supported')
    if pattern == 'identity':
        kept = tuple(range(original_prefix))
    elif pattern in ('center_columns', 'checkerboard'):
        visual = tuple(camera * 256 + row * 16 + column
                       for camera in range(2) for row in range(16) for column in range(16)
                       if (4 <= column < 12 if pattern == 'center_columns' else (row + column) % 2 == 0))
        kept = visual + tuple(range(512, original_prefix))
    else:
        raise ValueError('Unknown prospective prefix pattern')
    if len(set(kept)) != len(kept) or tuple(sorted(kept)) != kept:
        raise ValueError('Prefix packing must preserve original order without duplicates')
    packed = len(kept)
    decoder_tokens = horizon + 1
    return Layout(pattern, original_prefix, packed, kept,
                  tuple(range(original_prefix, original_prefix + decoder_tokens)),
                  tuple(range(packed, packed + decoder_tokens)),
                  (18, packed + decoder_tokens, 256), (decoder_tokens * 8, packed + decoder_tokens))


def validate_buffers(layout, key_shape, value_shape, attention_shape, contiguous, decoder_absolute_positions,
                     logits_shape=None, logits_contiguous=True):
    if tuple(key_shape) != layout.kv_shape or tuple(value_shape) != layout.kv_shape:
        raise ValueError('Actual K/V token dimension must equal packed prefix plus all51 decoder tokens')
    if tuple(attention_shape) != layout.attention_shape or contiguous is not True:
        raise ValueError('Attention/KV scratch must use matching token counts and contiguous storage')
    if tuple(decoder_absolute_positions) != layout.decoder_absolute_positions:
        raise ValueError('Packing storage must not shift original decoder RoPE positions')
    if logits_shape is not None and (tuple(logits_shape) != layout.attention_shape or logits_contiguous is not True):
        raise ValueError('Optional FP32 attention-logit scratch must match the packed attention layout')
    return True
