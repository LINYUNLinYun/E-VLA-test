"""CPU-only lifecycle contracts. No model, torch, environment or GPU is used."""
import ast
import sys
from types import SimpleNamespace
import unittest

from .prompt_runtime import KERNEL_PATH, PromptRuntimeOwner, prompt_buffer_shapes, verify_kernel_source


class Tensor:
    def __init__(self, shape, dtype='bfloat16', device='cuda:fake', token=None):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.token = token


def constructor_shapes(name, length):
    """Independent oracle from pinned constructor expressions, not our specs."""
    verify_kernel_source()
    tree = ast.parse(KERNEL_PATH.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Pi0Inference')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    assignment = next(n for n in init.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and
        t.value.id == 'self' and t.attr == name for t in n.targets))
    variables = dict(self=SimpleNamespace(prompt_len=length), num_views=2,
                     chunk_size=50, encoder_seq_len=512 + length, decoder_seq_len=51)
    allowed = (ast.Constant, ast.Tuple, ast.List, ast.Name, ast.Load, ast.Attribute,
               ast.BinOp, ast.Mult, ast.Add, ast.Sub)
    shapes = {}
    for key, value in zip(assignment.value.keys, assignment.value.values):
        if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Attribute) or value.func.attr != 'empty':
            raise AssertionError('Pinned constructor allocation format changed')
        dims = []
        for argument in value.args:
            if any(not isinstance(n, allowed) for n in ast.walk(argument)):
                raise AssertionError('Unexpected allocation shape expression')
            dims.append(eval(compile(ast.Expression(argument), '<pinned-shape>', 'eval'),
                             {'__builtins__': {}}, variables))
        shape = tuple(dims[0]) if len(dims) == 1 and isinstance(dims[0], (tuple, list)) else tuple(dims)
        dtype = next(k.value.attr for k in value.keywords if k.arg == 'dtype')
        shapes[key.value] = (shape, dtype)
    return shapes


def runtime(length=5, *, logits=False):
    weights = {k: Tensor(shape, dtype) for k, (shape, dtype) in constructor_shapes('weights', length).items()}
    buffers = {k: Tensor(shape, dtype) for k, (shape, dtype) in constructor_shapes('buffers', length).items()}
    if logits:
        buffers['decoder_attn_logits_buf'] = Tensor((51 * 8, 512 + length + 51), 'float32')
    return SimpleNamespace(num_views=2, chunk_size=50, prompt_len=length,
        weights=weights, buffers=buffers, infer_graph=object(), attention_logits_fp32=logits)


class Operations:
    def __init__(self, fail_at=None):
        self.allocations, self.ropes, self.captures, self.synchronizations = [], [], 0, 0
        self.fail_at = fail_at

    def synchronize(self, device):
        self.synchronizations += 1

    def allocate(self, shape, template):
        self.allocations.append(tuple(shape))
        if len(self.allocations) == self.fail_at:
            raise MemoryError('Synthetic scratch allocation failure')
        return Tensor(shape, template.dtype, template.device)

    def copy_embedding(self, destination, embedding):
        if destination.shape != embedding.shape:
            raise AssertionError('Embedding shape mismatch')
        destination.token = embedding.token

    def initialize_rope(self, target):
        self.ropes.append((512 + target.prompt_len, 51))
        target.buffers['decoder_rope_weights'].token = 512 + target.prompt_len

    def capture(self, target):
        self.captures += 1
        target.infer_graph = object()


class ResidencyContracts(unittest.TestCase):
    def test_import_does_not_load_torch(self):
        self.assertNotIn('torch', sys.modules)

    def test_specs_cover_all_constructor_prompt_dependent_allocations(self):
        before, after = constructor_shapes('buffers', 5), constructor_shapes('buffers', 9)
        changed = {k for k in before if before[k] != after[k]}
        self.assertEqual(set(prompt_buffer_shapes(9)), changed)
        for key, shape in prompt_buffer_shapes(9).items():
            self.assertEqual(shape, after[key][0])

    def test_short_long_short_retains_every_invariant_weight_and_static_buffer(self):
        target, ops = runtime(), Operations()
        owner = PromptRuntimeOwner(target, operations=ops)
        anchors = {k: v for k, v in target.weights.items() if k != 'language_embeds'}
        static = {k: v for k, v in target.buffers.items() if k not in prompt_buffer_shapes(5)}
        weight_dict, buffer_dict = target.weights, target.buffers
        caches = [{'old_mask_graph': object()}, {'old_condition_graph': object()}]
        for length in (9, 5):
            for cache in caches:
                cache['graph'] = object()
            result = owner.rebind(Tensor((length, 2048), token=str(length)),
                                  episode_boundary=True, graph_caches=caches)
            self.assertTrue(result['changed'])
            self.assertIs(target.weights, weight_dict)
            self.assertIs(target.buffers, buffer_dict)
            self.assertTrue(all(target.weights[k] is v for k, v in anchors.items()))
            self.assertTrue(all(target.buffers[k] is v for k, v in static.items()))
            self.assertTrue(all(not cache for cache in caches))
            self.assertEqual(target.buffers['decoder_rope_weights'].token, 512 + length)
            owner.assert_ready()
        self.assertEqual(ops.captures, 2)
        self.assertEqual(ops.ropes, [(521, 51), (517, 51)])

    def test_same_length_changes_embedding_without_graph_reallocation(self):
        target, ops = runtime(), Operations()
        owner = PromptRuntimeOwner(target, operations=ops)
        old_graph, old_language = target.infer_graph, target.weights['language_embeds']
        result = owner.rebind(Tensor((5, 2048), token='different_prompt'), episode_boundary=True)
        self.assertFalse(result['changed'])
        self.assertIs(target.infer_graph, old_graph)
        self.assertIs(target.weights['language_embeds'], old_language)
        self.assertEqual(old_language.token, 'different_prompt')
        self.assertEqual(ops.allocations, [])

    def test_boundary_required_before_any_mutation(self):
        target, ops = runtime(), Operations()
        owner = PromptRuntimeOwner(target, operations=ops)
        with self.assertRaises(ValueError):
            owner.rebind(Tensor((9, 2048)), episode_boundary=False)
        self.assertEqual(target.prompt_len, 5)
        self.assertEqual(ops.synchronizations, 0)
        self.assertTrue(owner.ready)

    def test_failed_transition_remains_unusable_without_rebuilding_model(self):
        target, ops = runtime(), Operations(fail_at=3)
        owner = PromptRuntimeOwner(target, operations=ops)
        anchors = dict(owner._anchors)
        with self.assertRaises(MemoryError):
            owner.rebind(Tensor((9, 2048)), episode_boundary=True)
        self.assertFalse(owner.ready)
        self.assertTrue(all(target.weights[k] is v for k, v in anchors.items()))
        with self.assertRaises(RuntimeError):
            owner.rebind(Tensor((5, 2048)), episode_boundary=True)

    def test_replaced_weight_poisoned_before_serving(self):
        target = runtime()
        owner = PromptRuntimeOwner(target, operations=Operations())
        target.weights['encoder_attn_qkv_w'] = Tensor((1,))
        with self.assertRaises(RuntimeError):
            owner.assert_ready()
        self.assertFalse(owner.ready)

    def test_same_shape_copy_or_synchronization_error_cannot_leave_ready(self):
        for phase in ('synchronize', 'copy_embedding'):
            with self.subTest(phase=phase):
                target, ops = runtime(), Operations()
                def fail(*args):
                    raise OSError('Synthetic device operation failure')
                setattr(ops, phase, fail)
                owner = PromptRuntimeOwner(target, operations=ops)
                with self.assertRaises(OSError):
                    owner.rebind(Tensor((5, 2048)), episode_boundary=True)
                self.assertFalse(owner.ready)

    def test_changed_scratch_shape_is_rejected_before_graph_use(self):
        target = runtime()
        owner = PromptRuntimeOwner(target, operations=Operations())
        target.buffers['encoder_K'] = Tensor((1,))
        with self.assertRaises(ValueError):
            owner.assert_ready()
        self.assertFalse(owner.ready)

    def test_fp32_attention_scratch_preserves_its_dtype(self):
        target, ops = runtime(logits=True), Operations()
        owner = PromptRuntimeOwner(target, operations=ops)
        owner.rebind(Tensor((9, 2048)), episode_boundary=True)
        self.assertEqual(target.buffers['decoder_attn_logits_buf'].shape, (408, 572))
        self.assertEqual(target.buffers['decoder_attn_logits_buf'].dtype, 'float32')


if __name__ == '__main__':
    unittest.main(verbosity=2)
