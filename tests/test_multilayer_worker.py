"""CPU logic check of the actual worker body; native graph validation is separate."""
import ast
from pathlib import Path
from types import SimpleNamespace


def check_multilayer_worker(source=None):
    import torch

    class Base:
        def load_model(self, *args, **kwargs):
            self.base_loaded = True

    if source is None:
        source = (Path(__file__).resolve().parents[1] /
            'src/inference_scaling/retrieval/multilayer_signals_worker.py').read_text()
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and
        node.module == 'inference_scaling.retrieval.internal_signals_worker')]
    namespace = {'RetrievalSignalWorker': Base}
    exec(compile(tree, '<actual-multilayer-worker>', 'exec'), namespace)
    cls = namespace['MultilayerSignalWorker']

    class Norm(torch.nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.offset = offset

        def forward(self, value, residual=None):
            return value + self.offset if residual is None else (value + residual + self.offset, residual)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(64)])
            for i, layer in enumerate(self.model.layers):
                layer.input_layernorm = Norm(i + 1)
            self.model.norm = Norm(65)

        def compute_logits(self, value):
            return value * 2

    model = Model()
    runner = SimpleNamespace(model=model, max_num_tokens=16, dtype=torch.float32,
        input_batch=SimpleNamespace(num_reqs=2, sampling_metadata=SimpleNamespace(max_num_logprobs=6)),
        query_start_loc=SimpleNamespace(gpu=torch.tensor([0, 2, 5])), sampler=SimpleNamespace())
    worker = cls()
    worker.model_runner, worker.device = runner, torch.device('cpu')
    worker.model_config = SimpleNamespace(get_hidden_size=lambda: 8)
    before = torch.random.get_rng_state().clone()
    worker.load_model()
    assert worker.base_loaded and torch.equal(before, torch.random.get_rng_state()), 'projection changed global RNG'
    x = torch.arange(40, dtype=torch.float32).view(5, 8)
    norms = [model.model.layers[i].input_layernorm for i in (16, 32, 48)] + [model.model.norm]
    outputs = [norm(x, x)[0] for norm in norms]
    indices = torch.tensor([1, 4])
    hidden = outputs[-1][indices]
    assert torch.equal(model.compute_logits(hidden), hidden * 2)
    assert runner.sampler._retrieval_hidden_projection is None
    assert all(torch.equal(norm._retrieval_state_buffer[:5], out) for norm, out in zip(norms, outputs))
    runner.input_batch.sampling_metadata.max_num_logprobs = 523
    assert torch.equal(model.compute_logits(hidden), hidden * 2)
    payload = runner.sampler._retrieval_hidden_projection
    assert payload.shape == (2, 513) and torch.equal(payload[:, -1], torch.zeros(2))
    generator = torch.Generator(device='cpu').manual_seed(2026111502)
    projection = (torch.randint(0, 2, (8, 128), generator=generator, dtype=torch.int32).float()*2-1)/(128**.5)
    for block, out in enumerate(outputs):
        assert torch.equal(payload[:, block*128:(block+1)*128], out[indices] @ projection)
    # Wrong indices produce an observable failure diagnostic; raw measurements
    # must be persisted by the native collector before rejecting that diagnostic.
    model.compute_logits(hidden.flip(0))
    assert bool((runner.sampler._retrieval_hidden_projection[:, -1] > 0).all())
    try:
        model.compute_logits(torch.ones(3, 8))
    except RuntimeError as error:
        assert 'padded' in str(error)
    else:
        raise AssertionError('padded logits accepted')
    # OFF capture must continue updating, including non-tuple norm outputs.
    runner.input_batch.sampling_metadata.max_num_logprobs = 6
    for norm in norms:
        norm(x + 11)
    expected = (x + 11 + 65)[indices]
    runner.input_batch.sampling_metadata.max_num_logprobs = 523
    model.compute_logits(expected)
    assert torch.equal(runner.sampler._retrieval_hidden_projection[:, -1], torch.zeros(2))
    assert torch.equal(before, torch.random.get_rng_state())
    return {'passed': True, 'checks': ['original_logits_unchanged', 'private_projection_rng',
        'each_real_block_and_row_selected', 'off_capture_updates', 'tuple_and_tensor_norm_outputs',
        'misindex_diagnostic_nonzero', 'padded_rows_rejected'],
        'scope': 'CPU worker-body logic only; not native module binding or compiled graph proof'}


def test_multilayer_worker():
    import pytest
    pytest.importorskip('torch')
    assert check_multilayer_worker()['passed']


if __name__ == '__main__':
    print(check_multilayer_worker())
