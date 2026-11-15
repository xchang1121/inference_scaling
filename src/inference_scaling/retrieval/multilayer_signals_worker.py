"""Opt-in multi-layer capture; engineering validation is required before use."""
from inference_scaling.retrieval.internal_signals_worker import RetrievalSignalWorker


LAYER_INDICES = (16, 32, 48)
PROJECTION_DIMENSIONS = 128
# Four actual representation blocks plus one final-buffer alignment diagnostic.
PAYLOAD_DIMENSIONS = 4 * PROJECTION_DIMENSIONS + 1


class MultilayerSignalWorker(RetrievalSignalWorker):
    """Capture normalized residual streams entering layers16/32/48 and finalnorm.

    Buffers are updated by graph operations even with collection OFF. The final
    buffer is checked against vLLM's indexed compute_logits input on every ON
    call. Earlier layers use those exact row indices. No model weights, sampled
    logits, or random-generator state are changed by this worker.
    """
    hidden_dimensions = PAYLOAD_DIMENSIONS

    def load_model(self, *args, **kwargs):
        super().load_model(*args, **kwargs)
        import math
        import torch
        runner = self.model_runner
        modules = dict(runner.model.named_modules())
        capture_modules = []
        base_name = None
        for layer in LAYER_INDICES:
            suffix = f'.layers.{layer}.input_layernorm'
            matches = [(name, mod) for name, mod in modules.items() if name.endswith(suffix)]
            if len(matches) != 1:
                raise RuntimeError(f'expected exactly one real input norm for layer{layer}')
            name, mod = matches[0]
            root = name[:-len(suffix)]
            if base_name is not None and root != base_name:
                raise RuntimeError('capture modules belong to different model trunks')
            base_name = root
            capture_modules.append(mod)
        final_norm = modules.get(base_name + '.norm')
        if final_norm is None:
            raise RuntimeError('missing actual final norm in the same model trunk')
        capture_modules.append(final_norm)
        hidden_size = self.model_config.get_hidden_size()

        def install_capture(module):
            module.register_buffer('_retrieval_state_buffer', torch.zeros(
                (runner.max_num_tokens, hidden_size), dtype=runner.dtype,
                device=self.device), persistent=False)
            original = module.forward

            def forward(*inner_args, **inner_kwargs):
                result = original(*inner_args, **inner_kwargs)
                normalized = result[0] if isinstance(result, tuple) else result
                # Copy inside compiled graph; Python-only hooks become stale on replay.
                module._retrieval_state_buffer[:normalized.shape[0]].copy_(normalized)
                return result

            module.forward = forward

        for module in capture_modules:
            install_capture(module)
        generator = torch.Generator(device='cpu').manual_seed(2026111502)
        projection = (torch.randint(0, 2, (hidden_size, PROJECTION_DIMENSIONS),
            generator=generator, dtype=torch.int32).float() * 2 - 1) / math.sqrt(PROJECTION_DIMENSIONS)
        projection = projection.to(self.device)
        original_logits = runner.model.compute_logits

        def logits(hidden_states, *inner_args, **inner_kwargs):
            metadata = getattr(runner.input_batch, 'sampling_metadata', None)
            if getattr(metadata, 'max_num_logprobs', None) == 10 + PAYLOAD_DIMENSIONS:
                n = runner.input_batch.num_reqs
                if hidden_states.shape[0] != n:
                    raise RuntimeError('multi-layer capture rejects speculative/padded logit rows')
                indices = (runner.query_start_loc.gpu[1:n+1] - 1).long()
                states = [m._retrieval_state_buffer[indices] for m in capture_modules]
                difference = (states[-1].float() - hidden_states.float()).square().sum(-1)
                scale = hidden_states.float().square().sum(-1).clamp_min(1e-20)
                error = (difference / scale).sqrt().unsqueeze(-1)
                # Expose error in payload so even alignment failures are preserved by
                # the short-request collector before it rejects the measurement.
                blocks = [state.float() @ projection for state in states]
                runner.sampler._retrieval_hidden_projection = torch.cat((*blocks, error), dim=-1)
            else:
                runner.sampler._retrieval_hidden_projection = None
            return original_logits(hidden_states, *inner_args, **inner_kwargs)

        runner.model.compute_logits = logits
