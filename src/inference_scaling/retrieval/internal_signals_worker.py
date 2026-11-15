"""Ascend experiment worker, retaining the existing native sampling adapter."""
from inference_scaling.arllm.backends.vllm_ascend_worker import AscendLogprobWorker
from inference_scaling.retrieval.internal_signals import install_distribution_signals


class RetrievalSignalWorker(AscendLogprobWorker):
    hidden_dimensions = 0
    attention_topk = 0

    def init_device(self):
        super().init_device()
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_config.tokenizer, local_files_only=True)
        eos, close = tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('</think>')
        if eos is None or close is None or close == tokenizer.unk_token_id:
            raise RuntimeError('missing native terminal token IDs')
        install_distribution_signals(self.model_runner.sampler,
                                     self.model_config.get_vocab_size(), eos, close, self.hidden_dimensions, self.attention_topk)


class HiddenSignalWorker(RetrievalSignalWorker):
    """128 fixed random features of the real final normalized hidden state.

compute_logits receives vLLM's already indexed per-request hidden rows. The
projection is read-only and uses its own CPU generator, preserving sampling RNG.
This is the final layer representation, not an attention map or middle layer.
"""
    hidden_dimensions = 128

    def load_model(self, *args, **kwargs):
        super().load_model(*args, **kwargs)
        import math
        import torch
        runner = self.model_runner
        original = runner.model.compute_logits
        hidden_size = self.model_config.get_hidden_size()
        generator = torch.Generator(device='cpu').manual_seed(2026111502)
        projection = (torch.randint(0, 2, (hidden_size, self.hidden_dimensions),
                                    generator=generator, dtype=torch.int32).float()*2-1)/math.sqrt(self.hidden_dimensions)
        projection = projection.to(self.device)

        def logits(hidden_states, *inner_args, **inner_kwargs):
            metadata = getattr(runner.input_batch, 'sampling_metadata', None)
            if getattr(metadata, 'max_num_logprobs', None) == 10+self.hidden_dimensions:
                runner.sampler._retrieval_hidden_projection = hidden_states.float() @ projection
            else:
                runner.sampler._retrieval_hidden_projection = None
            return original(hidden_states, *inner_args, **inner_kwargs)

        runner.model.compute_logits = logits


class AttentionSignalWorker(RetrievalSignalWorker):
    """Experimental layer-63 full attention; validate before method experiments.

The capture copies are part of every compiled graph, including collection-OFF
controls. This avoids stale Python hooks during graph replay. Reconstruction is
read-only, and P@V must match the real pre-output-gate attention result.
"""
    attention_topk = 16

    def load_model(self, *args, **kwargs):
        super().load_model(*args, **kwargs)
        import torch
        from inference_scaling.retrieval.attention_signals import reconstruct_attention
        runner = self.model_runner
        matches = [(name,mod) for name,mod in runner.model.named_modules()
                   if name.endswith('layers.63.self_attn') and hasattr(mod,'attn')]
        if len(matches) != 1:
            raise RuntimeError('expected exactly one layer-63 full-attention module')
        parent = matches[0][1]
        attn = parent.attn
        if getattr(attn.impl,'sliding_window',None) is not None or getattr(attn.impl,'sinks',None) is not None:
            raise RuntimeError('attention extraction requires full causal attention without sinks')
        width = parent.num_heads*parent.head_dim
        for name in ('_retrieval_q_buffer','_retrieval_o_buffer'):
            attn.register_buffer(name,torch.zeros((runner.max_num_tokens,width),dtype=runner.dtype,
                                                  device=self.device),persistent=False)
        original_attention = attn.forward

        def capture(query, key, value, *inner_args, **inner_kwargs):
            attn._retrieval_q_buffer[:query.shape[0]].copy_(query.reshape(query.shape[0],width))
            result = original_attention(query,key,value,*inner_args,**inner_kwargs)
            attn._retrieval_o_buffer[:result.shape[0]].copy_(result.reshape(result.shape[0],width))
            return result

        attn.forward = capture
        original_logits = runner.model.compute_logits

        def logits(hidden_states, *inner_args, **inner_kwargs):
            metadata = getattr(runner.input_batch,'sampling_metadata',None)
            enabled = getattr(metadata,'max_num_logprobs',None) == 10+2*self.attention_topk+3
            if enabled:
                n = runner.input_batch.num_reqs
                if hidden_states.shape[0] != n:
                    raise RuntimeError('attention extraction does not support speculative/padded logit rows')
                groups = [i for i,g in enumerate(runner.kv_cache_config.kv_cache_groups)
                          if attn.layer_name in g.layer_names]
                if len(groups) != 1:
                    raise RuntimeError('attention KV-cache group is ambiguous')
                table = runner.input_batch.block_table[groups[0]].get_device_tensor()
                lengths = runner.seq_lens[:n].tolist()
                indices = runner.query_start_loc.gpu[1:n+1]-1
                queries = attn._retrieval_q_buffer[indices.long()].view(n,parent.num_heads,parent.head_dim)
                outputs = attn._retrieval_o_buffer[indices.long()].view_as(queries)
                keys, values = attn.impl.key_cache, attn.impl.value_cache
                if keys is None or keys.ndim != 4 or keys.shape[-2:] != (parent.num_kv_heads,parent.head_dim):
                    raise RuntimeError('unexpected native KV-cache layout')
                rows = []
                for i,length in enumerate(lengths):
                    blocks = table[i,:(length+keys.shape[1]-1)//keys.shape[1]].long()
                    k = keys.index_select(0,blocks).reshape(-1,parent.num_kv_heads,parent.head_dim)[:length]
                    v = values.index_select(0,blocks).reshape(-1,parent.num_kv_heads,parent.head_dim)[:length]
                    rows.append(reconstruct_attention(queries[i],k,v,outputs[i],topk=self.attention_topk))
                runner.sampler._retrieval_attention = torch.stack(rows)
            else:
                runner.sampler._retrieval_attention = None
            return original_logits(hidden_states,*inner_args,**inner_kwargs)

        runner.model.compute_logits = logits
