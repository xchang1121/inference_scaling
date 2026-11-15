"""Reconstruct actual grouped-query softmax weights from a native paged KV cache."""


def reconstruct_attention(query, keys, values, output, *, topk=16, head_pooling='all'):
    import torch
    if query.ndim != 2 or keys.ndim != 3 or values.shape != keys.shape:
        raise ValueError('expected Q[heads,dim] and K/V[length,kv_heads,dim]')
    length, kv_heads, width = keys.shape
    if query.shape[1] != width or query.shape[0] % kv_heads or output.shape != query.shape:
        raise ValueError('invalid grouped-query attention dimensions')
    grouped = query.float().reshape(kv_heads,-1,width)
    logits = torch.einsum('hgd,lhd->hgl',grouped,keys.float())/(width**.5)
    weights = logits.softmax(dim=-1)
    rebuilt = torch.einsum('hgl,lhd->hgd',weights,values.float()).reshape_as(query)
    error = (rebuilt-output.float()).norm()/output.float().norm().clamp_min(1e-8)
    normalization = (weights.sum(-1)-1).abs().max()
    if head_pooling == 'all':
        average = weights.mean(dim=(0,1))
    else:
        from inference_scaling.retrieval.attention_geometry import head_indices
        indices = head_indices(query.shape[0], head_pooling)
        # Reconstruction error above still covers EVERY head, including unselected heads.
        average = weights.reshape(query.shape[0], length)[list(indices)].mean(dim=0)
    scores, indices = average.topk(min(topk,length))
    # Short-prefix padding has zero mass and remains a valid physical index.
    if length < topk:
        indices = torch.cat((indices,indices.new_zeros(topk-length)))
        scores = torch.cat((scores,scores.new_zeros(topk-length)))
    return torch.cat((indices.float(),scores,average.new_tensor([length]),error[None],normalization[None]))
