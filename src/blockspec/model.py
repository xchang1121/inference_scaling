"""A small, explicit causal Transformer with token-gated low-rank projections."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
Cache = tuple[tuple[Tensor, Tensor], ...]


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32
    hidden_size: int = 64
    intermediate_size: int = 128
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 16
    rms_norm_eps: float = 1e-6
    norm_groups: int = 1
    norm_weight_in_fp32: bool = False
    query_key_norm: bool = False
    attention_bias: bool = False
    attention_output_bias: bool = False
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_factor: float = 1.0
    rope_original_length: int = 8192
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    rope_attention_factor: float = 1.0
    rope_truncate: bool = True
    adapter_rank: int = 8
    adapter_alpha: float = 8.0
    adapter_targets: tuple[str, ...] = PROJECTIONS

    def __post_init__(self):
        for name in ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                     "num_attention_heads", "num_key_value_heads", "head_dim", "norm_groups"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.head_dim % 2 or self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("even rotary head dimension and integral query/KV ratio required")
        if self.hidden_size % self.norm_groups or self.adapter_rank < 0:
            raise ValueError("invalid norm groups or adapter rank")
        if not set(self.adapter_targets).issubset(PROJECTIONS):
            raise ValueError("unknown adapter projection")
        if len(set(self.adapter_targets)) != len(self.adapter_targets):
            raise ValueError("duplicate adapter projection")
        if not all(math.isfinite(v) and v > 0 for v in (
            self.rms_norm_eps, self.rope_theta, self.rope_factor, self.rope_attention_factor,
            self.rope_beta_fast, self.rope_beta_slow, self.adapter_alpha,
        )):
            raise ValueError("normalization, rotation, and adapter scales must be finite and positive")
        if self.rope_theta <= 1 or self.rope_factor < 1:
            raise ValueError("rotary base > 1 and extension factor >= 1 required")
        if self.rope_beta_fast <= self.rope_beta_slow or self.rope_original_length < 1:
            raise ValueError("invalid rotary correction interval")

    def to_dict(self):
        result = asdict(self)
        result["adapter_targets"] = list(self.adapter_targets)
        return result

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if "adapter_targets" in value:
            value["adapter_targets"] = tuple(value["adapter_targets"])
        return cls(**value)


class GatedLinear(nn.Module):
    """W x + m (alpha/r) B A x. A mask of None does not execute the adapter."""

    def __init__(self, inputs, outputs, rank, alpha, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(outputs, inputs))
        self.bias = nn.Parameter(torch.zeros(outputs)) if bias else None
        self.rank = rank
        self.scale = alpha / rank if rank else 0.0
        if rank:
            self.lora_A = nn.Parameter(torch.randn(rank, inputs) / math.sqrt(inputs))
            self.lora_B = nn.Parameter(torch.zeros(outputs, rank))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x, mask=None):
        out = F.linear(x, self.weight, self.bias)
        if self.rank and mask is not None:
            low = F.linear(x, self.lora_A.to(x.dtype))
            delta = F.linear(low, self.lora_B.to(x.dtype))
            out = out + mask[..., None].to(x.dtype) * (delta * self.scale)
        return out


class GroupedRMSNorm(nn.Module):
    def __init__(self, size, eps, groups=1, weight_in_fp32=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps, self.groups = eps, groups
        self.weight_in_fp32 = weight_in_fp32

    def forward(self, x):
        dtype = x.dtype
        work = x.float() if dtype in (torch.float16, torch.bfloat16) else x
        grouped = work.reshape(*x.shape[:-1], self.groups, -1)
        unit = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + self.eps)
        if self.weight_in_fp32:
            return (unit.reshape_as(x) * self.weight.to(work.dtype)).to(dtype)
        return unit.reshape_as(x).to(dtype) * self.weight


def rotary_frequencies(config: ModelConfig, *, device=None):
    """Ordinary rotary frequencies, or the fixed YaRN interpolation ramp."""
    d = config.head_dim
    indices = torch.arange(d // 2, dtype=torch.float32, device=device)
    frequency = config.rope_theta ** (-2 * indices / d)
    if config.rope_factor == 1:
        return frequency
    def boundary(rotations):
        return d * math.log(config.rope_original_length / (2 * math.pi * rotations)) / (
            2 * math.log(config.rope_theta)
        )
    lo, hi = boundary(config.rope_beta_fast), boundary(config.rope_beta_slow)
    if config.rope_truncate:
        lo, hi = math.floor(lo), math.ceil(hi)
    lo, hi = max(0, lo), min(d - 1, hi)
    ramp = ((indices - lo) / max(hi - lo, 1e-3)).clamp(0, 1)
    return frequency * (1 - ramp + ramp / config.rope_factor)


def rotate(x, positions, frequencies, attention_factor=1.0):
    angles = positions.float()[..., None] * frequencies
    phase = torch.cat((angles, angles), dim=-1)[:, None, :, :]
    cosine = (phase.cos() * attention_factor).to(x.dtype)
    sine = (phase.sin() * attention_factor).to(x.dtype)
    half = x.shape[-1] // 2
    perpendicular = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cosine + perpendicular * sine


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        for name, outputs in (("q_proj", config.num_attention_heads * config.head_dim),
                              ("k_proj", config.num_key_value_heads * config.head_dim),
                              ("v_proj", config.num_key_value_heads * config.head_dim)):
            setattr(self, name, GatedLinear(config.hidden_size, outputs,
                    config.adapter_rank if name in config.adapter_targets else 0,
                    config.adapter_alpha, config.attention_bias))
        self.o_proj = GatedLinear(config.num_attention_heads * config.head_dim,
                                 config.hidden_size,
                                 config.adapter_rank if "o_proj" in config.adapter_targets else 0,
                                 config.adapter_alpha, config.attention_output_bias)
        if config.query_key_norm:
            self.q_norm = GroupedRMSNorm(config.head_dim, config.rms_norm_eps)
            self.k_norm = GroupedRMSNorm(config.head_dim, config.rms_norm_eps)
        self.register_buffer("frequencies", rotary_frequencies(config), persistent=False)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # model.to(BF16) must not quantize the rotary frequency table.
        self.frequencies = rotary_frequencies(self.config, device=self.q_proj.weight.device)
        return self

    def forward(self, x, positions, allowed, mask, past):
        b, length, _ = x.shape
        c = self.config
        q = self.q_proj(x, mask).reshape(b, length, c.num_attention_heads, c.head_dim)
        k = self.k_proj(x, mask).reshape(b, length, c.num_key_value_heads, c.head_dim)
        v = self.v_proj(x, mask).reshape(b, length, c.num_key_value_heads, c.head_dim)
        if c.query_key_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q = rotate(q.transpose(1, 2), positions, self.frequencies, c.rope_attention_factor)
        k = rotate(k.transpose(1, 2), positions, self.frequencies, c.rope_attention_factor)
        v = v.transpose(1, 2)
        if past is not None:
            k, v = torch.cat((past[0], k), dim=2), torch.cat((past[1], v), dim=2)
        # Explicit repeat is the readable reference path, not a fused paged kernel.
        repeat = c.num_attention_heads // c.num_key_value_heads
        attended = F.scaled_dot_product_attention(
            q, k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1),
            attn_mask=allowed, dropout_p=0.0,
        )
        out = attended.transpose(1, 2).reshape(b, length, -1)
        return self.o_proj(out, mask), (k, v)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        for name, inputs, outputs in (
            ("gate_proj", config.hidden_size, config.intermediate_size),
            ("up_proj", config.hidden_size, config.intermediate_size),
            ("down_proj", config.intermediate_size, config.hidden_size),
        ):
            setattr(self, name, GatedLinear(inputs, outputs,
                    config.adapter_rank if name in config.adapter_targets else 0,
                    config.adapter_alpha))

    def forward(self, x, mask):
        return self.down_proj(F.silu(self.gate_proj(x, mask)) * self.up_proj(x, mask), mask)


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = GroupedRMSNorm(config.hidden_size, config.rms_norm_eps,
                                             config.norm_groups, config.norm_weight_in_fp32)
        self.post_attention_layernorm = GroupedRMSNorm(config.hidden_size, config.rms_norm_eps,
                                                      config.norm_groups, config.norm_weight_in_fp32)
        self.self_attn = Attention(config)
        self.mlp = MLP(config)

    def forward(self, x, positions, allowed, mask, past):
        attention, cache = self.self_attn(self.input_layernorm(x), positions, allowed, mask, past)
        x = x + attention
        return x + self.mlp(self.post_attention_layernorm(x), mask), cache


class Body(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        nn.init.normal_(self.embed_tokens.weight, std=0.02)
        self.layers = nn.ModuleList([Layer(config) for _ in range(config.num_hidden_layers)])
        self.norm = GroupedRMSNorm(config.hidden_size, config.rms_norm_eps,
                                   config.norm_groups, config.norm_weight_in_fp32)


class Decoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = Body(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, std=0.02)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, tokens, *, positions=None, allowed=None, adapter_mask=None,
                cache: Cache | None = None, return_cache=False):
        if tokens.ndim != 2 or tokens.shape[1] < 1:
            raise ValueError("tokens must have shape [batch, nonempty sequence]")
        b, length = tokens.shape
        prefix = cache_length(cache)
        if cache is not None and len(cache) != self.config.num_hidden_layers:
            raise ValueError("cache layer count differs from model")
        if positions is None:
            positions = torch.arange(prefix, prefix + length, device=tokens.device)[None].expand(b, -1)
        if positions.shape != tokens.shape:
            raise ValueError("position ids must match tokens")
        if adapter_mask is not None and adapter_mask.shape != tokens.shape:
            raise ValueError("adapter mask must match tokens")
        if allowed is None:
            keys = torch.arange(prefix + length, device=tokens.device)
            queries = prefix + torch.arange(length, device=tokens.device)
            allowed = (keys[None, :] <= queries[:, None])[None, None]
        if allowed.dtype != torch.bool or allowed.shape[-2:] != (length, prefix + length):
            raise ValueError("attention mask must be boolean with query/key dimensions")
        hidden = self.model.embed_tokens(tokens)
        new_cache = []
        for i, layer in enumerate(self.model.layers):
            hidden, kv = layer(hidden, positions, allowed, adapter_mask,
                               None if cache is None else cache[i])
            if return_cache:
                new_cache.append(kv)
        logits = self.lm_head(self.model.norm(hidden))
        return (logits, tuple(new_cache)) if return_cache else logits

    def adapter_parameters(self):
        return [p for n, p in self.named_parameters() if is_adapter(n)]

    def train_adapters_only(self):
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(is_adapter(name))
            if is_adapter(name) and parameter.dtype in (torch.float16, torch.bfloat16):
                parameter.data = parameter.data.float()
        return self

    def train_base_only(self):
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(not is_adapter(name))
        return self


def is_adapter(name):
    return name.endswith(".lora_A") or name.endswith(".lora_B")


def cache_length(cache: Cache | None):
    if cache is not None and not cache:
        raise ValueError("empty cache tuple; use None for an empty prefix")
    return 0 if cache is None else cache[0][0].shape[2]


def trim_cache(cache: Cache | None, length: int) -> Cache | None:
    if length < 0 or length > cache_length(cache):
        raise ValueError("cannot extend cache by trimming")
    if cache is None or length == 0:
        return None
    return tuple((k[:, :, :length].detach(), v[:, :, :length].detach()) for k, v in cache)
