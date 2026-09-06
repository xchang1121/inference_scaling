"""Strict tensor-name bridge and resumable own-format dual-view checkpoints."""

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from .backbone import DualViewConfig, DualViewDecoder


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_key_map(config):
    """Own key -> public key. Both the output projection and Q/K norms are routed."""
    mapping = {"embedding.weight": "model.embed_tokens.weight", "norm.weight": "model.norm.weight"}
    if not config.tie_word_embeddings:
        mapping["head.weight"] = "lm_head.weight"
    for index in range(config.num_hidden_layers):
        own, source = f"layers.{index}.", f"model.layers.{index}."
        for name, original in (("input_norm", "input_layernorm"),
                               ("post_norm", "post_attention_layernorm")):
            mapping[own + name + ".weight"] = source + original + ".weight"
        for name in ("gate", "up", "down"):
            mapping[own + name + ".weight"] = source + "mlp." + name + "_proj.weight"
        for view, suffix in (("ar", ""), ("draft", "_diff")):
            for name in ("q", "k", "v", "o"):
                for kind in (("weight", "bias") if config.attention_bias else ("weight",)):
                    mapping[own + f"attention.{view}.{name}.{kind}"] = (
                        source + f"self_attn.{name}_proj{suffix}.{kind}")
            for name in ("q_norm", "k_norm"):
                mapping[own + f"attention.{view}.{name}.weight"] = (
                    source + f"self_attn.{name}{suffix}.weight")
    return mapping


def load_public(path, *, device="cpu", dtype=torch.float32, expected_sha256=None):
    path = Path(path)
    config = DualViewConfig.from_public(json.loads((path / "config.json").read_text()))
    weights_path = path / "model.safetensors"
    fingerprint = file_sha256(weights_path)
    if expected_sha256 is not None and fingerprint != expected_sha256:
        raise ValueError("public weight SHA256 mismatch")
    with torch.device("meta"):
        model = DualViewDecoder(config)
    expected = model.state_dict()
    mapping = public_key_map(config)
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        actual_keys = set(handle.keys())
        extra_tied_head = config.tie_word_embeddings and "lm_head.weight" in actual_keys
        required = set(mapping.values())
        if extra_tied_head:
            required.add("lm_head.weight")
        if actual_keys != required:
            raise ValueError(f"public tensor keys differ: missing={required - actual_keys}, "
                             f"unexpected={actual_keys - required}")
        if extra_tied_head and not torch.equal(handle.get_tensor("lm_head.weight"),
                                              handle.get_tensor("model.embed_tokens.weight")):
            raise ValueError("tied output head differs from token embeddings")
        state = {}
        for own, original in mapping.items():
            tensor = handle.get_tensor(original)
            if tensor.shape != expected[own].shape:
                raise ValueError(f"shape mismatch for {original}: {tensor.shape} != {expected[own].shape}")
            state[own] = tensor.to(device=device, dtype=dtype)
    if config.tie_word_embeddings:
        state["head.weight"] = state["embedding.weight"]
    model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    model.frequencies = model._frequencies(model.embedding.weight.device)
    model.source = {"weight_sha256": fingerprint, "config_sha256": file_sha256(path / "config.json"),
                    "directory": str(path)}
    return model.eval().requires_grad_(False)


def save_checkpoint(path, model, *, optimizer=None, step=0, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": "blockspec-dual-view-v1", "config": model.config.to_dict(),
               "state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
               "trainable": [key for key, value in model.named_parameters() if value.requires_grad],
               "optimizer": None if optimizer is None else optimizer.state_dict(),
               "step": step, "metadata": metadata or {}, "source": getattr(model, "source", {})}
    with path.open("xb") as handle:
        torch.save(payload, handle)


def load_checkpoint(path, *, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "blockspec-dual-view-v1":
        raise ValueError("unknown dual-view checkpoint format")
    config = DualViewConfig(**payload["config"])
    with torch.device("meta"):
        model = DualViewDecoder(config)
    expected, state = model.state_dict(), payload["state"]
    if set(expected) != set(state):
        raise ValueError("checkpoint tensor keys differ")
    for key, value in state.items():
        if value.shape != expected[key].shape or not torch.isfinite(value).all():
            raise ValueError(f"invalid checkpoint tensor: {key}")
    if config.tie_word_embeddings and not torch.equal(state["embedding.weight"], state["head.weight"]):
        raise ValueError("checkpoint tied weights differ")
    model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    model.frequencies = model._frequencies(model.embedding.weight.device)
    model.requires_grad_(False)
    parameters = dict(model.named_parameters())
    if set(payload["trainable"]) - parameters.keys():
        raise ValueError("unknown trainable checkpoint parameters")
    for name in payload["trainable"]:
        parameters[name].requires_grad_(True)
    model.source = payload["source"]
    return model, {key: payload[key] for key in ("optimizer", "step", "metadata")}
