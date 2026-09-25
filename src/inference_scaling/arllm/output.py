"""Resolve thinking formats and mode from explicit settings and tokenizer metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from inference_scaling.shared.model.output import ThinkingFormat, ThinkingParser


_THINKING_MARKERS = (
    ("think", "<think>", "</think>"),
    ("thinking", "<thinking>", "</thinking>"),
    ("bracket_think", "[THINK]", "[/THINK]"),
    ("reasoning", "<reasoning>", "</reasoning>"),
)


def output_settings_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """The ``output`` settings, with an automatic thinking mode resolved from the chat template switch."""
    options = dict(config["output"])
    template = config["prompt"]["chat_template_kwargs"]
    if options["thinking_mode"] == "auto" and "enable_thinking" in template:
        enabled = template["enable_thinking"]
        if not isinstance(enabled, bool):
            raise ValueError("prompt.chat_template_kwargs.enable_thinking must be boolean")
        options["thinking_mode"] = "enabled" if enabled else "disabled"
    return options


def thinking_format_from_backend(backend: Any, options: Mapping[str, Any]) -> ThinkingParser:
    """Resolve the thinking markers from the ``output`` settings and the tokenizer.

    Without recognized markers every output falls back to the full sequence.
    """
    mode = options["thinking_mode"]
    if mode not in {"auto", "enabled", "disabled"}:
        raise ValueError("thinking_mode must be auto, enabled or disabled")
    start_text, end_text = options["thinking_start_text"], options["thinking_end_text"]
    tokenizer = getattr(backend, "tokenizer", None)
    get_vocab = getattr(tokenizer, "get_vocab", None)
    vocabulary = get_vocab() if get_vocab is not None else {}
    template = str(getattr(tokenizer, "chat_template", "") or "")
    if end_text is not None:
        definitions = [{"name": "configured", "start_text": start_text, "end_text": end_text}]
    elif start_text is not None:
        raise ValueError("thinking_start_text requires thinking_end_text")
    else:
        # Known delimiter pairs that the tokenizer vocabulary or chat template uses.
        definitions = [
            {"name": name, "start_text": start, "end_text": end}
            for name, start, end in _THINKING_MARKERS
            if (start in vocabulary and end in vocabulary) or (start in template and end in template)
        ]
    encode = getattr(backend, "encode", None) or getattr(tokenizer, "encode", None)
    formats = []
    for definition in definitions:
        start, end = definition["start_text"], definition["end_text"]
        if not isinstance(end, str) or not end:
            raise ValueError("thinking end_text must be a nonempty string")
        if start is not None and (not isinstance(start, str) or not start):
            raise ValueError("thinking start_text must be a nonempty string")
        if encode is None:
            raise ValueError("resolving thinking markers requires a tokenizer encoder")
        formats.append(ThinkingFormat(
            end_token_ids=tuple(encode(end, add_special_tokens=False)),
            start_token_ids=None if start is None else tuple(encode(start, add_special_tokens=False)),
            starts_in_thinking=options["starts_in_thinking"], name=str(definition["name"]),
        ))
    tokenizer_decode = getattr(tokenizer, "decode", None)
    backend_decode = getattr(backend, "decode", None)

    def is_blank(tokens):
        if not tokens:
            return True
        if tokenizer_decode is not None:
            text = tokenizer_decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        else:
            text = backend_decode(tokens, skip_special_tokens=False)
        return str(text).strip() == ""

    return ThinkingParser(
        tuple(formats), mode="disabled" if mode == "disabled" else "enabled" if mode == "enabled" else "auto",
        is_blank=is_blank if tokenizer_decode is not None or backend_decode is not None else None,
    )


__all__ = ["thinking_format_from_backend", "output_settings_from_config"]
