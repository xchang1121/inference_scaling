"""Prompt rendering and JSON serialization shared by the retrieval runner."""
import math
from inference_scaling.shared.model.prompting import render_prompt

def prompt_config(ar, thinking):
    return {**ar, "prompt": {**ar["prompt"], "chat_template_kwargs": {
        "enable_thinking": thinking, "reasoning_effort": "medium"}},
        "output": {**ar["output"], "thinking_mode": "enabled" if thinking else "disabled"}}


def render(tokenizer, ar, system, user, thinking):
    text = render_prompt(tokenizer, [{"role": "system", "content": system},
        {"role": "user", "content": user}], prompt_config(ar, thinking))
    if "Reasoning effort is set to" in text:
        raise ValueError("effort instruction unexpectedly rendered")
    if thinking and not text.endswith("<think>\n"):
        raise ValueError("main prompt must leave thinking open")
    if not thinking and not text.endswith("</think>\n\n"):
        raise ValueError("checker must start outside thinking")
    return tuple(tokenizer.encode(text, add_special_tokens=False))


def clean(value):
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    return value
