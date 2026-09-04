"""Per-token LoRA routing compatible with PEFT linear layers."""

from __future__ import annotations

from contextlib import AbstractContextManager

from torch import Tensor


class TokenwiseLoraRouter(AbstractContextManager["TokenwiseLoraRouter"]):
    """Zero PEFT's low-rank branch on base-only token rows.

    Uno evaluates the uncached causal seed with the frozen AR model and the
    noisy future rows with ``base + diffusion LoRA`` in the same forward. PEFT
    does not expose a token mask, so this router masks every LoRA-A activation.
    """

    def __init__(self, model: object) -> None:
        self.token_mask: Tensor | None = None
        self.handles: list[object] = []
        hooked: set[int] = set()
        for module in model.modules():
            lora_a_layers = getattr(module, "lora_A", None)
            if lora_a_layers is None:
                continue
            if any(getattr(module, "use_dora", {}).values()):
                raise RuntimeError("Tokenwise Uno routing does not support DoRA.")
            for lora_a in lora_a_layers.values():
                if id(lora_a) in hooked:
                    continue
                hooked.add(id(lora_a))
                self.handles.append(lora_a.register_forward_hook(self._mask_output))
        if not self.handles:
            raise RuntimeError("No PEFT LoRA-A layers were found.")

    @property
    def hook_count(self) -> int:
        return len(self.handles)

    def set_token_mask(self, token_mask: Tensor) -> None:
        if token_mask.ndim != 2:
            raise ValueError("Uno token mask must have shape [batch, current_tokens].")
        self.token_mask = token_mask

    def _mask_output(self, module: object, inputs: object, output: Tensor) -> Tensor:
        del module, inputs
        if self.token_mask is None:
            raise RuntimeError("Set the Uno token mask before an adapter-enabled forward.")
        if output.shape[:-1] != self.token_mask.shape:
            raise RuntimeError(
                "Uno LoRA mask and activation shapes differ: "
                f"mask={tuple(self.token_mask.shape)}, output={tuple(output.shape)}."
            )
        mask = self.token_mask.to(device=output.device, dtype=output.dtype)
        return output * mask.unsqueeze(-1)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
