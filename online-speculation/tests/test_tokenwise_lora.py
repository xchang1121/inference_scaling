from __future__ import annotations

import torch
from torch import nn

from online_speculation.tokenwise_lora import TokenwiseLoraRouter


class _FakePeftLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(2, 1, bias=False)})
        self.use_dora = {"default": False}
        nn.init.ones_(self.lora_A["default"].weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.lora_A["default"](inputs)


def test_tokenwise_router_masks_only_requested_rows() -> None:
    model = nn.Sequential(_FakePeftLinear())
    router = TokenwiseLoraRouter(model)
    router.set_token_mask(torch.tensor([[0.0, 1.0, 1.0]]))
    inputs = torch.tensor([[[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]])
    output = model(inputs)
    assert output.tolist() == [[[0.0], [9.0], [13.0]]]
    assert router.hook_count == 1
    router.close()


def test_tokenwise_router_fails_closed_on_shape_mismatch() -> None:
    model = nn.Sequential(_FakePeftLinear())
    with TokenwiseLoraRouter(model) as router:
        router.set_token_mask(torch.ones((1, 2)))
        try:
            model(torch.ones((1, 3, 2)))
        except RuntimeError as error:
            assert "shapes differ" in str(error)
        else:
            raise AssertionError("shape mismatch did not fail closed")
