import json

import pytest
import torch

from blockspec.checkpoint import base_fingerprint
from blockspec.data import load_sequences, sample_batch
from blockspec.model import Decoder, ModelConfig
from blockspec.training import TrainingConfig, train_adapter


def test_jsonl_records_are_never_packed_across_boundaries(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({"input_ids": [1] * 8}) + "\n" +
                    json.dumps({"input_ids": [2] * 8}), encoding="utf-8")
    sequences = load_sequences(path, 3)
    batch = sample_batch(sequences, batch_size=16, length=6, bos_id=0, device="cpu",
                         generator=torch.Generator().manual_seed(2))
    assert torch.equal(batch[:, 0], torch.zeros(16, dtype=torch.long))
    assert all(row[1:].unique().numel() == 1 for row in batch)


@pytest.mark.parametrize("ids", [[0, 7], [0, -1], [0, 1.0], [True, 1], [1]])
def test_bad_dataset_fails(ids, tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({"input_ids": ids}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_sequences(path, 7)


def test_curriculum_and_loss_schedule_execute_with_frozen_teacher():
    model = Decoder(ModelConfig(vocab_size=8, hidden_size=16, intermediate_size=24,
                                 num_hidden_layers=1, num_attention_heads=2,
                                 num_key_value_heads=1, head_dim=8, adapter_rank=2))
    frozen = base_fingerprint(model)
    events = []
    result = train_adapter(model, [torch.tensor([1, 2, 3, 4, 5, 6, 7])],
                           TrainingConfig(steps=4, batch_size=1, sequence_length=6,
                                          blocks=(2, 4), warmup_steps=2), progress=events.append)
    assert [e["block"] for e in events] == [2, 2, 4, 4]
    assert [e["loss_kind"] for e in events] == ["reverse_kl", "reverse_kl", "l1", "l1"]
    assert base_fingerprint(model) == frozen
    assert result["training_tokens"] == 24
