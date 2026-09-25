import copy
import json

import pytest

from inference_scaling.app.ar import ARFamily
from inference_scaling.app.run import Choices
from inference_scaling.app.settings import load_settings
from inference_scaling.shared.model.loading import checkpoint_weight_files


def _family(tmp_path, model_path, **model):
    settings = copy.deepcopy(load_settings())
    settings["run"]["hash_cache_dir"] = str(tmp_path / "hashes")
    settings["ar"]["model"].update({"path": str(model_path), "revision": None, "weight_sha256": None, **model})
    return ARFamily(settings, Choices("sample", "ar", None, "gsm8k"), dataset=type("D", (), {"settings": {"max_new_tokens": 8}})())


def test_sharded_weights_are_hashed_and_a_pin_is_enforced(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"fixture"}')
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}))
    for number in (1, 2):
        (model / f"model-{number:05d}-of-00002.safetensors").write_bytes(bytes([number]))

    identity = _family(tmp_path, model).artifacts()["base"]
    assert sorted(identity["weight_files"]) == ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    assert "config.json" in identity["metadata_sha256"]
    assert _family(tmp_path, model, weight_sha256=identity["weight_sha256"]).artifacts()["base"] == identity
    (model / "model-00002-of-00002.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash to"):
        _family(tmp_path, model, weight_sha256=identity["weight_sha256"]).artifacts()


def test_adapter_files_are_part_of_the_model_identity(tmp_path):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    (base / "pytorch_model.bin").write_bytes(b"tiny fixture")
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.bin").write_bytes(b"adapter fixture")
    identity = _family(tmp_path, base, adapter={"path": str(adapter), "revision": None}).artifacts()["base"]
    assert set(identity["adapter"]["sha256"]) == {"adapter_config.json", "adapter_model.bin"}


def test_checkpoint_shards_require_safe_existing_names(tmp_path):
    index = tmp_path / "model.safetensors.index.json"
    index.write_text('{"weight_map":{"a":"../outside.safetensors"}}')
    with pytest.raises(ValueError, match="relative"):
        checkpoint_weight_files(tmp_path)
    index.write_text('{"weight_map":{"a":"missing.safetensors"}}')
    with pytest.raises(FileNotFoundError):
        checkpoint_weight_files(tmp_path)


def test_local_sharded_transformer_round_trip_uses_independent_tokenizer(tmp_path):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import AutoModelForCausalLM, GPT2Config, PreTrainedTokenizerFast
    from inference_scaling.arllm.backends.transformers_backend import TransformersBackend

    model = AutoModelForCausalLM.from_config(GPT2Config(vocab_size=5, n_layer=1, n_head=1, n_embd=8, n_positions=32))
    model.save_pretrained(tmp_path / "weights", max_shard_size="1KB")
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "a": 3, "b": 4}, unk_token="[UNK]")), bos_token="[BOS]", eos_token="[EOS]")
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    with pytest.warns(RuntimeWarning, match="precision"):
        backend = TransformersBackend.from_pretrained(str(tmp_path / "weights"),
            tokenizer_name_or_path=str(tmp_path / "tokenizer"), device="cpu", dtype="auto",
            local_files_only=True, max_score_batch_size=8, score_chunk_size=4, device_map="cpu")
    assert next(backend.model.parameters()).device == torch.device("cpu")
    assert backend.tokenizer.get_vocab() == tokenizer.get_vocab()
    assert backend.tokenizer.pad_token_id == tokenizer.eos_token_id
    assert "tokenizer=" in backend.model_id
    backend.close()
    assert backend.model is None
    del model
