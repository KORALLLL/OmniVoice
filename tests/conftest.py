from pathlib import Path
from types import SimpleNamespace

import pytest
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel


class ToyOmniConfig(PretrainedConfig):
    model_type = "toy_omnivoice"

    def __init__(self, vocab_size=32, hidden_size=8, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size


class ToyOmniVoice(PreTrainedModel):
    config_class = ToyOmniConfig

    def __init__(self, config=None):
        config = config or ToyOmniConfig()
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.audio_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        for name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ):
            setattr(
                self,
                name,
                nn.Linear(config.hidden_size, config.hidden_size, bias=False),
            )
        self.audio_heads = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, input_ids, audio_ids=None, labels=None, **kwargs):
        hidden = self.embed_tokens(input_ids)
        if audio_ids is not None:
            hidden = hidden + self.audio_embeddings(audio_ids)
        for name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ):
            hidden = hidden + 0.01 * getattr(self, name)(hidden)
        logits = self.audio_heads(hidden)
        loss = logits.float().square().mean() if labels is None else F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
        )
        return SimpleNamespace(loss=loss, logits=logits)


class DummyTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __len__(self):
        return 32

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer_config.json").write_text("{}")


@pytest.fixture
def toy_omnivoice():
    return ToyOmniVoice()
