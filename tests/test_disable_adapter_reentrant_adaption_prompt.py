import pytest
from transformers import GPT2Config, GPT2LMHeadModel

from peft import AdaptionPromptConfig, TaskType, get_peft_model


class TestDisableAdapterReentrantAdaptionPrompt:
    @pytest.fixture
    def gpt2_config(self):
        return GPT2Config(
            n_layer=2,
            n_head=2,
            n_embd=16,
            n_positions=32,
            n_ctx=32,
            vocab_size=64,
            bos_token_id=1,
            eos_token_id=2,
        )

    def test_adaption_prompt_disable_adapter_is_reentrant(self, gpt2_config):
        model = get_peft_model(
            GPT2LMHeadModel(gpt2_config),
            AdaptionPromptConfig(adapter_layers=1, adapter_len=2, task_type=TaskType.CAUSAL_LM),
        )

        with model.disable_adapter():
            with model.disable_adapter():
                # Nested contexts should not raise and the model should remain disabled.
                assert not model.has_active_enabled_adapter
