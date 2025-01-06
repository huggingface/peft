import pytest
import torch


class TestGetPeftModel:
    RELOAD_WARNING_EXPECTED_MATCH = r"You are trying to modify a model .*"

    @pytest.fixture
    def get_peft_model(self):
        from peft import get_peft_model

        return get_peft_model

    @pytest.fixture
    def lora_config(self):
        from peft import LoraConfig

        return LoraConfig(target_modules="0")

    @pytest.fixture
    def base_model(self):
        return torch.nn.Sequential(torch.nn.Linear(10, 2))

    def test_get_peft_model_warns_when_reloading_model(self, get_peft_model, lora_config, base_model):
        get_peft_model(base_model, lora_config)

        with pytest.warns(UserWarning, match=self.RELOAD_WARNING_EXPECTED_MATCH):
            get_peft_model(base_model, lora_config)

    def test_get_peft_model_proposed_fix_in_warning_helps(self, get_peft_model, lora_config, base_model, recwarn):
        peft_model = get_peft_model(base_model, lora_config)
        peft_model.unload()
        get_peft_model(base_model, lora_config)

        warning_checker = pytest.warns(UserWarning, match=self.RELOAD_WARNING_EXPECTED_MATCH)

        for warning in recwarn:
            if warning_checker.matches(warning):
                pytest.fail("Warning raised even though model was unloaded.")
