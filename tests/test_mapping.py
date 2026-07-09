# Copyright 2025-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import torch

from peft import LoraConfig, get_peft_model


class TestGetPeftModel:
    RELOAD_WARNING_EXPECTED_MATCH = r"You are trying to modify a model .*"
    META_DEVICE_WARNING_EXPECTED_MATCH = r"The model has parameters on the meta device.*"

    @pytest.fixture
    def lora_config_0(self):
        return LoraConfig(target_modules="0")

    @pytest.fixture
    def base_model(self):
        return torch.nn.Sequential(torch.nn.Linear(10, 2), torch.nn.Linear(2, 10))

    def test_get_peft_model_warns_when_reloading_model(self, lora_config_0, base_model):
        get_peft_model(base_model, lora_config_0)

        with pytest.warns(UserWarning, match=self.RELOAD_WARNING_EXPECTED_MATCH):
            get_peft_model(base_model, lora_config_0)

    def test_get_peft_model_proposed_fix_in_warning_helps(self, lora_config_0, base_model, recwarn):
        peft_model = get_peft_model(base_model, lora_config_0)
        peft_model.unload()
        get_peft_model(base_model, lora_config_0)

        warning_checker = pytest.warns(UserWarning, match=self.RELOAD_WARNING_EXPECTED_MATCH)

        for warning in recwarn:
            if warning_checker.matches(warning):
                pytest.fail("Warning raised even though model was unloaded.")

    def test_get_peft_model_repeated_invocation(self, lora_config_0, base_model):
        peft_model = get_peft_model(base_model, lora_config_0)

        # use direct-addressing of the other layer to accommodate for the nested model
        lora_config_1 = LoraConfig(target_modules="base_model.model.1")

        with pytest.warns(UserWarning, match=self.RELOAD_WARNING_EXPECTED_MATCH):
            get_peft_model(peft_model, lora_config_1)

    def test_get_peft_model_warns_when_params_on_meta_device(self, lora_config_0):
        with torch.device("meta"):
            base_model = torch.nn.Sequential(torch.nn.Linear(10, 2), torch.nn.Linear(2, 10))

        with pytest.warns(UserWarning, match=self.META_DEVICE_WARNING_EXPECTED_MATCH):
            get_peft_model(base_model, lora_config_0)

    def test_get_peft_model_no_meta_device_warning_on_materialized_model(self, lora_config_0, base_model, recwarn):
        get_peft_model(base_model, lora_config_0)

        warning_checker = pytest.warns(UserWarning, match=self.META_DEVICE_WARNING_EXPECTED_MATCH)
        for warning in recwarn:
            if warning_checker.matches(warning):
                pytest.fail("Meta device warning raised even though all parameters are materialized.")

    def test_get_peft_model_no_meta_device_warning_with_hf_device_map(self, lora_config_0, recwarn):
        # models loaded with a device_map (e.g. device_map="auto") may legitimately have offloaded parameters on the
        # meta device; these flows handle materialization themselves and should not trigger the warning
        with torch.device("meta"):
            base_model = torch.nn.Sequential(torch.nn.Linear(10, 2), torch.nn.Linear(2, 10))
        base_model.hf_device_map = {"": "cpu"}

        get_peft_model(base_model, lora_config_0)

        warning_checker = pytest.warns(UserWarning, match=self.META_DEVICE_WARNING_EXPECTED_MATCH)
        for warning in recwarn:
            if warning_checker.matches(warning):
                pytest.fail("Meta device warning raised even though the model has an hf_device_map.")

    def test_get_peft_model_no_meta_device_warning_with_low_cpu_mem_usage(self, lora_config_0, recwarn):
        with torch.device("meta"):
            base_model = torch.nn.Sequential(torch.nn.Linear(10, 2), torch.nn.Linear(2, 10))

        get_peft_model(base_model, lora_config_0, low_cpu_mem_usage=True)

        warning_checker = pytest.warns(UserWarning, match=self.META_DEVICE_WARNING_EXPECTED_MATCH)
        for warning in recwarn:
            if warning_checker.matches(warning):
                pytest.fail("Meta device warning raised even though low_cpu_mem_usage=True was set.")
