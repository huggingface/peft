# Copyright 2026-present the HuggingFace Inc. team.
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

import copy

import pytest
import torch
from torch import nn
from transformers import AutoModelForCausalLM

from peft import (
    ActivationSteeringConfig,
    get_peft_model,
    inject_adapter_in_model,
)
from peft.tuners.activation_steering import calibrate_activation_steering

from .testing_common import PeftCommonTester


class SteeringBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.down_proj.weight.copy_(torch.tensor([[3.0, 0.0, 1.0], [4.0, 2.0, 0.0]]))

    def forward(self, hidden_states):
        output = self.down_proj(hidden_states)
        return output


class SteeringModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = SteeringBlock()

    def forward(self, hidden_states, attention_mask=None, past_key_values=None):
        return self.block(hidden_states)


def steering_config(indices=None, init_val=-1.0, statistic="median_l2", **kwargs):
    return ActivationSteeringConfig(
        steering_pattern={
            "block": {
                "source_module": "down_proj",
                "indices": [0] if indices is None else indices,
                "init_val": init_val,
                "statistic": statistic,
            }
        },
        **kwargs,
    )


def calibrated_model(base, config, inputs, attention_mask=None, **kwargs):
    model = get_peft_model(base, config, **kwargs)
    calibrate_activation_steering(
        model,
        {"hidden_states": inputs, "attention_mask": attention_mask},
        adapter_name=kwargs.get("adapter_name", "default"),
    )
    return model


class TestActivationSteering:
    @pytest.fixture
    def inputs(self):
        # Position zero has a large norm and must not contribute to calibration.
        return torch.tensor([[[100.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])

    @pytest.mark.parametrize(
        "indices,dose,expected_change",
        [
            ([0], 0.0, [0.0, 0.0]),
            ([0], 2.0, [12.0, 16.0]),
            ([0, 1], 0.5, [3.0, 9.0]),
            ([0, 1], [-1.0, 0.5], [-6.0, -3.0]),
        ],
    )
    def test_doses_control_direction_and_strength(self, inputs, indices, dose, expected_change):
        base = SteeringModel()
        reference = base(inputs)
        model = calibrated_model(base, steering_config(indices=indices, init_val=dose), inputs)
        torch.testing.assert_close(model(inputs), reference + torch.tensor(expected_change))

    @pytest.mark.parametrize("batched", [True, False])
    def test_calibration_ignores_padding_and_first_tokens(self, inputs, batched):
        sequences = inputs.repeat(2, 1, 1)
        sequences[1] *= 2
        mask = torch.tensor([[1, 1, 1, 0], [0, 1, 1, 1]])
        base = SteeringModel()
        reference = base(sequences)
        model = get_peft_model(base, steering_config(init_val=1.0))
        if batched:
            data = {"hidden_states": sequences, "attention_mask": mask}
        else:
            data = [
                {"hidden_states": sequences[:1, :3]},
                {"hidden_states": sequences[1:, 1:]},
            ]
        calibrate_activation_steering(model, data)
        # Both representations contain the same four calibration tokens; padded outputs stay unchanged.
        expected = reference + torch.tensor([9.0, 12.0]) * mask[..., None]
        torch.testing.assert_close(model(sequences, attention_mask=mask), expected)

    def test_calibrate_multiple_blocks(self):
        class TwoBlockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Sequential(nn.Linear(2, 2, bias=False)) for _ in range(2)])
                with torch.no_grad():
                    for block in self.layers:
                        block[0].weight.copy_(torch.eye(2))

            def forward(self, hidden_states):
                for block in self.layers:
                    hidden_states = block(hidden_states)
                return hidden_states

        config = ActivationSteeringConfig(
            steering_pattern={
                "layers.0": {"source_module": "0", "indices": [0], "init_val": 1.0},
                "layers.1": {"source_module": "0", "indices": [1], "init_val": -1.0},
            }
        )
        inputs = torch.tensor([[[100.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
        model = get_peft_model(TwoBlockModel(), config)
        calibrate_activation_steering(model, {"hidden_states": inputs})
        # Both blocks use baseline calibration; the earlier intervention must not affect the later scale.
        torch.testing.assert_close(model(inputs), inputs + torch.tensor([2.0, -2.0]))

    def test_calibration_is_fixed_until_recalibration(self, inputs):
        base = SteeringModel()
        probe = (inputs[:, -1:] * 10).repeat(2, 1, 1)
        reference = base(probe)
        model = calibrated_model(base, steering_config(init_val=1.0), inputs)
        # New prompts, including single-token inputs and a different batch size, reuse the trained scale.
        torch.testing.assert_close(model(probe), reference + torch.tensor([6.0, 8.0]))
        calibrate_activation_steering(model, {"hidden_states": inputs * 2})
        torch.testing.assert_close(model(probe), reference + torch.tensor([12.0, 16.0]))

    def test_named_adapters_keep_separate_calibration(self, inputs):
        base = SteeringModel()
        reference = base(inputs)
        model = calibrated_model(base, steering_config(init_val=1.0), inputs)
        first = model(inputs)
        model.add_adapter("other", steering_config(indices=[1], init_val=-1.0))
        calibrate_activation_steering(model, {"hidden_states": inputs * 2}, adapter_name="other")
        torch.testing.assert_close(model(inputs), first)

        model.set_adapter("other")
        torch.testing.assert_close(model(inputs), reference + torch.tensor([0.0, -20.0]))

        model.base_model.set_adapter(["default", "other"])
        torch.testing.assert_close(model(inputs), reference + torch.tensor([6.0, -12.0]))

        model.base_model.set_adapter(["other", "default"])
        torch.testing.assert_close(model(inputs), reference + torch.tensor([6.0, -12.0]))

    def test_failed_recalibration_preserves_working_adapter(self, inputs):
        base = SteeringModel()
        reference = base(inputs)
        model = calibrated_model(base, steering_config(init_val=1.0), inputs)
        model.train()
        steered = model(inputs)

        def interrupted_data():
            yield {"hidden_states": inputs * 2}
            raise RuntimeError("Calibration data could not be loaded")

        with model.disable_adapter():
            with pytest.raises(RuntimeError, match="Calibration data could not be loaded"):
                calibrate_activation_steering(model, interrupted_data())
            torch.testing.assert_close(model(inputs), reference)
        assert model.training
        torch.testing.assert_close(model(inputs), steered)

        # A later calibration can succeed after the interrupted attempt.
        calibrate_activation_steering(model, {"hidden_states": inputs})
        assert model.training
        torch.testing.assert_close(model(inputs), steered)

    def test_uncalibrated_adapter_raises(self, inputs):
        model = get_peft_model(SteeringModel(), steering_config())
        with pytest.raises(RuntimeError, match="not calibrated"):
            model(inputs)

    def test_directly_injected_adapter_can_be_calibrated(self, inputs):
        base = SteeringModel()
        reference = base(inputs)
        model = inject_adapter_in_model(steering_config(init_val=1.0), base)
        calibrate_activation_steering(model, {"hidden_states": inputs})
        torch.testing.assert_close(model(inputs), reference + torch.tensor([6.0, 8.0]))

    @pytest.mark.parametrize("method", ["merge_adapter", "merge_and_unload", "unmerge_adapter"])
    def test_merging_is_unsupported(self, method):
        model = get_peft_model(SteeringModel(), steering_config())
        with pytest.raises(ValueError, match="does not support"):
            getattr(model, method)()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"indices": []},
            {"indices": [0, 0]},
            {"indices": [-1]},
            {"init_val": float("nan")},
            {"indices": [0, 1], "init_val": [1.0]},
            {"indices": [0, 1], "statistic": ["median_l2"]},
            {"statistic": "mean_l2"},
        ],
    )
    def test_invalid_steering_configuration_is_rejected(self, kwargs):
        with pytest.raises(ValueError):
            steering_config(**kwargs)

    @pytest.mark.parametrize(
        "target,source,indices",
        [("missing", "down_proj", [0]), ("block", "missing", [0]), ("block", "down_proj", [3])],
    )
    def test_unknown_target_or_neuron_is_rejected(self, target, source, indices):
        config = ActivationSteeringConfig(
            steering_pattern={target: {"source_module": source, "indices": indices, "init_val": 1.0}}
        )
        with pytest.raises(ValueError):
            get_peft_model(SteeringModel(), config)

    def test_unknown_calibration_adapter_is_rejected(self, inputs):
        model = get_peft_model(SteeringModel(), steering_config())
        with pytest.raises(ValueError, match="No activation steering layers"):
            calibrate_activation_steering(model, {"hidden_states": inputs}, adapter_name="missing")


@pytest.mark.parametrize(
    "model_id",
    [
        "peft-internal-testing/tiny-random-OPTForCausalLM",
        "peft-internal-testing/tiny-random-GPT2LMHeadModel",
        "peft-internal-testing/tiny-random-GPTJForCausalLM",
        "trl-internal-testing/tiny-random-LlamaForCausalLM",
        "peft-internal-testing/tiny-dummy-qwen2",
        "hf-internal-testing/tiny-random-Gemma3ForCausalLM",
    ],
)
class TestActivationSteeringDecoderModels(PeftCommonTester):
    # Block targets and explicit initial doses do not fit ALL_CONFIGS' projection-targeting assumptions.
    transformers_class = AutoModelForCausalLM

    @pytest.fixture(autouse=True)
    def calibrate_new_adapters(self, monkeypatch):
        # The common battery constructs fresh adapters; perform their calibration before exercising the shared checks.
        def create_calibrated_model(model, config, adapter_name="default", **kwargs):
            adapted = get_peft_model(model, config, adapter_name=adapter_name, **kwargs)
            device = next(adapted.parameters()).device
            batch = {name: value.to(device) for name, value in self.prepare_inputs_for_testing().items()}
            calibrate_activation_steering(adapted, {**batch, "use_cache": False}, adapter_name=adapter_name)
            return adapted

        monkeypatch.setattr(f"{PeftCommonTester.__module__}.get_peft_model", create_calibrated_model)

    @pytest.fixture
    def steering_kwargs(self, model_id):
        if "OPT" in model_id:
            block, source = "decoder.layers.0", "fc2"
        elif "GPT2" in model_id:
            block, source = "h.0", "mlp.c_proj"
        elif "GPTJ" in model_id:
            block, source = "h.0", "mlp.fc_out"
        else:
            block, source = "layers.0", "mlp.down_proj"
        return {
            "task_type": "CAUSAL_LM",
            "steering_pattern": {block: {"source_module": source, "indices": [0, 1], "init_val": [0.1, -0.1]}},
        }

    def prepare_inputs_for_testing(self):
        return {
            "input_ids": torch.tensor([[1, 1, 1], [1, 2, 1]]).to(self.torch_device),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 0, 1]]).to(self.torch_device),
        }

    def test_prepare_for_training(self, model_id, steering_kwargs):
        self._test_prepare_for_training(model_id, ActivationSteeringConfig, copy.deepcopy(steering_kwargs))

    def test_save_pretrained(self, model_id, steering_kwargs):
        self._test_save_pretrained(model_id, ActivationSteeringConfig, copy.deepcopy(steering_kwargs))

    def test_load_model_low_cpu_mem_usage(self, model_id, steering_kwargs):
        self._test_load_model_low_cpu_mem_usage(model_id, ActivationSteeringConfig, copy.deepcopy(steering_kwargs))

    def test_load_multiple_adapters(self, model_id, steering_kwargs):
        self._test_load_multiple_adapters(model_id, ActivationSteeringConfig, copy.deepcopy(steering_kwargs))

    def test_training(self, model_id, steering_kwargs):
        self._test_training(model_id, ActivationSteeringConfig, copy.deepcopy(steering_kwargs))

    def test_inference_safetensors(self, model_id, steering_kwargs):
        self._test_inference_safetensors(model_id, ActivationSteeringConfig, copy.deepcopy(steering_kwargs))

    def test_generate(self, model_id, steering_kwargs):
        self._test_generate(model_id, ActivationSteeringConfig, copy.deepcopy(steering_kwargs))

    def test_disable_adapter(self, model_id, steering_kwargs):
        self._test_disable_adapter(model_id, ActivationSteeringConfig, copy.deepcopy(steering_kwargs))

    def test_delete_adapter(self, model_id, steering_kwargs):
        self._test_delete_adapter(model_id, ActivationSteeringConfig, copy.deepcopy(steering_kwargs))

    def test_delete_inactive_adapter(self, model_id, steering_kwargs):
        self._test_delete_inactive_adapter(model_id, ActivationSteeringConfig, copy.deepcopy(steering_kwargs))

    def test_unload_adapter(self, model_id, steering_kwargs):
        self._test_unload_adapter(model_id, ActivationSteeringConfig, copy.deepcopy(steering_kwargs))

    def test_get_base_model_state_dict(self, model_id, steering_kwargs):
        self._test_get_base_model_state_dict(model_id, ActivationSteeringConfig, copy.deepcopy(steering_kwargs))

    @pytest.mark.parametrize("use_reentrant", [True, False])
    def test_gradient_checkpointing(self, model_id, steering_kwargs, use_reentrant):
        self._test_training_gradient_checkpointing(
            model_id, ActivationSteeringConfig, copy.deepcopy(steering_kwargs), use_reentrant=use_reentrant
        )
