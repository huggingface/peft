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

"""Calibrate the Qwen3.5-4B intervention and reload it through Transformers' PEFT integration.

Requires this PEFT checkout and a Transformers version supporting Qwen3.5.

    python examples/activation_steering/qwen3_5.py calibrate
    python examples/activation_steering/qwen3_5.py generate --prompt "Tell me a short story."

Calibration runs a baseline forward without optimization. The signed doses remain at their configured values.
The saved adapter contains both the doses and the measured scales, so inference needs no recalibration.
"""

import argparse

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from peft import ActivationSteeringConfig, PeftConfig, TaskType
from peft.tuners.activation_steering import calibrate_activation_steering


MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

# The original author's fallback calibration text, copied from:
# https://github.com/lyramakesmusic/neuron-explorer/blob/5a8f8267f6bd3c8caa87d598870949ab0d25b914/explorer/runtime.py
# This uses the fallback text rather than the author's precomputed corpus statistics.
REFERENCE_TEXT = (
    "The history of the region is long and varied. Farmers settled the river valley first, then traders arrived, "
    "and by the twelfth century the town had a market, a bridge, and a quarrel with its neighbours over tolls. "
    "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0) if a[0] < b[0] else b.pop(0))\n"
    "    return out + a + b\nShe asked me what I thought of the plan, and I told her honestly that it seemed rushed. "
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-path", default="qwen3.5-4b-steering")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("calibrate", help="Measure baseline residual scales and save the adapter.")
    generate_parser = commands.add_parser("generate", help="Load the saved adapter using Transformers and generate.")
    generate_parser.add_argument("--prompt", default="Tell me a short story.")
    generate_parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    if args.command == "calibrate":
        config = ActivationSteeringConfig(
            task_type=TaskType.CAUSAL_LM,
            revision=MODEL_REVISION,
            steering_pattern={
                "layers.13": {
                    "source_module": "mlp.down_proj",
                    "indices": [7956],
                    "init_val": -1.0,
                    "statistic": "median_l2",
                },
                "layers.14": {
                    "source_module": "mlp.down_proj",
                    "indices": [8949],
                    "init_val": 1.0,
                    "statistic": "median_l2",
                },
            },
        )
        model_id, revision = MODEL_ID, MODEL_REVISION
    else:
        config = PeftConfig.from_pretrained(args.adapter_path)
        model_id, revision = config.base_model_name_or_path, config.revision

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id, revision=revision, dtype=getattr(torch, args.dtype), device_map=args.device_map
    )
    model.eval()

    if args.command == "calibrate":
        model.add_adapter(config)
        inputs = tokenizer(REFERENCE_TEXT, return_tensors="pt").to(model.device)
        calibrate_activation_steering(model, {**inputs, "use_cache": False})
        model.save_pretrained(args.adapter_path)
        tokenizer.save_pretrained(args.adapter_path)
        print(f"Saved calibrated steering adapter to {args.adapter_path}")
    else:
        model.load_adapter(args.adapter_path)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        print(tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True))


if __name__ == "__main__":
    main()
