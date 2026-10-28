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
    python examples/activation_steering/qwen3_5.py --preset sarcastiqwen calibrate
    python examples/activation_steering/qwen3_5.py --preset sarcastiqwen generate --prompt "Tell me a short story."

Calibration runs a baseline forward without optimization. The signed doses remain at their configured values.
The saved adapter contains both the doses and the measured scales, so inference needs no recalibration.
"""

import argparse
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList, Qwen3_5ForConditionalGeneration, set_seed
from transformers.utils import logging

from peft import PeftConfig
from peft.tuners.activation_steering import calibrate_activation_steering


PRESETS_DIR = Path(__file__).resolve().parent / "presets"

# The original author's fallback calibration text, copied from:
# https://github.com/lyramakesmusic/neuron-explorer/blob/5a8f8267f6bd3c8caa87d598870949ab0d25b914/explorer/runtime.py#L21
REFERENCE_TEXT = (
    "The history of the region is long and varied. Farmers settled the river valley first, then traders arrived, "
    "and by the twelfth century the town had a market, a bridge, and a quarrel with its neighbours over tolls. "
    "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0) if a[0] < b[0] else b.pop(0))\n"
    "    return out + a + b\nShe asked me what I thought of the plan, and I told her honestly that it seemed rushed. "
)


class PresencePenaltyLogitsProcessor(LogitsProcessor):
    """Subtract a fixed penalty once per distinct generated token, excluding the prompt."""

    def __init__(self, penalty: float, prompt_length: int):
        self.penalty = penalty
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        penalties = torch.zeros_like(scores)
        penalties.scatter_(1, input_ids[:, self.prompt_length :], self.penalty)
        return scores - penalties


def main():
    logging.set_verbosity_error()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=sorted(path.name for path in PRESETS_DIR.iterdir() if (path / "adapter_config.json").is_file()),
        default="evil",
        help="Intervention to calibrate and default checkpoint directory to use.",
    )
    parser.add_argument("--adapter-path", help="Checkpoint directory; defaults to a separate directory per preset.")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    commands = parser.add_subparsers(dest="command", required=True)
    calibrate_parser = commands.add_parser("calibrate", help="Measure baseline residual scales and save the adapter.")
    calibrate_parser.add_argument(
        "--reference-text", default=REFERENCE_TEXT, help="Calibration text; defaults to the author's fallback text."
    )
    generate_parser = commands.add_parser("generate", help="Load the saved adapter using Transformers and generate.")
    generate_parser.add_argument("--prompt", default="Tell me a short story.")
    generate_parser.add_argument("--max-new-tokens", type=int, default=256)
    generate_parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    generate_parser.add_argument("--temperature", type=float, default=0.7)
    generate_parser.add_argument("--top-p", type=float, default=0.8)
    generate_parser.add_argument("--top-k", type=int, default=20)
    generate_parser.add_argument("--presence-penalty", type=float, default=1.5)
    generate_parser.add_argument("--repetition-penalty", type=float, default=1.0)
    generate_parser.add_argument("--seed", type=int, help="Random seed for repeatable sampling on the same setup.")
    generate_parser.add_argument("--disable-adapter", action="store_true", help="Generate with steering disabled.")
    args = parser.parse_args()

    if args.command == "generate":
        if args.max_new_tokens <= 0:
            parser.error("--max-new-tokens must be positive.")
        if not math.isfinite(args.temperature) or args.temperature <= 0:
            parser.error("--temperature must be finite and positive; use --no-do-sample for greedy decoding.")
        if not 0 < args.top_p <= 1:
            parser.error("--top-p must be in (0, 1].")
        if args.top_k < 0:
            parser.error("--top-k must be non-negative.")
        if not -2 <= args.presence_penalty <= 2:
            parser.error("--presence-penalty must be in [-2, 2].")
        if not math.isfinite(args.repetition_penalty) or args.repetition_penalty <= 0:
            parser.error("--repetition-penalty must be finite and positive.")
        if args.seed is not None:
            set_seed(args.seed)

    if args.adapter_path is None:
        suffix = "" if args.preset == "original" else f"-{args.preset}"
        args.adapter_path = f"qwen3.5-4b-steering{suffix}"

    config_path = PRESETS_DIR / args.preset if args.command == "calibrate" else args.adapter_path
    config = PeftConfig.from_pretrained(str(config_path))
    model_id, revision = config.base_model_name_or_path, config.revision

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id, revision=revision, dtype=getattr(torch, args.dtype), device_map=args.device_map
    )
    model.eval()

    if args.command == "calibrate":
        model.add_adapter(config)
        inputs = tokenizer(args.reference_text, return_tensors="pt").to(model.device)
        calibrate_activation_steering(model, {**inputs, "use_cache": False})
        model.save_pretrained(args.adapter_path)
        tokenizer.save_pretrained(args.adapter_path)
        print(f"Saved calibrated steering adapter to {args.adapter_path}")
    else:
        model.load_adapter(args.adapter_path)
        if args.disable_adapter:
            model.disable_adapters()
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
        logits_processor = LogitsProcessorList()
        if args.presence_penalty != 0:
            logits_processor.append(
                PresencePenaltyLogitsProcessor(args.presence_penalty, prompt_length=inputs.input_ids.shape[1])
            )
        generation_kwargs = {"do_sample": args.do_sample, "repetition_penalty": args.repetition_penalty}
        if args.do_sample:
            generation_kwargs.update(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, min_p=0.0)
        with torch.inference_mode():
            # The pinned config only stops on <|endoftext|>; chat turns end with <|im_end|>.
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                logits_processor=logits_processor,
                eos_token_id=tokenizer.convert_tokens_to_ids(["<|endoftext|>", "<|im_end|>"]),
                pad_token_id=tokenizer.pad_token_id,
                **generation_kwargs,
            )
        print(tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True))


if __name__ == "__main__":
    main()
