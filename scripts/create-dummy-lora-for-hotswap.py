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

"""Create a dummy LoRA config from adapter directories or Hub IDs, without loading model weights.

The purpose of this helper function is to create a dummy LoRA config which can be used to initialze a LoRA model before
hotswapping different LoRA adapters (see https://huggingface.co/docs/peft/main/en/package_reference/hotswap). One issue
with hotswapping is that the first LoRA adapter that is loaded needs to install LoRA weights on all layers that the
later, hotswapped LoRAs will target. If these LoRAs target disjunct layers, none of them can serve as the first
LoRA. This helper function creates a dummy LoRA that targets the union of the layers of all the individual LoRAs. It
also pre-allocates LoRA weights with a sufficiently large rank.


Usage:

```sh
python scripts/create-dummy-lora-for-hotswap.py adapter0 adapter1 [...] --output-dir dummy-lora
```
"""

import argparse
from pathlib import Path

from peft import PeftConfig
from peft.helpers import create_dummy_lora_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapters", nargs="+", help="Adapter directories or Hugging Face Hub repository IDs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dummy-lora"),
        help="Directory for adapter_config.json (default: dummy-lora)",
    )
    args = parser.parse_args()
    configs = [PeftConfig.from_pretrained(adapter) for adapter in args.adapters]
    try:
        config = create_dummy_lora_config(configs)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    config.save_pretrained(args.output_dir)
    print(f"Saved dummy LoRA configuration to {args.output_dir / 'adapter_config.json'} (no weights).")


if __name__ == "__main__":
    main()
