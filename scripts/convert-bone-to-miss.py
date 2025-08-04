#!/usr/bin/env python3
# Copyright (c) 2025 Your Organization/Project. All rights reserved.
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

"""Convert Bone checkpoint to MiSS format."""

import argparse
import json
import os
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from peft.utils import CONFIG_NAME, SAFETENSORS_WEIGHTS_NAME


def convert_bone_to_miss(bone_dir: Path, miss_dir: Path) -> None:
    """Convert Bone checkpoint files to MiSS format."""
    bone_config_path = bone_dir / CONFIG_NAME
    miss_config_path = miss_dir / CONFIG_NAME
    if not os.path.exists(miss_dir):
        os.makedirs(miss_dir, exist_ok=True)
    with open(bone_config_path, encoding="utf-8") as f:
        config = json.load(f)

    config["peft_type"] = "MISS"

    with open(miss_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    bone_weight_path = bone_dir / SAFETENSORS_WEIGHTS_NAME
    miss_weight_path = miss_dir / SAFETENSORS_WEIGHTS_NAME

    new_data = {}

    with safe_open(bone_weight_path, framework="pt") as f:
        for old_key in f.keys():
            tensor = f.get_tensor(old_key)
            new_key = old_key.replace(".bone_", ".miss_")
            new_data[new_key] = tensor

    save_file(new_data, miss_weight_path)

    print(f"Converted checkpoint saved at {miss_weight_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Bone checkpoint to MiSS format.")
    parser.add_argument("bone_dir", type=Path, help="Directory containing Bone checkpoint files")
    parser.add_argument("miss_dir", type=Path, help="Directory to save MiSS checkpoint files")
    args = parser.parse_args()

    args.miss_dir.mkdir(parents=True, exist_ok=True)
    convert_bone_to_miss(args.bone_dir, args.miss_dir)


if __name__ == "__main__":
    main()
