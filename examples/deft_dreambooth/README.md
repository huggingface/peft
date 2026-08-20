<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# DreamBooth fine-tuning with DEFT

[DEFT](https://proceedings.neurips.cc/paper_files/paper/2025/hash/93a34a7138bdad95e874018d5f491cc6-Abstract-Conference.html)
(Decompositional Efficient Fine-Tuning) adapts a frozen weight by *removing* a learned low-rank sub-space and
*injecting* a new one in its place (`W' = (I - P_proj) @ W + Q_P @ R`). On its native text-to-image domain it is well
suited to personalizing a diffusion model from a few images while preserving the base model's editability. This example
is adapted from [`oft_dreambooth`](https://github.com/huggingface/peft/tree/main/examples/oft_dreambooth).

## Setup

```bash
cd peft/examples/deft_dreambooth
pip install "git+https://github.com/huggingface/peft" diffusers accelerate transformers
```

## Train

Point `--instance_data_dir` at a few images of your subject:

```bash
python train_dreambooth.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1-base" \
    --instance_data_dir "path/to/subject/images" \
    --output_dir "deft-dreambooth-model" \
    --instance_prompt "a photo of sks dog" \
    --resolution 512 \
    --train_batch_size 1 \
    --max_train_steps 800 \
    --learning_rate 1e-4 \
    --use_deft \
    --deft_r 8 \
    --deft_alpha 16 \
    --deft_decomposition_method "qr"
```

`qr` is the default decomposition and works best for image generation (use `relu` for text tasks). Add
`--train_text_encoder` (with the `--deft_text_encoder_*` options) to also adapt the text encoder.

## Inference

See [`deft_dreambooth_inference.ipynb`](./deft_dreambooth_inference.ipynb): load the base pipeline and attach the
trained adapters with `PeftModel.from_pretrained(pipe.unet, output_dir + "/unet")` (and likewise for the text encoder
if it was trained).
