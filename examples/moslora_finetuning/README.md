# Mixture-of-Subspaces in Low-Rank Adaptation

![MoSLoRA](https://raw.githubusercontent.com/wutaiqiang/MoSLoRA/refs/heads/main/method.png)


## Introduction
[MoSLoRA](https://arxiv.org/abs/2406.11909) is a novel approach that is computationally efficient, easy to implement, and readily applicable to large language, multimodal, and diffusion models. Initially, we equivalently decompose the weights of LoRA into two subspaces, and find that simply mixing them can enhance performance. To study such a phenomenon, we revisit it through a fine-grained subspace lens, showing that such modification is equivalent to employing a fixed mixer to fuse the subspaces. To be more flexible, we jointly learn the mixer with the original LoRA weights, and term the method Mixture-of-Subspaces LoRA (MoSLoRA). MoSLoRA consistently outperforms LoRA on tasks in different modalities, including commonsense reasoning, visual instruction tuning, and subject-driven text-to-image generation, demonstrating its effectiveness and robustness. 

## Quick start
```python
import torch
from peft import MoSLoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
lora_config = MoSLoraConfig(
    use_moslora=True
)
peft_model = get_peft_model(model, lora_config)
trainer = transformers.Trainer(
    model=peft_model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
)
trainer.train()
peft_model.save_pretrained("moslora-tinyllama-1.1")
```

There is no additional change needed to your standard LoRA procedure, except for replacing __LoraConfig__ with __MoSLoraConfig__ and set `use_moslora=True` option in your configuration. If you set the `use_moslora=False` then the training process would be the same as LoRA.


Run the finetuning script simply by running:
```bash
python examples/moslora_finetuning/moslora_finetuning.py --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --data_path timdettmers/openassistant-guanaco
```
This 👆🏻 by default will load the model in peft set up with LoRA config. Now if you wanna quickly compare it with MoSLoRA, all you need to do is to input ` --use_moslora` in the command line. So same above example would be 👇🏻;

```bash
python examples/moslora_finetuning/moslora_finetuning.py --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --data_path timdettmers/openassistant-guanaco --use_moslora True
```

Moreover, you can set use_moslora as `"kai"` for Kaiming Uniform initilization or `"orth"` for orthogonal initilization.


Similarly, by default the LoRA layers are the attention and MLP layers of LLama model, if you get to choose a different set of layers for LoRA to be applied on, you can simply define it using:
```bash
python examples/moslora_finetuning/moslora_finetuning.py --lora_target_modules "q_proj,k_proj,v_proj,o_proj" 
```

### Full example of the script 
```bash
python dora_finetuning.py \
    --base_model "PATH_TO_MODEL" \
    --data_path "PATH_TO_DATASET" \
    --output_dir "PATH_TO_OUTPUT_DIR" \
    --batch_size 1 \
    --num_epochs 3 \
    --learning_rate 3e-4 \
    --cutoff_len 512 \
    --val_set_size 500 \
    --use_moslora "kai" \
    --quantize \
    --eval_step 10 \
    --save_step 100 \
    --device "cuda:0" \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj" \
    --hub_model_id "YOUR_HF_REPO" \
    --push_to_hub
```
## Use the model on 🤗
You can load and use the model as any other 🤗 models.
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("ShirinYamani/huggyllama-llama-7b-finetuned")
```



## Citation
```
@inproceedings{wu-etal-2024-mixture-subspaces,
    title = "Mixture-of-Subspaces in Low-Rank Adaptation",
    author = "Wu, Taiqiang  and
      Wang, Jiahao  and
      Zhao, Zhe  and
      Wong, Ngai",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.450",
    doi = "10.18653/v1/2024.emnlp-main.450",
    pages = "7880--7899",
}
```