# PiSSA: Principal Singular values and Singular vectors Adaptation
## Introduction ([Paper](https://arxiv.org/abs/2404.02948), [code](https://github.com/GraphPKU/PiSSA))
PiSSA represents a matrix $W\in\mathbb{R}^{m\times n}$ within the model by the product of two trainable matrices $A \in \mathbb{R}^{m\times r}$ and $B \in \mathbb{R}^{r\times n}$, where $r \ll \min(m, n)$, plus a residual matrix $W^{res}\in\mathbb{R}^{m\times n}$ for error correction. Singular value decomposition (SVD) is employed to factorize $W$, and the principal singular values and vectors of $W$ are utilized to initialize $A$ and $B$. The residual singular values and vectors initialize the residual matrix $W^{res}$, which keeps frozen during fine-tuning. This straightforward modification allows PiSSA to converge more rapidly than LoRA and ultimately attain superior performance. Moreover, PiSSA reduces the quantization error compared to QLoRA, leading to further enhancements.

## Quick Start
### Step 1. 
Utilize the decomposed models directly from the [Hugging Face Collections](https://huggingface.co/collections/fxmeng/pissa-661ce700721235e542a5d7a8).
If the existing settings do not meet your needs, apply [PiSSA initialization](https://github.com/fxmeng/peft/blob/606a69279480bbdea847f4e5247804bdf7e6b898/examples/pissa_finetuning/pissa_finetuning.py#L85-L103) to a pre-trained model and save the decomposed parameters:

```
# Load an original pre-processed dodel:
model = AutoModelForCausalLM.from_pretrained(...)

# Configure the initialization method to "pissa", which may take several minutes to execute SVD on the pre-trained model:
lora_config = LoraConfig(init_lora_weights="pissa", ...) 

# Alternatively, execute fast SVD, which takes only a few seconds. The number of iterations determines the trade-off between the error and computation time:
# lora_config = LoraConfig(init_lora_weights="pissa_niter_[number of iters]", ...) 

# Perform PiSSA on the original model according to lora_config:
peft_model = get_peft_model(model, lora_config)
```
We utilize the fast SVD method to reduce the initialization time. The lower the rank and the fewer the iterations, the less time is needed. However, this approach leads to higher errors in the computed matrices $A$ and $B$. To preserve the model's initial capabilities, we calculate $W^{res} = W - BA$. Even with potential errors in $A$ and $B$, the sum of $W^{res}$ and $BA$ accurately equals $W$.

### Step 2 (Recommend)
[Saving the residual model and PiSSA Adapter](https://github.com/fxmeng/peft/blob/51161a52cac3a736d931d90e676b24a32c4f8cd6/src/peft/utils/pissa_utils.py#L27-L51):
```
pissa_pre_training_saving(peft_model, saving_path, ...)
```
Store the decomposed model locally to prevent the necessity of re-decomposition across multiple fine-tuning experiments, and reducing errors when quantization of the residual model pre-decomposed at full precision.

### Step 3 (Optional)
If quantization fine-tuning is desired, reload the pre-processed [residual model](https://github.com/fxmeng/peft/blob/606a69279480bbdea847f4e5247804bdf7e6b898/examples/pissa_finetuning/pissa_finetuning.py#L107-L116) in 4-bit or 8-bit configurations along with the full-precision [PiSSA Adapter](https://github.com/fxmeng/peft/blob/606a69279480bbdea847f4e5247804bdf7e6b898/examples/pissa_finetuning/pissa_finetuning.py#L122):
```
res_model = AutoModelForCausalLM.from_pretrained(saving_path, load_in_4/8bit=True, ...)
peft_model = PeftModel.from_pretrained(res_model, f"{saving_path}/pissa_init", is_trainable=True)
```
When SVD is conducted at full precision, the PiSSA adapter retains the high-frequency principal components of the original model. 
Then quantizing the residual model, rather than the original model, notably decreases the quantization error.

### Step 4. 
[Training](https://github.com/fxmeng/peft/blob/51161a52cac3a736d931d90e676b24a32c4f8cd6/examples/pissa_finetuning/pissa_finetuning.py#L131-L139) the principal singular values and singular vectors results in faster convergence and enhanced performance:
```
dataset = load_dataset(...)
trainer = SFTTrainer(peft_model, dataset, ...)
peft_model.save_pretrained(os.path.join(args.output_path, "pissa_init"))
trainer.train()
peft_model.save_pretrained(os.path.join(args.output_path, "pissa_ft"))
```

### Step 5. 
Upon completion of training, it is recommended to [convert PiSSA into LoRA](https://github.com/fxmeng/peft/blob/51161a52cac3a736d931d90e676b24a32c4f8cd6/src/peft/utils/pissa_utils.py#L60-L99) for storage-efficient sharing:


```
pissa_post_training_saving(
    init_path = f"{saving_path}/pissa_init",
    finetuned_path = f"{saving_path}/pissa_ft",
    output_path = f"{saving_path}/pissa_lora",
)
```
Convert PiSSA to LoRA according to $\Delta W = A \times B - A_0 \times B_0 =  [A | A_0] \times [B | -B_0]^T=A^{'}B^{'}$.
Using the converted LoRA does not require modifying the parameters of the base model. When multiple converted LoRAs are needed simultaneously, each adapter operates independently without interference, allowing for the adapters to be freely deleted or added.

## Citation
```
@article{meng2024pissa,
  title={PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models},
  author={Meng, Fanxu and Wang, Zhaohui and Zhang, Muhan},
  journal={arXiv preprint arXiv:2404.02948},
  year={2024}
}
```