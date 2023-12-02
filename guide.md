This conceptual guide gives a brief overview of LoRA, 


In principle, BOFT can be used similar to LoRA and be applied to any subset of weight matrices in a neural network to reduce the number of trainable parameters. Currently, BOFT follows the common practice of LoRA to be applied to attention blocks only. The resulting number of trainable parameters in a BOFT model depends on the number of blocks and the number of butterfly components.

## Utils for BOFT
Use merge_adapter() to merge the BOFT layers into the base model while retaining the PeftModel. This will help in later unmerging, deleting, loading different adapters and so on.

Use unmerge_adapter() to unmerge the LoRa layers from the base model while retaining the PeftModel. This will help in later merging, deleting, loading different adapters and so on.

Use unload() to get back the base model without the merging of the active lora modules. This will help when you want to get back the pretrained base model in some applications when you want to reset the model to its original state. For example, in Stable Diffusion WebUi, when the user wants to infer with base model post trying out LoRAs.

Use delete_adapter() to delete an existing adapter.

Use add_weighted_adapter() to combine multiple LoRAs into a new adapter based on the user provided weighing scheme.

## Common BOFT parameters in PEFT

To fine-tune a model using BOFT from PEFT, you need to:

1. Instantiate a base model.
2. Create a configuration (BOFTConfig) where you define BOFT-specific parameters.
3. Wrap the base model with get_peft_model() to get a trainable PeftModel.
4. Train the PeftModel as you normally would train the base model.

BOFTConfig allows you to control how BOFT is applied to the base model through the following parameters:

* 

```python
import transformers
from peft import BOFTConfig, get_peft_model

config = BOFTConfig(
    boft_block_size=8,
    boft_block_num=args.block_num,
    boft_n_butterfly_factor=args.n_butterfly_factor,
    target_modules=["query", "value", "key", "output.dense", "mlp.fc1", "mlp.fc2"],
    boft_dropout=args.boft_dropout,
    bias="boft_only",
    modules_to_save=["classifier"],
)

model = transformers.Dinov2ForImageClassification.from_pretrained(
    "facebook/dinov2-large",
    num_labels=100,
    )
boft_model = get_peft_model(model, config)
boft_model.print_trainable_parameters()
# output: 

from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
model_name_or_path = "bigscience/mt0-large"
tokenizer_name_or_path = "bigscience/mt0-large"

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# output: trainable params: 2359296 || all params: 1231940608 || trainable%: 0.19151053100118282
```

## Example Usage
You can initialize the BOFT config for a base model as follows:
```python
from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
model_name_or_path = "bigscience/mt0-large"
tokenizer_name_or_path = "bigscience/mt0-large"

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# output: trainable params: 2359296 || all params: 1231940608 || trainable%: 0.19151053100118282
```

## BOFT examples

We also provide some examples of applying BOFT to finetune diffusion model for various downstream tasks, please refere to the following guides:

* Dreambooth fine-tuning with BOFT
* ControlNet fine-tuning with BOFT

