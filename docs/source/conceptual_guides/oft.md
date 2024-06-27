<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Orthogonal Finetuning (OFT and BOFT) 

This conceptual guide gives a brief overview of [OFT](https://arxiv.org/abs/2306.07280) and [BOFT](https://arxiv.org/abs/2311.06243), a parameter-efficient fine-tuning technique that utilizes orthogonal matrix to multiplicatively transform the pretrained weight matrices.

To achieve efficient fine-tuning, OFT represents the weight updates with an orthogonal transformation. The orthogonal transformation is parameterized by an orthogonal matrix multiplied to the pretrained weight matrix. These new matrices can be trained to adapt to the new data while keeping the overall number of changes low. The original weight matrix remains frozen and doesn’t receive any further adjustments. To produce the final results, both the original and the adapted weights are multiplied togethor.

Orthogonal Butterfly (BOFT) generalizes OFT with Butterfly factorization and further improves its parameter efficiency and finetuning flexibility. In short, OFT can be viewed as a special case of BOFT. Different from LoRA that uses additive low-rank weight updates, BOFT uses multiplicative orthogonal weight updates. The comparison is shown below.

<div class="flex justify-center">
    <img src="https://raw.githubusercontent.com/wy1iu/butterfly-oft/main/assets/BOFT_comparison.png"/>
</div>


BOFT has some advantages compared to LoRA: 

* BOFT proposes a simple yet generic way to finetune pretrained models to downstream tasks, yielding a better preservation of pretraining knowledge and a better parameter efficiency.
* Through the orthogonality, BOFT introduces a structural constraint, i.e., keeping the [hyperspherical energy](https://arxiv.org/abs/1805.09298) unchanged during finetuning. This can effectively reduce the forgetting of pretraining knowledge.
* BOFT uses the butterfly factorization to efficiently parameterize the orthogonal matrix, which yields a compact yet expressive learning space (i.e., hypothesis class).
* The sparse matrix decomposition in BOFT brings in additional inductive biases that are beneficial to generalization.

In principle, BOFT can be applied to any subset of weight matrices in a neural network to reduce the number of trainable parameters. Given the target layers for injecting BOFT parameters, the number of trainable parameters can be determined based on the size of the weight matrices.

## Merge OFT/BOFT weights into the base model

Similar to LoRA, the weights learned by OFT/BOFT can be integrated into the pretrained weight matrices using the merge_and_unload() function. This function merges the adapter weights with the base model which allows you to effectively use the newly merged model as a standalone model.

<div class="flex justify-center">
    <img src="https://raw.githubusercontent.com/wy1iu/butterfly-oft/main/assets/boft_merge.png"/>
</div>

This works because during training, the orthogonal weight matrix (R in the diagram above) and the pretrained weight matrices are separate. But once training is complete, these weights can actually be merged (multiplied) into a new weight matrix that is equivalent.

## Utils for OFT / BOFT

### Common OFT / BOFT parameters in PEFT

As with other methods supported by PEFT, to fine-tune a model using OFT or BOFT, you need to:

1. Instantiate a base model.
2. Create a configuration (`OFTConfig` or `BOFTConfig`) where you define OFT/BOFT-specific parameters.
3. Wrap the base model with `get_peft_model()` to get a trainable `PeftModel`.
4. Train the `PeftModel` as you normally would train the base model.


### BOFT-specific paramters

`BOFTConfig` allows you to control how OFT/BOFT is applied to the base model through the following parameters:

- `boft_block_size`: the BOFT matrix block size across different layers, expressed in `int`. Smaller block size results in sparser update matrices with fewer trainable paramters. **Note**, please choose `boft_block_size` to be divisible by most layer's input dimension (`in_features`), e.g., 4, 8, 16. Also, please only 
specify either `boft_block_size` or `boft_block_num`, but not both simultaneously or leaving both to 0, because `boft_block_size` x `boft_block_num` must equal the layer's input dimension.
- `boft_block_num`: the number of BOFT matrix blocks across different layers, expressed in `int`. Fewer blocks result in sparser update matrices with fewer trainable paramters. **Note**, please choose `boft_block_num` to be divisible by most layer's input dimension (`in_features`), e.g., 4, 8, 16. Also, please only 
specify either `boft_block_size` or `boft_block_num`, but not both simultaneously or leaving both to 0, because `boft_block_size` x `boft_block_num` must equal the layer's input dimension.
- `boft_n_butterfly_factor`: the number of butterfly factors. **Note**, for `boft_n_butterfly_factor=1`, BOFT is the same as vanilla OFT, for `boft_n_butterfly_factor=2`, the effective block size of OFT becomes twice as big and the number of blocks become half.
- `bias`: specify if the `bias` parameters should be trained. Can be `"none"`, `"all"` or `"boft_only"`.
- `boft_dropout`: specify the probability of multiplicative dropout.
- `target_modules`: The modules (for example, attention blocks) to inject the OFT/BOFT matrices.
- `modules_to_save`: List of modules apart from OFT/BOFT matrices to be set as trainable and saved in the final checkpoint. These typically include model's custom head that is randomly initialized for the fine-tuning task.



## BOFT Example Usage

For an example of the BOFT method application to various downstream tasks, please refer to the following guides:

Take a look at the following step-by-step guides on how to finetune a model with BOFT:
- [Dreambooth finetuning with BOFT](../task_guides/boft_dreambooth) 
- [Controllable generation finetuning with BOFT (ControlNet)](../task_guides/boft_controlnet) 

For the task of image classification, one can initialize the BOFT config for a DinoV2 model as follows:

```py
import transformers
from transformers import AutoModelForSeq2SeqLM, BOFTConfig
from peft import BOFTConfig, get_peft_model

config = BOFTConfig(
    boft_block_size=4,
    boft_n_butterfly_factor=2,
    target_modules=["query", "value", "key", "output.dense", "mlp.fc1", "mlp.fc2"],
    boft_dropout=0.1,
    bias="boft_only",
    modules_to_save=["classifier"],
)

model = transformers.Dinov2ForImageClassification.from_pretrained(
    "facebook/dinov2-large",
    num_labels=100,
)

boft_model = get_peft_model(model, config)
```
