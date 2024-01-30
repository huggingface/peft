<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Model merging

Training a model for each task can be costly and take up storage, and these models aren't able to learn new information to improve performance. Multitask learning can train a model to learn multiple tasks, but this is costly to train and designing a dataset for it can be challenging. *Model merging* offers a solution to these challenges by combining multiple pretrained models into one model, giving it the combined abilities of each individual model, without any additional training.

PEFT provides two methods for model merging:

* [TIES-Merging](https://hf.co/papers/2306.01708) - TrIm, Elect, and Merge (TIES) is a three-step method for merging models. First, redundant parameters are trimmed, then conflicting signs are resolved into an aggregated vector, and finally the parameters whose signs are the same as the aggregate sign are averaged. This method takes into account that some values (redundant and sign disagreement) can reduce model performance in the merged model.
* [DARE](https://hf.co/papers/2311.03099) - Drop And REscale is a method that can be used to prepare model merging methods like TIES-Merging. It works by randomly dropping parameters according to a drop rate, and rescaling the remaining parameters. This helps to reduce the number of redundant and potentially interfering parameters among multiple models.

Models are merged with the [`~LoraModel.add_weighted_adapter`] method, and the specific model merging method is specified in the `combination_type` parameter. This guide will show you how to merge models with TIES, DARE, and a combination of both TIES and DARE.

## TIES

The [`~utils.ties`] method uses a [`~utils.magnitude_based_pruning`] approach to trim redundant parameters such that only the top-k percent of values are kept from each task vector. The number of values to keep are specified by the `density` parameter. The task tensors are weighted, and the [`~utils.calculate_majority_sign_mask`] *elects* the sign vector which means calculating the total magnitude for each parameter across all models. Lastly, the [`~utils.disjoint_merge`] function calculates the mean of the parameter values whose sign is the same as the *elected sign vector*.

With PEFT, TIES merging is enabled by setting `combination_type="ties"` and setting `ties_density` to a value of the weights to keep from the individual models. For example, let's merge three [TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T) models trained with LoRA: [tinyllama_lora_nobots](https://huggingface.co/smangrul/tinyllama_lora_norobots), [tinyllama_lora_sql](https://huggingface.co/smangrul/tinyllama_lora_sql), and [tinyllama_lora_adcopy](https://huggingface.co/smangrul/tinyllama_lora_adcopy).

Load the base model and then use the [`~PeftModel.load_adapter`] method to load and assign each adapter a name:

```py

```

Use [`~LoraModel.add_weighted_adapter`] to set the weights for each adapter, the `adapter_name`, the `combination_type`, and `ties_density`.

```py

```

Make the newly merged model the active model with the [`~LoraModel.set_adapter`] method, using the new `adapter_name`.

```py

```

Now you can use the merged model as an instruction-tuned model to write ad copy or SQL queries!

<hfoptions id="ties">
<hfoption id="instruct">

</hfoption>
<hfoption id="ad copy">

</hfoption>
<hfoption id="SQL">

</hfoption>
</hfoptions>

## DARE

The DARE method uses the [`~utils.random_pruning`] approach to randomly drop parameters, and only preserving a percentage of the parameters set in the `density` parameter. The remaining tensors are rescaled to keep the expected output unchanged.

With PEFT, DARE is enabled by setting `combination_type="dare_ties"` and setting `density` to a value of the weights to keep from the individual models.

> [!TIP]
> DARE is a super useful method for preparing models for merging which means it can be combined with other methods like TIES, `linear` (a weighted average of the task tensors) or `svd` (calculated from the *delta weights*, the model parameters before and after finetuning) or a combination of all of the above like `dare_ties_svd`.

Let's merge three diffusion models to generate a variety of images in different styles using only one model. The models you'll use are (feel free to choose your own): [nerijs/pixel-art-xl](https://huggingface.co/nerijs/pixel-art-xl), [ostris/super-cereal-sdxl-lora](https://huggingface.co/ostris/super-cereal-sdxl-lora), and [KappaNeuro/studio-ghibli-style](https://huggingface.co/KappaNeuro/studio-ghibli-style).

```py

```

Load the base model and then use the [`~PeftModel.load_adapter`] method to load and assign each adapter a name:

```py

```

Use [`~LoraModel.add_weighted_adapter`] to set the weights for each adapter, the `adapter_name`, the `combination_type`, and `ties_density`.

```py

```

Make the newly merged model the active model with the [`~LoraModel.set_adapter`] method, using the new `adapter_name`.

```py

```

Now you can use the merged model to generate images in three different styles!

<hfoptions id="dare">
<hfoption id="pixel art">

</hfoption>
<hfoption id="cereal box cover">

</hfoption>
<hfoption id="Studio Ghibli">

</hfoption>
</hfoptions>
