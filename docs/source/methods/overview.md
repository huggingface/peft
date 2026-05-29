<!--Copyright 2024-present The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->





# Prompt-based methods

 Prompting primes a frozen pretrained model for a specific downstream task by including a text prompt that describes the task or even demonstrates an example of the task. With prompting, you can avoid fully training a separate model for each downstream task, and use the same frozen pretrained model instead. This is a lot easier because you can use the same model for several different tasks, and it is significantly more efficient to train and store a smaller set of prompt parameters than to train all the model's parameters.

There are two categories of prompting methods:

- hard prompts are manually handcrafted text prompts with discrete input tokens; the downside is that it requires a lot of effort to create a good prompt
- soft prompts are learnable tensors concatenated with the input embeddings that can be optimized to a dataset; the downside is that they aren't human readable because you aren't matching these "virtual tokens" to the embeddings of a real word

The PEFT library supports several types of prompting methods (p-tuning, prefix tuning, prompt tuning, ...), explore the table of contents for a full listing of soft prompt methods.
If you're interested in applying these methods to other tasks and use cases, take a look at our [notebook collection](https://huggingface.co/spaces/PEFT/soft-prompting)!

> [!TIP]
> Some familiarity with the general process of training a causal language model would be really helpful and allow you to focus on the soft prompting methods. If you're new, we recommend taking a look at the [Causal language modeling](https://huggingface.co/docs/transformers/tasks/language_modeling) guide first from the Transformers documentation. When you're ready, come back and see how easy it is to drop PEFT in to your training!






### PEFT configuration and model

For any PEFT method, you'll need to create a configuration which contains all the parameters that specify how the PEFT method should be applied. Once the configuration is setup, pass it to the [`~peft.get_peft_model`] function along with the base model to create a trainable [`PeftModel`].

> [!TIP]
> Call the [`~PeftModel.print_trainable_parameters`] method to compare the number of trainable parameters of [`PeftModel`] versus the number of parameters in the base model!

<hfoptions id="configurations">
<hfoption id="p-tuning">

[P-tuning](../conceptual_guides/prompting#p-tuning) adds a trainable embedding tensor where the prompt tokens can be added anywhere in the input sequence. Create a [`PromptEncoderConfig`] with the task type, the number of virtual tokens to add and learn, and the hidden size of the encoder for learning the prompt parameters.

```py
from peft import PromptEncoderConfig, get_peft_model

peft_config = PromptEncoderConfig(task_type="CAUSAL_LM", num_virtual_tokens=20, encoder_hidden_size=128)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"trainable params: 300,288 || all params: 559,514,880 || trainable%: 0.05366935013417338"
```

</hfoption>
<hfoption id="prefix tuning">

[Prefix tuning](../conceptual_guides/prompting#prefix-tuning) adds task-specific parameters in all of the model layers, which are optimized by a separate feed-forward network. Create a [`PrefixTuningConfig`] with the task type and number of virtual tokens to add and learn.

```py
from peft import PrefixTuningConfig, get_peft_model

peft_config = PrefixTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=20)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"trainable params: 983,040 || all params: 560,197,632 || trainable%: 0.1754809274167014"
```

</hfoption>
<hfoption id="prompt tuning">

[Prompt tuning](../conceptual_guides/prompting#prompt-tuning) formulates all tasks as a *generation* task and it adds a task-specific prompt to the input which is updated independently. The `prompt_tuning_init_text` parameter specifies how to finetune the model (in this case, it is classifying whether tweets are complaints or not). For the best results, the `prompt_tuning_init_text` should have the same number of tokens that should be predicted. To do this, you can set `num_virtual_tokens` to the number of tokens of the `prompt_tuning_init_text`.

Create a [`PromptTuningConfig`] with the task type, the initial prompt tuning text to train the model with, the number of virtual tokens to add and learn, and a tokenizer.

```py
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model

prompt_tuning_init_text = "Classify if the tweet is a complaint or no complaint.\n"
peft_config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=len(tokenizer(prompt_tuning_init_text)["input_ids"]),
    prompt_tuning_init_text=prompt_tuning_init_text,
    tokenizer_name_or_path="bigscience/bloomz-560m",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"trainable params: 8,192 || all params: 559,222,784 || trainable%: 0.0014648902430985358"
```

</hfoption>
</hfoptions>

