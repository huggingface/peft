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

# Prefix tuning

[Prefix tuning](https://hf.co/papers/2101.00190) prefixes a series of task-specific vectors to the input sequence that can be learned while keeping the pretrained model frozen. The prefix parameters are inserted in all of the model layers.

**Note** For encoder-decoder models (seq2seq), the prefix is only applied to the decoder, which does not correspond to the paper specification (see e.g. Figure 2). Prefix tuning can still be fine-tuned on these model architectures but the performance could be sub-par; consider using other PEFT methods for encoder-decoder models.

## Initialize from a KV cache prefix

By default, prefix tuning is randomly initialized.

PEFT also provides utilities to initialize a prefix-tuning adapter from an existing KV cache prefix (for example, from
the first `p` tokens of a prompt/corpus). This is only supported when `prefix_projection=False` (the default), because
in that case the learned parameters are the KV prefix itself.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PrefixTuningConfig, get_peft_model, initialize_kv_prefix_from_text

base = AutoModelForCausalLM.from_pretrained("gpt2")
tok = AutoTokenizer.from_pretrained("gpt2")

peft_cfg = PrefixTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=20, prefix_projection=False)
model = get_peft_model(base, peft_cfg)

initialize_kv_prefix_from_text(
    model,
    tok,
    text="...a long context with at least num_virtual_tokens tokens...",
    use_chat_template=False,
)
```

Make sure the text is long enough to produce at least `num_virtual_tokens` tokens, otherwise initialization will fail.

The abstract from the paper is:

*Fine-tuning is the de facto way to leverage large pretrained language models to perform downstream tasks. However, it modifies all the language model parameters and therefore necessitates storing a full copy for each task. In this paper, we propose prefix-tuning, a lightweight alternative to fine-tuning for natural language generation tasks, which keeps language model parameters frozen, but optimizes a small continuous task-specific vector (called the prefix). Prefix-tuning draws inspiration from prompting, allowing subsequent tokens to attend to this prefix as if it were "virtual tokens". We apply prefix-tuning to GPT-2 for table-to-text generation and to BART for summarization. We find that by learning only 0.1\% of the parameters, prefix-tuning obtains comparable performance in the full data setting, outperforms fine-tuning in low-data settings, and extrapolates better to examples with topics unseen during training*.

## PrefixTuningConfig

[[autodoc]] tuners.prefix_tuning.config.PrefixTuningConfig

## PrefixEncoder

[[autodoc]] tuners.prefix_tuning.model.PrefixEncoder
