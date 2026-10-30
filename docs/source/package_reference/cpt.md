<!-- Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0


⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Context-aware Prompt Tuning: Advancing In-Context Learning with Adversarial Methods


<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/cpt.png"/>
</div>
<small>CPT optimizing only specific token embeddings while keeping the rest of the model frozen <a href="https://huggingface.co/papers/2410.17222">(image source)</a>.</small>

[Context-Aware Prompt Tuning (CPT)](https://huggingface.co/papers/2410.17222) is designed to enhance few-shot classification by refining only context embeddings.
This approach combines ideas from In-Context Learning (ICL), [Prompt Tuning](../package_reference/prompt_tuning) (PT), and adversarial optimization, focusing on making model adaptation both parameter-efficient and effective.
In CPT, only specific context token embeddings are optimized, while the rest of the model remains frozen.
To prevent overfitting and maintain stability, CPT uses controlled perturbations to limit the allowed changes to context embeddings within a defined range.
Additionally, to address the phenomenon of recency bias—where examples near the end of the context tend to be prioritized over earlier ones—CPT applies a decay loss factor.

The abstract from the paper is:

> Large Language Models (LLMs) can perform few-shot learning using either optimization-based approaches or In-Context Learning (ICL). Optimization-based methods often suffer from overfitting, as they require updating a large number of parameters with limited data. In contrast, ICL avoids overfitting but typically underperforms compared to optimization-based methods and is highly sensitive to the selection, order, and format of demonstration examples. To overcome these challenges, we introduce Context-aware Prompt Tuning (CPT), a method inspired by ICL, Prompt Tuning (PT), and adversarial attacks. CPT builds on the ICL strategy of concatenating examples before the input, extending it by incorporating PT-like learning to refine the context embedding through iterative optimization, extracting deeper insights from the training examples. Our approach carefully modifies specific context tokens, considering the unique structure of the examples within the context. In addition to updating the context with PT-like optimization, CPT draws inspiration from adversarial attacks, adjusting the input based on the labels present in the context while preserving the inherent value of the user-provided data. To ensure robustness and stability during optimization, we employ a projected gradient descent algorithm, constraining token embeddings to remain close to their original values and safeguarding the quality of the context. Our method has demonstrated superior accuracy across multiple classification tasks using various LLM models, outperforming existing baselines and effectively addressing the overfitting challenge in few-shot learning.

Take a look at [Example](https://github.com/huggingface/peft/blob/main/examples/cpt_finetuning/README.md) for a step-by-step guide on how to train a model with CPT.

## Working Principle

Context-aware Prompt Tuning is similar to [Prompt Tuning](./prompt_tuning) in that you can take a text prompt of your
choosing but make the tokens of this prompt, and only this prompt, trainable by virtue of having a second token
embedding over the prompt tokens only. Whenever you query a CPT model, your 'soft' prompt will be prepended to the
embedding of your normal input. CPT differs from Prompt Tuning in that your soft prompt can now be contextualized by
marking specific parts NOT to be trained. The CPT-specific token embedding matrix consists of a static embedding
initialized from the base model's embedding matrix and learnable token-specific delta values.

CPT is useful when you have a prompt with instructions alongside examples and ground-truth/labels. Instructions can be
suboptimal and therefore benefit from being optimized during training (like with prompt tuning) but the examples/labels
you included in the prompt are not supposed to change. This can be enforced by explicitly defining where prompting ends
and examples begin. CPT also adds a token loss to the standard next token prediction loss of your model for the
examples in the prompt to make optimal use of the given examples.

CPT uses five different type ids to mark the different sections in the prompt:

* 0: separator: will be ignored completely, useful for begin/end-of-sentence tokens
* 1: input marker: tokens describing the prompt that marks the whole input-side template, including additional info
* 2: input query: tokens describing the user input of an example
* 3: output marker: tokens describing the start of the expected output of an example
* 4: label: tokens belonging to the label of the example

This is very abstract, so here's an example of a CPT prompt and the corresponding type identifiers.
We assume that each word = 1 token for simplicity:

```
prompt_text = "Example 1: Input: 1+1 Answer: 2 Example 2: 1+3 Answer: 4"

token_ids = [15,  2,      11, 9, 1,    9, 11, 3, 11, 8,     9, 4, 2,       4, 9, 11, 3, 6, 8,     9,  10]
#            BOS, Example 1   :  Input :  1   +  1   Answer :  2  Example  2  :  1   +  3  Answer :   4
type_ids  = [0,   1,      1,  1, 1,    1, 2,  2, 2,  3,     3, 4, 5,       5, 5, 6,  6, 6, 7,     7,  8]
                                                                # 1,       1, 1, 2,  2, 2, 3,     3,  4
```

The `token_ids` are your prompt encoded by the base model's tokenizer and correspond to [`~CPTConfig.cpt_token_ids`].
`type_ids` are the prompt token type ids, defining the different sections of the prompt and correspond to
[`~CPTConfig.cpt_tokens_type_mask`].

The individual ids have different effects on the training process:

* 0: will be completely ignored, doesn't contribute to the loss and these tokens are not trained, useful for tokens
  that are just there for technical reasons and would otherwise uselessly enlarge the parameter space (e.g., EOS tokens)
* 1 and 3: these tokens are considered for formatting and the maximum norm of their embedding delta is controlled by
  [`~CPTConfig.opt_projection_format_epsilon`]. The norm controls the maximum magnitude in which the learned delta
  embeddings for this type can point. Higher norms mean a higher potential change in meaning.
* 2: these tokens are considered for the input of an example and the maximum norm of their embedding delta is controlled
  by [`~CPTConfig.opt_projection_epsilon`]. The norm controls the maximum magnitude in which the learned delta
  embeddings for this type can point. Higher norms mean a higher potential change in meaning.
* 4: The prompt's label tokens are added as extra next-token-prediction targets to the standard loss on your training
  data; every other soft-prompt token is masked out of the loss. The label token delta embeddings are never changed
  to keep the label semantics intact.

You may have noticed that the numbers in `type_ids` for the second example are shifted by 4. CPT has an exponential
weighting for the loss of the examples you provide in the prompt, the last example being the least decayed and the first
example's loss being the most decaying. This gives more weight to examples closer to the actual input query and is
controlled by [`~CPTConfig.opt_loss_decay_factor`] when [`~CPTConfig.opt_weighted_loss_type`] is set to `"decay"`. By
shifting the examples by 4 we can enumerate the different examples so that the loss can give the different decaying
weightings to each example.

Note that `cpt_tokens_type_mask` only describes the soft prompt itself. When training, additionally pass an
`input_type_mask` tensor alongside each batch, assigning type ids to the tokens of your actual input (using ids 0-4 as
above, with padding or end-of-sentence tokens marked as 0). CPT automatically shifts these ids by the number of
examples in your prompt, so you don't need to shift them yourself. This is then the same as adding the input at the end
of your list of examples in the prompt. Keep in mind that the ids are interpreted modulo 4: shifting them by a multiple
of 4 is harmless, but any other offset silently changes which tokens contribute to the loss.  If no `input_type_mask` is
passed, CPT falls back to computing the loss over the whole input sequence, i.e. assuming type 4 for all input tokens.


## Benchmark overview

There is no benchmark for this method yet. Feel free to contribute an experiment
configuration but make sure to first create an issue
[here](https://github.com/huggingface/peft/issues).

# API

## CPTConfig

[[autodoc]] tuners.cpt.config.CPTConfig

## CPTEmbedding

[[autodoc]] tuners.cpt.model.CPTEmbedding

