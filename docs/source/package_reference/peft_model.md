<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Models

[`PeftModel`] is the base model class for specifying the base Transformer model and configuration to apply a PEFT method to. The base `PeftModel` contains methods for loading and saving models from the Hub.

## PeftModel

### Common training interface

Call [`PeftModel.get_peft_loss`] after forward and add its result to your loss before backward. Call
[`PeftModel.on_peft_optimizer_step`] after each successful optimizer update, before clearing gradients. These methods
return `0.0` or do nothing when the PEFT method does not need them, when adapters are disabled, or in evaluation mode.
The additional loss excludes any loss already added by the model's forward method.

Both methods accept `**kwargs` for additional training context and forward these arguments unchanged to the tuner.
Implementations may ignore arguments they do not use. Overrides should retain `**kwargs` so training integrations
can supply additional context without requiring changes to every tuner. Configure algorithm settings through the
PEFT config.

```python
for global_step, batch in enumerate(dataloader):
    outputs = peft_model(**batch)
    loss = outputs.loss + peft_model.get_peft_loss()
    loss.backward()
    optimizer.step()
    peft_model.on_peft_optimizer_step(global_step)
    optimizer.zero_grad()
```

`global_step` follows the training loop's optimizer-step numbering and must use the same convention as the adapter's
schedule. With gradient accumulation, invoke the update hook once per optimizer update, rather than per microbatch.
Normalize the additional loss consistently with the task loss. For example, with equally weighted microbatches and
`K` accumulation steps, backpropagate `(task_loss + peft_model.get_peft_loss()) / K` on each microbatch.

With AMP, scale the combined task and PEFT loss once. The update hook requires unscaled gradients and runs after
any gradient clipping. It does not unscale gradients itself. Skip the hook if GradScaler skips the optimizer update
because of nonfinite gradients. For a single optimizer with the default GradScaler scale-update policy:

```python
loss = task_loss + peft_model.get_peft_loss()
scaler.scale(loss).backward()
scale_before = scaler.get_scale()
scaler.step(optimizer)  # unscales gradients before the optimizer update
scaler.update()
if scaler.get_scale() >= scale_before:
    peft_model.on_peft_optimizer_step(global_step)
optimizer.zero_grad()
```

AMP-aware custom optimizers must also expose unscaled gradients at this point. Distributed parameter access and
loss computation depend on the individual PEFT method's support for the distributed training backend.

For Transformers `Trainer` and its subclasses, use [`helpers.PeftTrainerMixin`] to handle both calls automatically.

[[autodoc]] PeftModel
    - all

## PeftModelForSequenceClassification

A `PeftModel` for sequence classification tasks.

[[autodoc]] PeftModelForSequenceClassification
    - all

## PeftModelForTokenClassification

A `PeftModel` for token classification tasks.

[[autodoc]] PeftModelForTokenClassification
    - all

## PeftModelForCausalLM

A `PeftModel` for causal language modeling.

[[autodoc]] PeftModelForCausalLM
    - all

## PeftModelForSeq2SeqLM

A `PeftModel` for sequence-to-sequence language modeling.

Note: this will automatically generate decoder input tokens by shifting the input tokens one step right using
[`shift_tokens_right`] if you supply input tokens but don't supply decoder input tokens explicitly.

[[autodoc]] PeftModelForSeq2SeqLM
    - all

## PeftModelForQuestionAnswering

A `PeftModel` for question answering.

[[autodoc]] PeftModelForQuestionAnswering
    - all

## PeftModelForFeatureExtraction

A `PeftModel` for getting extracting features/embeddings from transformer models.

[[autodoc]] PeftModelForFeatureExtraction
    - all

## PeftMixedModel

A `PeftModel` for mixing different adapter types (e.g. LoRA and LoHa).

[[autodoc]] PeftMixedModel
    - all

## Utilities

[[autodoc]] utils.cast_mixed_precision_params

[[autodoc]] get_peft_model

[[autodoc]] inject_adapter_in_model

[[autodoc]] utils.get_peft_model_state_dict

[[autodoc]] utils.prepare_model_for_kbit_training

[[autodoc]] get_layer_status

[[autodoc]] get_model_status
