# Orthogonal Subspace Learning with adaptive SVD

This example shows how to wrap a pretrained model with SVD-decomposed weights to enable orthogonal subspace training.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import wrap_model_with_svd, optim_wrapper

model = AutoModelForCausalLM.from_pretrained("gpt2")
model = wrap_model_with_svd(model)  # add trainable low-rank parameters

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
optimizer = optim_wrapper(optimizer, model)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
input_ids = tokenizer("Hello world", return_tensors="pt").input_ids
loss = model(input_ids, labels=input_ids).loss
loss.backward()
optimizer.step()
optimizer.zero_grad()
```
