# Orthogonal Subspace Learning with Adaptive OSF

This example shows how to wrap a pretrained model with SVD-decomposed weights to enable orthogonal subspace training.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import OSFConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("gpt2")
model = get_peft_model(model, OSFConfig())  # add trainable low-rank parameters

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
input_ids = tokenizer("Hello world", return_tensors="pt").input_ids
loss = model(input_ids, labels=input_ids).loss
loss.backward()
optimizer.step()
optimizer.zero_grad()
```
