# Orthogonal Subspace Learning with Adaptive OSF

## TODO: Runnable Example Needed

This folder is a placeholder for a comprehensive OSF example. As suggested in the review feedback:

> "If you can, provide a runnable example in this folder instead, you can take a look at the EVA example for inspiration. A runnable example can be a good place to showcase the different features. Jupyter notebooks are fine as well."

### Planned Example Features:
- Complete continual learning scenario with multiple tasks
- Demonstration of OSF's catastrophic forgetting prevention
- Configuration examples (target_modules, effective_rank, rank_pattern)
- Performance comparison with baseline methods
- Memory usage analysis

### Current Basic Usage:
For basic usage examples and API documentation, see the [OSF documentation](../../docs/source/package_reference/osf.md).

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import OSFConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("gpt2")
config = OSFConfig(target_modules=["c_attn", "c_proj"], effective_rank=8)
model = get_peft_model(model, config)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer("Hello world", return_tensors="pt", padding=True)
loss = model(**inputs, labels=inputs.input_ids).loss
loss.backward()
optimizer.step()
optimizer.zero_grad()
```
