<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# PSOFT

[PSOFT](https://hf.co/papers/2505.11235) is an Orthogonal Fine-Tuning (OFT)-based parameter-efficient fine-tuning method that preserves the geometric relationships of pre-trained weight column vectors while achieving a balanced trade-off between performance and multi-dimensional efficiency, including parameter count, memory usage, and computational cost. By restricting orthogonal transformations to a low-rank principal subspace derived from pre-trained weights, PSOFT bridges the gap between LoRA and OFT, providing both theoretical guarantees and practical adaptability. Its effectiveness is validated through extensive evaluations on diverse benchmarks, including GLUE, VTAB-1K, GSM8K, MATH, and commonsense reasoning benchmarks.

- Only `nn.Linear` layers are supported.
- Quantized layers are not supported.

The abstract from the paper is:

*Driven by the rapid growth of model parameters, parameter-efficient fine-tuning (PEFT) has become essential for adapting large models to diverse downstream tasks under constrained computational resources. Within this paradigm, orthogonal fine-tuning and its variants preserve semantic representations of pre-trained models, but struggle to achieve both expressiveness and efficiency in terms of parameter counts, memory, and computation. To overcome this limitation, we propose efficient Orthogonal Fine-Tuning with Principal Subspace adaptation (PSOFT), which confines orthogonal transformations to the principal subspace of pre-trained weights. Specifically, PSOFT constructs this subspace via matrix decomposition to enable compatible transformations, establishes a theoretical condition that strictly maintains the geometry of this subspace for essential semantic preservation, and introduces efficient tunable vectors that gradually relax orthogonality during training to enhance adaptability. Extensive experiments on 35 NLP and CV tasks across four representative models demonstrate that PSOFT offers a practical and scalable solution to simultaneously achieve semantic preservation, expressiveness, and multi-dimensional efficiency in PEFT.*


## How PSOFT Works

PSOFT decomposes each weight matrix $W_{pre}$ into $W_{pri}$ and $W_{res}$ using SVD:
$W_{\text{pre}} = U S V^\top$

The principal subspace $W_{\text{pri}} = U_r S_r V_r^\top = AB$ is constructed from the top-$r$ singular components:

$W_{\text{pre}} = W_{\text{pri}} + W_{\text{res}} = AB + W_{\text{res}},$


$W_{\text{ps-tuned}} = ARB + W_{\text{res}}.$ (PSOFT-SO: PSOFT with strict orthogonality)


$W_{\text{ps-tuned}} = A \, \mathrm{diag}(\alpha) \, R \, \mathrm{diag}(\beta) \, B + W_{\text{res}}.$ (PSOFT-RO: PSOFT with relaxed orthogonality)

During training, $A$, $B$, and $W_{\text{res}}$ are frozen, and only $R$ (or $R$ with $\alpha$ and $\beta$) is trainable.

For compatibility with the PEFT framework (which expects additive weight updates), PSOFT is implemented in the following additive form:
$W_{\text{ps-tuned}} = W_{\text{pre}} + A (R - I_r) B$


## Trainable Parameters

After applying PSOFT:

- The original model weights ($A$, $B$, and $W_{\text{res}}$) are frozen.
- Only the orthogonal matrix $R$ (and optionally $\alpha$, $\beta$) are trainable.
- No additional bias parameters are introduced.

## Basic Usage
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PsoftConfig, get_peft_model

# Load base model
model_id = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(model_id)

# Configure PSOFT
config = PsoftConfig(
    r=32,                                   # the dimension of trainable matrix R, 
    psoft_alpha=32,                         # scaling factor (typically set to r in PSOFT),
    target_modules=["q_proj", "v_proj"],    # target attention projection layers
    ab_svd_init="psoft_init",        # principal subspace initialization
    psoft_svd="full",                       # SVD method
    psoft_orth=True,                        # enable orthogonal R (Cayley parameterization)
    psoft_mag_a=True,                       # enable tunable vector alpha
    psoft_mag_b=True,                       # enable tunable vector beta
    use_cayley_neumann=False,               # disable Cayley–Neumann approximation
    num_cayley_neumann_terms=5,             # number of Neumann series terms
    cayley_neumann_eps=None,                # improve numerical stability
)

# Apply PSOFT
model = get_peft_model(model, config)
model.train()

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Train
inputs = tokenizer("Hello world", return_tensors="pt", padding=True)
loss = model(**inputs, labels=inputs["input_ids"]).loss
loss.backward()

trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=5e-4)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```
## Configuration Options

### Different Mode

(PSOFT-SO: PSOFT with strict orthogonality)

```python
config = PsoftConfig(psoft_orth=True,psoft_mag_a=False,psoft_mag_b=False)
```

(PSOFT-RO: PSOFT with relaxed orthogonality)
```python
config = PsoftConfig(psoft_orth=True,psoft_mag_a=True,psoft_mag_b=True)
```

### Best Practices
1. **Rank Choice**: Smaller ranks (e.g., `32–128`) are suitable for simpler tasks, while larger ranks (e.g., `64–256`) provide greater expressiveness for more complex tasks at the cost of increased parameters and computation.
2. **Scaling Factor**: The scaling factor is typically set to $r$ in PSOFT.
3. **Learning Rate**: Use standard learning rates (e.g., `1e-4` to `5e-3`) for stable training.
4. **SVD Initialization**: The `lowrank` option is more memory- and compute-efficient than `full`, making it more suitable for large models.
5. **Cayley–Neumann Approximation**: When the rank is large, enabling the Cayley–Neumann approximation can significantly improve computational efficiency, while the benefit is less pronounced for small ranks. In practice, a small number of Neumann series terms (typically `5`) usually provides a good balance between accuracy and efficiency.


## PsoftConfig

[[autodoc]] tuners.psoft.config.PsoftConfig

## PsoftModel

[[autodoc]] tuners.psoft.model.PsoftModel