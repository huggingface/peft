<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# PVeRA: Probabilistic Vector-Based Random Matrix Adaptation

[PVeRA](https://huggingface.co/papers/2512.07703) is a parameter-efficient fine-tuning technique that is base on VeRA, in the family of the LoRA-based adapters. It keeps the very low parameter budget of VeRA, but increases the performance by learning a distribution of latent adaptations. This also enables models adapted with PVeRA to generate Monte Carlo confidence interval estimates, by sampling from the learned distribution at inference.

When saving the adapter parameters, it's possible to eschew storing the low rank matrices by setting `save_projection=False` on the `PveraConfig`. In that case, these matrices will be restored based on the fixed random seed from the `projection_prng_key` argument. This cuts down on the size of the checkpoint, but we cannot guarantee reproducibility on all devices and for all future versions of PyTorch. If you want to ensure reproducibility, set `save_projection=True` (which is the default).

To handle different shapes of adapted layers, PVeRA initializes shared A and B matrices with the largest required size for each dimension. During the forward pass, submatrices A and B for a given layer are sliced out from these shared matrices and used as described in the paper. For example, adapting two linear layers of shapes (100, 20) and (80, 50) will create A and B matrices of shapes (rank, 50) and (100, rank) respectively. Then, to adapt a layer of shape (100, 20), submatrices A and B of shapes (rank, 20) and (100, rank) will be extracted.

PVeRA currently has the following constraint:

- Only `nn.Linear` layers are supported.
- The latent representation is not easily accessible, for training using the KL divergence.

The abstract from the paper is:

> Large foundation models have emerged in the last years and are pushing performance boundaries for a variety of tasks. Training or even finetuning such models demands vast datasets and computational resources, which are often scarce and costly. Adaptation methods provide a computationally efficient solution to address these limitations by allowing such models to be finetuned on small amounts of data and computing power. This is achieved by appending new trainable modules to frozen backbones with only a fraction of the trainable parameters and fitting only these modules on novel tasks. Recently, the VeRA adapter was shown to excel in parameter-efficient adaptations by utilizing a pair of frozen random low-rank matrices shared across all layers. In this paper, we propose PVeRA, a probabilistic version of the VeRA adapter, which modifies the low-rank matrices of VeRA in a probabilistic manner. This modification naturally allows handling inherent ambiguities in the input and allows for different sampling configurations during training and testing. A comprehensive evaluation was performed on the VTAB-1k benchmark and seven adapters, with PVeRA outperforming VeRA and other adapters.

## PveraConfig

[[autodoc]] tuners.pvera.config.PveraConfig

## PveraModel

[[autodoc]] tuners.pvera.model.PveraModel

## Confidence interval generation

PVeRA can be used at inference to generate Monte Carlo confidence interval estimations. Here is an example of how to generate such confidence intervals.
```python
from datasets import load_dataset
import numpy as np
from peft import PveraConfig, get_peft_model
import scipy
import torch
from torch.nn import LazyLinear, Softmax, Sequential
from torchvision.transforms import Normalize, Resize, Compose
from transformers import AutoModel
from tqdm import tqdm

# load the dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = load_dataset("beans", split="train").with_format("torch")
transform = Compose((Resize((224, 224)), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))))
num_classes = dataset.features["labels"].num_classes

# load the model with adapters and create the linear probe
base_model = AutoModel.from_pretrained("facebook/dinov2-base")
config = PveraConfig(r=128, sample_at_inference=False)
model = get_peft_model(base_model, config).to(device)
linear_probe = Sequential(LazyLinear(num_classes), Softmax(-1)).to(device)

# train the model
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(model.parameters()) + list(linear_probe.parameters()), lr=1e-4)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
for batch in tqdm(dataloader):
    imgs, lbls = transform(batch['image'].float()), batch['labels']
    pred = linear_probe(model(imgs.to(device)).pooler_output)
    loss = criterion(pred, lbls.to(device))
    loss.backward()
    optimizer.step()

# switch to sample_at_inference=True
model.eval()
linear_probe.eval()
for layer in model.base_model.model.encoder.layer:
    layer.attention.attention.query.sample_at_inference = True
    layer.attention.attention.value.sample_at_inference = True

# make multiple predictions on an image
img = dataset[0]["image"].unsqueeze(0).to(device)
with torch.no_grad():
    all_preds = [linear_probe(model(img).pooler_output) for _ in range(16)]
all_preds = torch.vstack(all_preds)
top_pred = all_preds.argmax(-1).mode(0).values
softmax_top_pred = all_preds[:, top_pred]

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return max(0, m-h), min(1, m+h)

print(mean_confidence_interval(softmax_top_pred.cpu()))
```