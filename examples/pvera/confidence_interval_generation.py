import tempfile

import numpy as np
import scipy
import torch
from datasets import load_dataset
from torch.nn import LazyLinear, Sequential, Softmax
from torchvision.transforms import Compose, Normalize, Resize
from tqdm import tqdm
from transformers import AutoModel

from peft import PeftModel, PveraConfig, get_peft_model


# load the dataset
device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
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
    imgs, lbls = transform(batch["image"].float()), batch["labels"]
    pred = linear_probe(model(imgs.to(device)).pooler_output)
    loss = criterion(pred, lbls.to(device))
    loss.backward()
    optimizer.step()

# save the model and load it with sample_at_inference=True
model.eval()
linear_probe.eval()
with tempfile.TemporaryDirectory() as tmpdir:
    # save the model and the linear probe
    model.save_pretrained(tmpdir)
    torch.save(linear_probe.state_dict(), tmpdir + "/linear_probe.bin")

    # load the model with sample_at_inference=True
    base_model = AutoModel.from_pretrained("facebook/dinov2-base")
    config = PveraConfig.from_pretrained(tmpdir)
    config.sample_at_inference = True
    loaded_model = PeftModel.from_pretrained(base_model, tmpdir, config=config).to(device)
    loaded_model.eval()

    # load the linear probe
    loaded_linear_probe = Sequential(LazyLinear(num_classes), Softmax(-1)).to(device)
    loaded_linear_probe.load_state_dict(torch.load(tmpdir + "/linear_probe.bin"))
    loaded_linear_probe.eval()

# make multiple predictions on an image
img = dataset[0]["image"].unsqueeze(0).to(device)
with torch.no_grad():
    all_preds = [loaded_linear_probe(loaded_model(img).pooler_output) for _ in range(16)]
all_preds = torch.vstack(all_preds)
top_pred = all_preds.argmax(-1).mode(0).values
softmax_top_pred = all_preds[:, top_pred]


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return max(0, m - h), min(1, m + h)


print(mean_confidence_interval(softmax_top_pred.cpu()))
