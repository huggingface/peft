import copy
import os

import torch
import peft

from simple_model import MLP 
from mock_dataset import train_dataloader, eval_dataloader

device = 'cpu' if not torch.cuda.is_available() else 'cuda:0'

config = peft.LoraConfig(
    r=8,
    target_modules=["seq.0", "seq.2"],
    modules_to_save=["seq.4"],
)


module = MLP().to(device)
module_copy = copy.deepcopy(module)  # we keep a copy of the original model for later
peft_model = peft.get_peft_model(module, config)
peft_model.print_trainable_parameters()