import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("cuda_version:", torch.version.cuda)

try:
    import bitsandbytes as bnb
    print("bnb:", bnb.__version__)
except ImportError:
    print("bnb: not installed")

import transformers
print("transformers:", transformers.__version__)

try:
    import peft
    print("peft:", peft.__version__)
except ImportError:
    print("peft: not installed")

try:
    import accelerate
    print("accelerate:", accelerate.__version__)
except ImportError:
    print("accelerate: not installed")