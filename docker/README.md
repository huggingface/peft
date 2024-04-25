# PEFT Docker images

Here we store all PEFT Docker images used in our testing infrastructure. We use python 3.8 for now on all our images.

- `peft-cpu`: PEFT compiled on CPU with all other HF libraries installed on main branch
- `peft-gpu`: PEFT complied for NVIDIA GPUs wih all other HF libraries installed on main branch
- `peft-gpu-bnb-source`: PEFT complied for NVIDIA GPUs with `bitsandbytes` and all other HF libraries installed from main branch
- `peft-gpu-bnb-latest`: PEFT complied for NVIDIA GPUs with `bitsandbytes` complied from main and all other HF libraries installed from latest PyPi
- `peft-gpu-bnb-multi-source`: PEFT complied for NVIDIA GPUs with `bitsandbytes` complied from `multi-backend` branch and all other HF libraries installed from main branch

`peft-gpu-bnb-source` and `peft-gpu-bnb-multi-source` are essentially the same, with the only difference being `bitsandbytes` compiled on another branch. Make sure to propagate the changes you applied on one file to the other!
