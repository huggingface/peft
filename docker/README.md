# PEFT Docker images

Here we store all PEFT Docker images used in our testing infrastructure. We use python 3.11 for now on all our images.

- `peft-cpu`: PEFT compiled on CPU with all other HF libraries installed on main branch
- `peft-gpu`: PEFT complied for NVIDIA GPUs with all other HF libraries installed on main branch
- `peft-gpu-bnb-source`: PEFT complied for NVIDIA GPUs with `bitsandbytes` and all other HF libraries installed from main branch
- `peft-gpu-bnb-latest`: PEFT complied for NVIDIA GPUs with `bitsandbytes` complied from main and all other HF libraries installed from latest PyPi
