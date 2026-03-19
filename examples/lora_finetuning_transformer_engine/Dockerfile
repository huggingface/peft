FROM nvcr.io/nvidia/pytorch:26.01-py3

WORKDIR /workspace/lora_finetuning

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY lora_finetuning_te.py .
