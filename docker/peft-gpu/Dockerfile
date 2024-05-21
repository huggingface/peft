# Builds GPU docker image of PyTorch
# Uses multi-staged approach to reduce size
# Stage 1
# Use base conda image to reduce time
FROM continuumio/miniconda3:latest AS compile-image
# Specify py version
ENV PYTHON_VERSION=3.8
# Install apt libs - copied from https://github.com/huggingface/accelerate/blob/main/docker/accelerate-gpu/Dockerfile
RUN apt-get update && \
    apt-get install -y curl git wget software-properties-common git-lfs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*

# Install audio-related libraries 
RUN apt-get update && \
    apt install -y ffmpeg

RUN apt install -y libsndfile1-dev
RUN git lfs install

# Create our conda env - copied from https://github.com/huggingface/accelerate/blob/main/docker/accelerate-gpu/Dockerfile
RUN conda create --name peft python=${PYTHON_VERSION} ipython jupyter pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Below is copied from https://github.com/huggingface/accelerate/blob/main/docker/accelerate-gpu/Dockerfile
# We don't install pytorch here yet since CUDA isn't available
# instead we use the direct torch wheel
ENV PATH /opt/conda/envs/peft/bin:$PATH
# Activate our bash shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# Stage 2
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS build-image
COPY --from=compile-image /opt/conda /opt/conda
ENV PATH /opt/conda/bin:$PATH

RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]
RUN source activate peft && \ 
    python3 -m pip install --no-cache-dir bitsandbytes optimum auto-gptq

# Add autoawq for quantization testing
RUN source activate peft && \
    python3 -m pip install --no-cache-dir https://github.com/casper-hansen/AutoAWQ/releases/download/v0.2.4/autoawq-0.2.4-cp38-cp38-linux_x86_64.whl
RUN source activate peft && \
    python3 -m pip install --no-cache-dir https://github.com/casper-hansen/AutoAWQ_kernels/releases/download/v0.0.6/autoawq_kernels-0.0.6-cp38-cp38-linux_x86_64.whl

# Install apt libs
RUN apt-get update && \
    apt-get install -y curl git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*

# Add eetq for quantization testing
RUN source activate peft && \
    python3 -m pip install git+https://github.com/NetEase-FuXi/EETQ.git

# Activate the conda env and install transformers + accelerate from source
RUN source activate peft && \
    python3 -m pip install -U --no-cache-dir \
    librosa \
    "soundfile>=0.12.1" \
    scipy \
    git+https://github.com/huggingface/transformers \
    git+https://github.com/huggingface/accelerate \
    peft[test]@git+https://github.com/huggingface/peft

# Add aqlm for quantization testing
RUN source activate peft && \
    pip install aqlm[gpu]>=1.0.2

# Add HQQ for quantization testing
RUN source activate peft && \
pip install hqq

RUN source activate peft && \ 
    pip freeze | grep transformers

RUN echo "source activate peft" >> ~/.profile

# Activate the virtualenv
CMD ["/bin/bash"]
