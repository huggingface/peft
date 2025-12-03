# Builds GPU docker image of PyTorch
# Uses multi-staged approach to reduce size
# Stage 1
# Use base conda image to reduce time
FROM continuumio/miniconda3:latest AS compile-image
# Specify py version
ENV PYTHON_VERSION=3.11
# Install apt libs - copied from https://github.com/huggingface/accelerate/blob/main/docker/accelerate-gpu/Dockerfile
# Install audio-related libraries
RUN apt-get update && \
    apt-get install -y curl git wget software-properties-common git-lfs ffmpeg libsndfile1-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*

RUN git lfs install

# Create our conda env - copied from https://github.com/huggingface/accelerate/blob/main/docker/accelerate-gpu/Dockerfile
RUN conda create --name peft python=${PYTHON_VERSION} ipython jupyter pip

# Below is copied from https://github.com/huggingface/accelerate/blob/main/docker/accelerate-gpu/Dockerfile
# We don't install pytorch here yet since CUDA isn't available
# instead we use the direct torch wheel
ENV PATH /opt/conda/envs/peft/bin:$PATH
# Activate our bash shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# Stage 2
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build-image
COPY --from=compile-image /opt/conda /opt/conda
ENV PATH /opt/conda/bin:$PATH

# Install apt libs
RUN apt-get update && \
    apt-get install -y curl git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*

RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

RUN conda run -n peft pip install --no-cache-dir bitsandbytes optimum auto-gptq

RUN \
    # Add autoawq for quantization testing
    conda run -n peft pip install --no-cache-dir https://github.com/casper-hansen/AutoAWQ/releases/download/v0.2.7.post2/autoawq-0.2.7.post2-py3-none-any.whl && \
    conda run -n peft pip install --no-cache-dir https://github.com/casper-hansen/AutoAWQ_kernels/releases/download/v0.0.9/autoawq_kernels-0.0.9-cp311-cp311-linux_x86_64.whl && \
    # Add eetq for quantization testing; needs to run without build isolation since the setup
    # script directly imports torch from the environment which would fail with isolation.
    conda run -n peft pip install --no-build-isolation git+https://github.com/NetEase-FuXi/EETQ.git

# Activate the conda env and install transformers + accelerate from source
RUN conda run -n peft pip install -U --no-cache-dir \
        librosa \
        "soundfile>=0.12.1" \
        scipy \
        torchao \
        fbgemm-gpu-genai>=1.2.0 \
        git+https://github.com/huggingface/transformers \
        git+https://github.com/huggingface/accelerate \
        peft[test]@git+https://github.com/huggingface/peft \
        # Add aqlm for quantization testing
        aqlm[gpu]>=1.0.2 \
        # Add HQQ for quantization testing
        hqq

RUN conda run -n peft pip freeze | grep transformers

RUN echo "source activate peft" >> ~/.profile

# Activate the virtualenv
CMD ["/bin/bash"]
