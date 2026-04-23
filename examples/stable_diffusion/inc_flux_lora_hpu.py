"""
This exampe demonstrates loading of LoRA adapter (via PEFT) into an FP8 INC-quantized FLUX model.

More info on Intel Neural Compressor (INC) FP8 quantization is available at:
https://github.com/intel/neural-compressor/tree/master/examples/helloworld/fp8_example

Requirements:
pip install optimum-habana sentencepiece neural-compressor[pt] peft
"""

import importlib

import torch
from neural_compressor.torch.quantization import FP8Config, convert, finalize_calibration, prepare


# Checks if HPU device is available
# Adapted from https://github.com/huggingface/accelerate/blob/b451956fd69a135efc283aadaa478f0d33fcbe6a/src/accelerate/utils/imports.py#L435
def is_hpu_available():
    if (
        importlib.util.find_spec("habana_frameworks") is None
        or importlib.util.find_spec("habana_frameworks.torch") is None
    ):
        return False

    import habana_frameworks.torch  # noqa: F401

    return hasattr(torch, "hpu") and torch.hpu.is_available()


# Ensure HPU device is available before proceeding
if is_hpu_available():
    from optimum.habana.diffusers import GaudiFluxPipeline
else:
    raise RuntimeError("HPU device not found. This code requires Intel Gaudi device to run.")

# Example: FLUX model inference on HPU via optimum-habana pipeline
hpu_configs = {
    "use_habana": True,
    "use_hpu_graphs": True,
    "sdp_on_bf16": True,
    "gaudi_config": "Habana/stable-diffusion",
}
pipe = GaudiFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, **hpu_configs)
prompt = "A picture of sks dog in a bucket"

# Quantize FLUX transformer to FP8 using INC (Intel Neural Compressor)
quant_configs = {
    "mode": "AUTO",
    "observer": "maxabs",
    "scale_method": "maxabs_hw",
    "allowlist": {"types": [], "names": []},
    "blocklist": {"types": [], "names": []},
    "dump_stats_path": "/tmp/hqt_output/measure",
}
config = FP8Config(**quant_configs)
pipe.transformer = prepare(pipe.transformer, config)
pipe(prompt)
finalize_calibration(pipe.transformer)
pipe.transformer = convert(pipe.transformer)

# Load LoRA weights with PEFT
pipe.load_lora_weights("dsocek/lora-flux-dog", adapter_name="user_lora")

# Run inference
image = pipe(prompt).images[0]
image.save("dog.png")
