# PEFT method comparison on a DreamBooth-style image generation task

## Goal

This benchmark mirrors the structure of [`method_comparison/MetaMathQA`](https://github.com/huggingface/peft/tree/main/method_comparison/MetaMathQA) but targets DreamBooth-style fine-tuning for image generation. It is designed to compare PEFT methods along multiple dimensions like:

- objective quality ([`DINOv2`](https://huggingface.co/facebook/dinov2-base) cosine similarity)
- runtime
- memory usage
- checkpoint size

## Setup choices

- Base model: [`black-forest-labs/FLUX.2-klein-base-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)
- Dataset (default): [`cat pillow`](https://huggingface.co/datasets/peft-internal-testing/cat-image-dataset)

## Running

Create an experiment under `experiments/<peft-method>/<experiment-name>/` or use one of the experiments there.

Each experiment directory may contain:

- `adapter_config.json` (optional; if missing, full fine-tuning is used)
- `training_params.json` (optional; overrides `default_training_params.json`)

Run one experiment:

```sh
python run.py -v experiments/lora/flux2-klein-rank16/
```

Run all experiments:

```sh
make
```

List experiments:

```sh
make list
```

## Training configs

### `adapter_config.json`

This must be a valid PEFT configuration. It is easiest to create it programmatically, e.g.:

```python
from peft import LoraConfig

config = LoraConfig(...)
config.save_pretrained(<path-to-experiment>)
```

### `training_parameters.json`

There is a default file for the non-PEFT parameters: `default_training_params.json`. This contains all the other parameters that are relevant for training, e.g. the base model id, number of steps, batch size, learning rate, etc. If parameters that differ from the defaults are needed for a specific experiment, place a `training_parameters.json` into the experiment directory and adjust the parameters that need changing. The other parametes are taken from the aforementioned default config.

For an overview of all possible arguments, you can also check the `TrainConfig` `dataclass` in `utils.py`.

## Dependencies

Install additional dependencies from:

```sh
python -m pip install -r requirements.txt
```

Python 3.12+ is required.

## TODO

- Add further experiments (more PEFT methods) and explore better hyper-parameters.
- Test images are already created but they're not uploaded anywhere.
- The method comparison Gradio app needs to be updated to show the generated images.
