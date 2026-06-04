# PEFT method comparison on a DreamBooth-style image generation task

## Goal

This benchmark mirrors the structure of [`method_comparison/MetaMathQA`](https://github.com/huggingface/peft/tree/main/method_comparison/MetaMathQA) but targets DreamBooth-style fine-tuning for image generation. It is designed to compare PEFT methods along multiple dimensions like:

- objective quality ([`DINOv2`](https://huggingface.co/facebook/dinov2-base) cosine similarity)
- runtime
- memory usage
- checkpoint size

Note that for max memory reserved, this benchmark measures the memory only for the training part, not the evaluation. This is because evaluation requires extra memory (for running the DINO model) which should not be attributed to the corresponding PEFT method.

## Setup choices

- Base model: [`black-forest-labs/FLUX.2-klein-base-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)
- Dataset (default): [`cat pillow`](https://huggingface.co/datasets/peft-internal-testing/cat-image-dataset)

## Running

### Experiment settings

Create an experiment under `experiments/<peft-method>/<experiment-name>/` or use one of the experiments there.

Each experiment directory may contain:

- `adapter_config.json` (optional; if missing, full fine-tuning is used)
- `training_params.json` (optional; overrides `default_training_params.json`)

### Running a single experiment

Run one experiment:

```sh
python run.py -v experiments/lora/flux2-klein-rank16/
```

By default, the adapter will be saved in a temporary file for further inspection if needed. To prevent this, add the `--clean` flag to the call. To upload the model checkpoint and sample images to a Hugging Face Hub Bucket, pass the `--bucket_name your_user/my_bucket_name` argument.

### Running all pending experiments

The Makefile checks which experiments are missing a corresponding results file and runs those experiments. Note that running a whole sweep can easily take many hours.

```sh
make
```

If you set `UPLOAD_BUCKET="your_user/bucket_name"` as an environment variable prior to starting experiments via `make`, all experiments will be called with the `--bucket_name $UPLOAD_BUCKET` parameter and therefore store the checkpoints and sample images in that bucket.

List experiments to run:

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

### `training_params.json`

There is a default file for the non-PEFT parameters: `default_training_params.json`. This contains all the other parameters that are relevant for training, e.g. the base model id, number of steps, batch size, learning rate, etc. If parameters that differ from the defaults are needed for a specific experiment, place a `training_params.json` into the experiment directory and adjust the parameters that need changing. The other parameters are taken from the aforementioned default config.

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
