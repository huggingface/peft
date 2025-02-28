# PEFT method comparison on the MetaMathQA dataset

## Goal

This goal is to provide a benchmarking framework for the different PEFT methods that are implemented. It is important that evaluating different PEFT methods is reproducible, idempotent, and version-controlled. Results for more PEFT methods can be added over time.

## Dataset

More details about the `meta-math/MetaMathQA` dataset can be found [here](https://huggingface.co/datasets/meta-math/MetaMathQA).

For the model to attain good accuracy, it needs to learn to adhere to the output format and it must express basic chain of thought reasoning capabilities to get to the correct result in the first place. The dataset is challenging for models in the sub 7B parameter range.

## Running

Create an experiment in the `experiment/<peft-method>` folder of your choice and give it a name (the name itself does not matter but helps identify the experiment). An example would be `experiments/lora/llama-3.2-3B-rank32/`. Inside that directory, create 2 files:

- `adapter_config.json`
- `training_parameters.json`

### `adapter_config.json`

This must be a valid PEFT configuration. It is easiest to create it programmatically, e.g.:

```python
from peft import LoraConfig

config = LoraConfig(...)
config.save_pretrained(<path-to-experiment>)
```

### `training_parameters.json`

This contains all the other parameters that are relevant for training, e.g. the base model id, number of steps, batch size, learning rate, etc. It is easiest to copy an existing config and adapt it for your use. For an overview of all possible arguments, you can also check the `TrainConfig` `dataclass` in `utils.py`.

### Start a run

Once everything is set up properly, start a run by using the `run.py` script. Pass `-v` for verbose output to the console (recommended if observing the progress is desired). As an example, for `experiments/lora/llama-3.2-3B-rank32/` the invocation would be:

```sh
python run.py -v experiments/lora/llama-3.2-3B-rank32/
```

### Run status

The run can be categorized 3 different states:

1. Main run: You are on the `main` branch and the run ended successfully. The results are stored in the `results` folder and are used for further analysis.
2. Test run: You are not on the `main` branch and the run ended successfully. The results are stored in the `temporary_results` folder and are not used for further analysis.
3. The run was cancelled (`ctrl + c`). The results are stored in the `cancelled_results` folder and are not used for further analysis.

## Outputs

Results are stored in one of the result directories. An example output could look like so:

```js
{
  "run_info": {
    "created_at": "2025-02-28T14:06:32+00:00",
    "total_time": 268.78478554499816,
    "experiment_name": "lora/llama-3.2-3B-rank32",
    "peft_branch": "ben-method-comparison",
    "train_params_sha": "a02ef88ef774afcceb7796f451a2831ea0aa1fac7530ecb0b319f88d2e24f1f1",
    "peft_config_sha": "3b1a995103b79579ce90b7d46b2138bfb070d60281cdfbdc91d6cbddaaf7e939"
  },
  "train_info": {
    "cuda_memory_avg": 7053282512,
    "cuda_memory_max": 19338018816,
    "cuda_memory_99th": 7411898951,
    "train_time": 226.14555068000482,
    "file_size": 36715216,
    "status": "success",
    "metrics": [
      {
        "step": 100,
        "valid accuracy": 0.09523809523809523,
        "train loss": 0.8747109097242355,
        "train samples": 400
      },
      {
        "step": 123,
        "test accuracy": 0.21,
        "train loss": 0.7670372170209885,
        "train samples": 492
      }
    ]
  },
  "meta_info": {
    "model_sha": "13afe5124825b4f3751f836b40dafda64c1ed062",
    "model_created_at": "2024-09-18T15:23:48+00:00",
    "dataset_sha": "aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18",
    "dataset_created_at": "2023-09-21T17:22:46+00:00",
    "package_info": null,
    "system_info": {
      "system": "Linux",
      "release": "6.11.0-17-generic",
      "version": "#17~24.04.2-Ubuntu SMP PREEMPT_DYNAMIC Mon Jan 20 22:48:29 UTC 2",
      "machine": "x86_64",
      "processor": "x86_64",
      "gpu": "NVIDIA GeForce RTX 4090"
    },
    "pytorch_info": "PyTorch built with: [...]"
  }
}
```

## Dependencies

Apart from the normal PEFT dependencies, ensure that the packages in the `requirements.txt` are installed, e.g. via:

```sh
python -m pip install -r requirements.txt
```

Python 3.11+ is recommended

## Open tasks

- double check calculation of durations, also log eval time and test time
- consider using `DataLoader`
- consider adding https://github.com/huggingface/Math-Verify
- consider adding `weight` argument to cross entropy calculation to downweight the EOS token, but it would require calculating the loss manually instead of relying on transformers (see https://github.com/huggingface/transformers/blob/6a876462c308bd7cd7d3ca8e93abaa7d5b02e90e/src/transformers/loss/loss_utils.py#L24-L48)
- do a sanity check against/comparison with transformers Trainer
