# PEFT method comparison on the MetaMathQA and GSM8K datasets

## Goal

This goal is to provide a benchmarking framework for the different PEFT methods that are implemented. It is important that evaluating different PEFT methods is reproducible, idempotent, and version-controlled. Results for more PEFT methods can be added over time.

## Dataset

This task trains on the [MetaMathQA]((https://huggingface.co/datasets/meta-math/MetaMathQA)) dataset and validates/tests on the [GSM8K](https://huggingface.co/datasets/openai/gsm8k) dataset ("main").

For the model to attain good accuracy, it needs to learn to adhere to the output format and it must express basic chain of thought reasoning capabilities to get to the correct result in the first place. The task is challenging for models in the sub 7B parameter range.

The train set uses the whole of MetaMathQA. The validation set is a random sample from the train set of GSM8K. The test set is the whole of the GSM8K test set.

## Running

Create an experiment in the `experiment/<peft-method>` folder of your choice and give it a name (the name itself does not matter but helps identify the experiment). An example would be `experiments/lora/llama-3.2-3B-rank32/`. Inside that directory, create 2 files:

- `adapter_config.json`
- Optional: `training_parameters.json`

Once you created these two files, you can either

- run the whole suite using by simply calling `make` (takes >24h)
- run one specific experiment by calling `make results/<experiment_name>-<experiment_variation>.json`,
  for example `results/vblora-llama-3.2-3B-default.json`

You can get a list of all runnable experiments by running `make list`, e.g.:
```
% make list                                                                                                                                                              (git)-[method-comparison-results]  ⛓ peft
Discovered experiment configurations:
  - experiments/ptuning/llama-3.2-3B-default/adapter_config.json
  [...]
  - experiments/vblora/llama-3.2-3B-default/adapter_config.json

Target result files:
  - results/ptuning-llama-3.2-3B-default.json
  [...]
  - results/vblora-llama-3.2-3B-default.json
```

In case you want to force the execution of an experiment, you can simply `touch` the respective adapter config
without modifying it. For example:

    touch experiments/vblora/llama-3.2-3B-default/adapter_config.json
    make

to run the VBLoRA default experiment again.

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

#### About `torch.compile`

Right now, compilation is a simple on/off switch in `training_params.json`. There is probably room for optimization here.

Due to the model being switched to `eval` mode for the validation metrics and then back to `train` mode, we will incur a re-compilation. This is acceptable to ensure that validation runs correctly. However, this prevents us from using `torch._dynamo.config.patch(error_on_recompile=True, inline_inbuilt_nn_modules=False)` to detect frequent recompilations. It should be noticeable from the duration of training steps, though, so we're fine with that.

### Runtime performance

Several factors should be considered to achieve a fast runtime performance. Besides the obvious factors like `max_steps` or the base model size, we found the following factors to have a significant impact:

#### Eval batch size

Regarding the `batch_size_eval` parameter, it is quite critical since evaluation takes up a significant portion of the training time and batching helps with reducing that. It should be possible to choose a value that is multiple times higher than the batch size used for training (`batch_size`). You should also pay attention to the size of the validation set -- e.g. if it's 50, don't choose a `batch_size_eval` of 40, as that results in a large batch of 30 and a small batch of 10. 25 might be a better choice. Also, ensure via a quick train run that the batch size does not lead to out of memory errors -- getting this error at the very end on evaluating the test set would be quite a loss of time.

#### Generation length

During testing, we discovered that the validation time is greatly inflated by just a few very long generations. Those can inflate the validation time by a factor of 3 or more. At the same time, we discovered that these long generations do not help with accuracy -- in fact, if they exceed the maximum configured length, they're just cut off mid sentence and would thus produce an accuracy of 0 anyway.

To remedy this, we now set both `max_length` and `max_new_tokens` for the generation kwargs in the default training parameters. Normally, this is not possible when using transformers, as the latter argument overrides the former. However, we have added special logic inside of `get_generation_config` which takes both and chooses the smaller of the two. This way, we can get rid of these excessively long generations, thus considerably reducing eval times, while still guaranteeing a maximum total generation length to guard against OOM errors. Testing showed that this does not hamper test accuracy. It is therefore recommended not to change these settings.

#### Bucketing

The length of the sequences in the training data can vary a lot. Therefore, if samples are taken randomly from the training dataset, we will end up with batches containing very short and very long sequences. This is bad because the batch will be padded to the longest sequence, slowing down training. The obvious solution would be to sort the whole dataset by sequence length, but this is also bad because it introduces an order bias (e.g. first training on only short and then on only long answers).

The solution is to find a trade off between the two factors. This is achieved by the `BucketIterator`. It first creates buckets that contain multiple batches, e.g. 20x the batch size. The bucket is then sorted by sequence length and then batches are yielded from the bucket. Therefore, we have a small order bias within a bucket but not between buckets, stricking a good balance between training speed and training loss.

From practical experiments, for a batch size of 4, a bucket size of 80 provides a good balance with only slightly lower training loss but cutting training time by 25%. For eval, we don't use the iterator since there, the batch size is relatively big and thus there is little upside.

### Start a run

Once everything is set up properly, start a run by using the `run.py` script. Pass `-v` for verbose output to the console (recommended if observing the progress is desired). As an example, for `experiments/lora/llama-3.2-3B-rank32/` the invocation would be:

```sh
python run.py -v experiments/lora/llama-3.2-3B-rank32/
```

By default, the adapter will be saved in a temporary file for further inspection if needed. The prevent this, add the `--clean` flag to the call.

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
    "created_at": "2025-03-05T13:50:05+00:00",
    "total_time": 2711.0915009640157,
    "experiment_name": "ia3/lr_0.001",
    "peft_branch": "ben-method-comparison",
    "train_config": {
      "model_id": "meta-llama/Llama-3.2-3B",
      "dtype": "bfloat16",
      "max_seq_length": 768,
      "batch_size": 4,
      "batch_size_eval": 51,
      "max_steps": 5000,
      "eval_steps": 250,
      "compile": false,
      "query_template": "Question: {query} Think step by step.\nAnswer:",
      "seed": 0,
      "grad_norm_clip": 1.0,
      "optimizer_kwargs": {
        "lr": 0.001
      },
      "lr_scheduler": "cosine",
      "use_amp": false,
      "generation_kwargs": {
        "max_length": 800
      },
      "attn_implementation": null
    },
    "peft_config": {
      "task_type": null,
      "peft_type": "IA3",
      "auto_mapping": null,
      "base_model_name_or_path": "meta-llama/Llama-3.2-3B",
      "revision": null,
      "inference_mode": false,
      "target_modules": [
        "v_proj",
        "k_proj",
        "down_proj"
      ],
      "exclude_modules": null,
      "feedforward_modules": [
        "down_proj"
      ],
      "fan_in_fan_out": false,
      "modules_to_save": null,
      "init_ia3_weights": true
    }
  },
  "train_info": {
    "accelerator_memory_reserved_avg": 14229219940,
    "accelerator_memory_max": 24847056896,
    "accelerator_memory_reserved_99th": 19115624366,
    "train_time": 2238.65277833899,
    "file_size": 1157064,
    "status": "success",
    "metrics": [
      {
        "step": 250,
        "valid accuracy": 0.0784313725490196,
        "train loss": 1.1336498007774354,
        "train samples": 1000
      },
      [...]
      {
        "step": 5000,
        "valid accuracy": 0.21568627450980393,
        "train loss": 0.6345920492410659,
        "train samples": 20000
      },
      {
        "step": 5000,
        "test accuracy": 0.35129740518962077,
        "train loss": 0.6345920492410659,
        "train samples": 20000,
        "train total tokens": 4197579
      }
    ]
  },
  "meta_info": {
    "model_sha": "13afe5124825b4f3751f836b40dafda64c1ed062",
    "model_created_at": "2024-09-18T15:23:48+00:00",
    "dataset_sha": "aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18",
    "dataset_created_at": "2023-09-21T17:22:46+00:00",
    "package_info": {
      "transformers-version": "4.50.0.dev0",
      "transformers-commit-hash": "752ef3fd4e70869626ec70657a770a85c0ad9219",
      "peft-version": "0.14.1.dev0",
      "peft-commit-hash": "a447a4e5ecd87b7d57733f4df9616a328cf130f4",
      "datasets-version": "3.3.2",
      "datasets-commit-hash": null,
      "bitsandbytes-version": "0.45.2",
      "bitsandbytes-commit-hash": null,
      "torch-version": "2.6.0+cu124",
      "torch-commit-hash": null
    },
    "system_info": {
      "system": "Linux",
      "release": "6.11.0-17-generic",
      "version": "#17~24.04.2-Ubuntu SMP PREEMPT_DYNAMIC Mon Jan 20 22:48:29 UTC 2",
      "machine": "x86_64",
      "processor": "x86_64",
      "accelerator": "NVIDIA GeForce RTX 4090"
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

Python 3.12+ is required.

## Open tasks

- consider using `DataLoader`
- consider adding https://github.com/huggingface/Math-Verify
- consider adding `weight` argument to cross entropy calculation to downweight the EOS token, but it would require calculating the loss manually instead of relying on transformers (see https://github.com/huggingface/transformers/blob/6a876462c308bd7cd7d3ca8e93abaa7d5b02e90e/src/transformers/loss/loss_utils.py#L24-L48)
- do a sanity check against/comparison with transformers Trainer
- consider using vLLM to potentially speed up generations, at least for the test set
- AMP does not appear to help, investigate
- packing of sequences (but this probably requires adjusting the attention matrix)
- clean up what gets printed and where (stdout, stderr)
