TODO

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

## Dependencies

Apart from the normal PEFT dependencies, ensure that the packages in the `requirements.txt` are installed, e.g. via:

```sh
python -m pip install -r requirements.txt
```


## Open tasks

- consider using `DataLoader`
- consider adding https://github.com/huggingface/Math-Verify
