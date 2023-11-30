# DreamBooth fine-tuning with BOFT

## Set up your environment
Start by cloning the PEFT repository:

```python
git clone https://github.com/huggingface/peft
```
Navigate to the directory containing the training scripts for fine-tuning Dreambooth with BOFT:

```python
cd peft/examples/lora_dreambooth
```
Set up your environment: install PEFT, and all the required libraries. At the time of writing this guide we recommend installing PEFT from source.

```python
pip install -r requirements.txt
pip install git+https://github.com/huggingface/peft
```

## Download the data

As an example, we download the [dreambooth](https://github.com/google/dreambooth) dataset in the example folder:

```
boft_dreambooth
├── data
│   ├── data_dir
│   └── dreambooth
│       └── data
│           ├── backpack
│           └── backpack_dog
│           ...
```