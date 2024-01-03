# DreamBooth fine-tuning with BOFT

## Set up your environment
Start by cloning the PEFT repository:

```python
git clone https://github.com/huggingface/peft
```
Set up your environment: install PEFT, and all the required libraries. At the time of writing this guide we recommend installing PEFT from source.

```python
conda create --name peft python=3.10
conda activate peft
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install xformers -c xformers
pip install -r requirements.txt
# pip install git+https://github.com/huggingface/peft
```

## Download the data
```
bash get_data.sh
```
This will download [dreambooth](https://github.com/google/dreambooth) dataset, the structure is:

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
You can also put your own data into `boft_dreambooth/data/dreambooth`

## Finetune Dreambooth with BOFT

Navigate to the directory containing the training scripts for fine-tuning Dreambooth with BOFT:

```python
cd peft/examples/boft_dreambooth
./train_dreambooth.sh
```