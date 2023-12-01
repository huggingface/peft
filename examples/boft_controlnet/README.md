# Controllable generation fine-tuning with BOFT

## Set up your environment
Start by cloning the PEFT repository:

```python
git clone https://github.com/huggingface/peft
```
Set up your environment: install PEFT, and all the required libraries. At the time of writing this guide we recommend installing PEFT from source.

```python
conda create --name peft python=3.10
conda activate peft
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers accelerate evaluate datasets wandb diffusers==0.17.1
conda install xformers -c xformers
# pip install git+https://github.com/huggingface/peft
```

Download the example images for validation:
```python
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png
```

# Train controllable generation (ControlNet) with BOFT

Navigate to the directory containing the training scripts for fine-tuning Dreambooth with BOFT:

```python
cd peft/examples/boft_controlnet
./train_controlnet.sh
```