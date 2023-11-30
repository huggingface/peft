getenv=True
source /home/yxiu/miniconda3/bin/activate OPT

# . /home/zqiu/miniconda3/etc/profile.d/conda.sh
# conda activate oft

idx=$1
prompt_idx=$((idx % 25))
class_idx=$((idx % 30))

# Define the unique_token, class_tokens, and subject_names
unique_token="qwe"

subject_names=(
    "backpack" "backpack_dog" "bear_plushie" "berry_bowl" "can"
    "candle" "cat" "cat2" "clock" "colorful_sneaker"
    "dog" "dog2" "dog3" "dog5" "dog6"
    "dog7" "dog8" "duck_toy" "fancy_boot" "grey_sloth_plushie"
    "monster_toy" "pink_sunglasses" "poop_emoji" "rc_car" "red_cartoon"
    "robot_toy" "shiny_sneaker" "teapot" "vase" "wolf_plushie"
)

class_tokens=(
    "backpack" "backpack" "stuffed animal" "bowl" "can"
    "candle" "cat" "cat" "clock" "sneaker"
    "dog" "dog" "dog" "dog" "dog"
    "dog" "dog" "toy" "boot" "stuffed animal"
    "toy" "glasses" "toy" "toy" "cartoon"
    "toy" "sneaker" "teapot" "vase" "stuffed animal"
)

class_token=${class_tokens[$class_idx]}
selected_subject=${subject_names[$class_idx]}

if [[ $class_idx =~ ^(0|1|2|3|4|5|8|9|17|18|19|20|21|22|23|24|25|26|27|28|29)$ ]]; then
  prompt_list=(
    "a ${unique_token} ${class_token} in the jungle."
    "a ${unique_token} ${class_token} in the snow."
    "a ${unique_token} ${class_token} on the beach."
    "a ${unique_token} ${class_token} on a cobblestone street."
    "a ${unique_token} ${class_token} on top of pink fabric."
    "a ${unique_token} ${class_token} on top of a wooden floor."
    "a ${unique_token} ${class_token} with a city in the background."
    "a ${unique_token} ${class_token} with a mountain in the background."
    "a ${unique_token} ${class_token} with a blue house in the background."
    "a ${unique_token} ${class_token} on top of a purple rug in a forest."
    "a ${unique_token} ${class_token} with a wheat field in the background."
    "a ${unique_token} ${class_token} with a tree and autumn leaves in the background."
    "a ${unique_token} ${class_token} with the Eiffel Tower in the background."
    "a ${unique_token} ${class_token} floating on top of water."
    "a ${unique_token} ${class_token} floating in an ocean of milk."
    "a ${unique_token} ${class_token} on top of green grass with sunflowers around it."
    "a ${unique_token} ${class_token} on top of a mirror."
    "a ${unique_token} ${class_token} on top of the sidewalk in a crowded street."
    "a ${unique_token} ${class_token} on top of a dirt road."
    "a ${unique_token} ${class_token} on top of a white rug."
    "a red ${unique_token} ${class_token}."
    "a purple ${unique_token} ${class_token}."
    "a shiny ${unique_token} ${class_token}."
    "a wet ${unique_token} ${class_token}."
    "a cube shaped ${unique_token} ${class_token}"
  )

  prompt_test_list=(
    "a ${class_token} in the jungle"
    "a ${class_token} in the snow"
    "a ${class_token} on the beach"
    "a ${class_token} on a cobblestone street"
    "a ${class_token} on top of pink fabric"
    "a ${class_token} on top of a wooden floor"
    "a ${class_token} with a city in the background"
    "a ${class_token} with a mountain in the background"
    "a ${class_token} with a blue house in the background"
    "a ${class_token} on top of a purple rug in a forest"
    "a ${class_token} with a wheat field in the background"
    "a ${class_token} with a tree and autumn leaves in the background"
    "a ${class_token} with the Eiffel Tower in the background"
    "a ${class_token} floating on top of water"
    "a ${class_token} floating in an ocean of milk"
    "a ${class_token} on top of green grass with sunflowers around it"
    "a ${class_token} on top of a mirror"
    "a ${class_token} on top of the sidewalk in a crowded street"
    "a ${class_token} on top of a dirt road"
    "a ${class_token} on top of a white rug"
    "a red ${class_token}"
    "a purple ${class_token}"
    "a shiny ${class_token}"
    "a wet ${class_token}"
    "a cube shaped ${class_token}"
  )

else
  prompt_list=(
    "a ${unique_token} ${class_token} in the jungle."
    "a ${unique_token} ${class_token} in the snow."
    "a ${unique_token} ${class_token} on the beach."
    "a ${unique_token} ${class_token} on a cobblestone street."
    "a ${unique_token} ${class_token} on top of pink fabric."
    "a ${unique_token} ${class_token} on top of a wooden floor."
    "a ${unique_token} ${class_token} with a city in the background."
    "a ${unique_token} ${class_token} with a mountain in the background."
    "a ${unique_token} ${class_token} with a blue house in the background."
    "a ${unique_token} ${class_token} on top of a purple rug in a forest."
    "a ${unique_token} ${class_token} wearing a red hat."
    "a ${unique_token} ${class_token} wearing a santa hat."
    "a ${unique_token} ${class_token} wearing a rainbow scarf."
    "a ${unique_token} ${class_token} wearing a black top hat and a monocle."
    "a ${unique_token} ${class_token} in a chef outfit."
    "a ${unique_token} ${class_token} in a firefighter outfit."
    "a ${unique_token} ${class_token} in a police outfit."
    "a ${unique_token} ${class_token} wearing pink glasses."
    "a ${unique_token} ${class_token} wearing a yellow shirt."
    "a ${unique_token} ${class_token} in a purple wizard outfit."
    "a red ${unique_token} ${class_token}."
    "a purple ${unique_token} ${class_token}."
    "a shiny ${unique_token} ${class_token}."
    "a wet ${unique_token} ${class_token}."
    "a cube shaped ${unique_token} ${class_token}"
  )

  prompt_test_list=(
    "a ${class_token} in the jungle"
    "a ${class_token} in the snow"
    "a ${class_token} on the beach"
    "a ${class_token} on a cobblestone street"
    "a ${class_token} on top of pink fabric"
    "a ${class_token} on top of a wooden floor"
    "a ${class_token} with a city in the background"
    "a ${class_token} with a mountain in the background"
    "a ${class_token} with a blue house in the background"
    "a ${class_token} on top of a purple rug in a forest"
    "a ${class_token} wearing a red hat"
    "a ${class_token} wearing a santa hat"
    "a ${class_token} wearing a rainbow scarf"
    "a ${class_token} wearing a black top hat and a monocle"
    "a ${class_token} in a chef outfit"
    "a ${class_token} in a firefighter outfit"
    "a ${class_token} in a police outfit"
    "a ${class_token} wearing pink glasses"
    "a ${class_token} wearing a yellow shirt"
    "a ${class_token} in a purple wizard outfit"
    "a red ${class_token}"
    "a purple ${class_token}"
    "a shiny ${class_token}"
    "a wet ${class_token}"
    "a cube shaped ${class_token}"
  )
fi

validation_prompt=${prompt_list[@]}
instance_prompt="a photo of ${unique_token} ${class_token}"
class_prompt="a photo of ${class_token}"

# export HF_HOME='/tmp'
export HF_HOME="/is/cluster/yxiu/.cache"
export CUDA_HOME="/is/software/nvidia/cuda-11.7"

export MODEL_NAME="stabilityai/stable-diffusion-2-1"
# export MODEL_NAME="runwayml/stable-diffusion-v1-5" 
export PEFT_TYPE="lora"
export PROJECT_NAME="${PEFT_TYPE}-dreambooth-stable-final"
export INSTANCE_DIR="./data/dreambooth/dataset/${selected_subject}"
export CLASS_DIR="./data/class_data/${class_token}"
export OUTPUT_DIR="./data/output/${PEFT_TYPE}"


accelerate launch --main_process_port=29505 train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir="$CLASS_DIR" \
  --output_dir=$OUTPUT_DIR \
  --wandb_project_name=$PROJECT_NAME \
  --wandb_run_name=$selected_subject \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="$instance_prompt" \
  --validation_prompt="$validation_prompt" \
  --class_prompt="$class_prompt" \
  --resolution=512 \
  --train_batch_size=4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --use_lora \
  --lora_r=64 \
  --lora_alpha=27 \
  --lora_dropout=0.1 \
  --lora_bias="lora_only" \
  --learning_rate=3e-5 \
  --max_train_steps=2000 \
  --checkpointing_steps=500 \
  --validation_steps=3000 \
  --enable_xformers_memory_efficient_attention \