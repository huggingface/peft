
prompt_idx=$1
class_idx=$2

hra_r=8

export MODEL_NAME="/data/shen_yuan/aliendao/dataroot/models/stabilityai/stable-diffusion-2-1-base"

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
    "a ${unique_token} ${class_token} in the jungle"
    "a ${unique_token} ${class_token} in the snow"
    "a ${unique_token} ${class_token} on the beach"
    "a ${unique_token} ${class_token} on a cobblestone street"
    "a ${unique_token} ${class_token} on top of pink fabric"
    "a ${unique_token} ${class_token} on top of a wooden floor"
    "a ${unique_token} ${class_token} with a city in the background"
    "a ${unique_token} ${class_token} with a mountain in the background"
    "a ${unique_token} ${class_token} with a blue house in the background"
    "a ${unique_token} ${class_token} on top of a purple rug in a forest"
    "a ${unique_token} ${class_token} with a wheat field in the background"
    "a ${unique_token} ${class_token} with a tree and autumn leaves in the background"
    "a ${unique_token} ${class_token} with the Eiffel Tower in the background"
    "a ${unique_token} ${class_token} floating on top of water"
    "a ${unique_token} ${class_token} floating in an ocean of milk"
    "a ${unique_token} ${class_token} on top of green grass with sunflowers around it"
    "a ${unique_token} ${class_token} on top of a mirror"
    "a ${unique_token} ${class_token} on top of the sidewalk in a crowded street"
    "a ${unique_token} ${class_token} on top of a dirt road"
    "a ${unique_token} ${class_token} on top of a white rug"
    "a red ${unique_token} ${class_token}"
    "a purple ${unique_token} ${class_token}"
    "a shiny ${unique_token} ${class_token}"
    "a wet ${unique_token} ${class_token}"
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
    "a ${unique_token} ${class_token} in the jungle"
    "a ${unique_token} ${class_token} in the snow"
    "a ${unique_token} ${class_token} on the beach"
    "a ${unique_token} ${class_token} on a cobblestone street"
    "a ${unique_token} ${class_token} on top of pink fabric"
    "a ${unique_token} ${class_token} on top of a wooden floor"
    "a ${unique_token} ${class_token} with a city in the background"
    "a ${unique_token} ${class_token} with a mountain in the background"
    "a ${unique_token} ${class_token} with a blue house in the background"
    "a ${unique_token} ${class_token} on top of a purple rug in a forest"
    "a ${unique_token} ${class_token} wearing a red hat"
    "a ${unique_token} ${class_token} wearing a santa hat"
    "a ${unique_token} ${class_token} wearing a rainbow scarf"
    "a ${unique_token} ${class_token} wearing a black top hat and a monocle"
    "a ${unique_token} ${class_token} in a chef outfit"
    "a ${unique_token} ${class_token} in a firefighter outfit"
    "a ${unique_token} ${class_token} in a police outfit"
    "a ${unique_token} ${class_token} wearing pink glasses"
    "a ${unique_token} ${class_token} wearing a yellow shirt"
    "a ${unique_token} ${class_token} in a purple wizard outfit"
    "a red ${unique_token} ${class_token}"
    "a purple ${unique_token} ${class_token}"
    "a shiny ${unique_token} ${class_token}"
    "a wet ${unique_token} ${class_token}"
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

validation_prompt=${prompt_list[$prompt_idx]}
test_prompt=${prompt_test_list[$prompt_idx]}
name="${selected_subject}-${prompt_idx}"
instance_prompt="a photo of ${unique_token} ${class_token}"
class_prompt="a photo of ${class_token}"

export OUTPUT_DIR="logs/${name}"
export INSTANCE_DIR="dreambooth/dataset/${selected_subject}"
export CLASS_DIR="class_data/${class_token}"

echo "class_token: $class_token, prompt: $validation_prompt"
accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir="$CLASS_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --instance_prompt="$instance_prompt" \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --class_prompt="$class_prompt" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-2 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1410 \
  --validation_prompt="$validation_prompt" \
  --seed="0" \
  --num_class_images=200 \
  --checkpointing_steps=200 \
  --validation_steps=200 \
  --use_hra \
  --hra_r=$hra_r \
  --hra_bias="hra_only"

echo "Please fill in the following parameters in the hra_dreambooth_inference.ipynb"
echo "BASE_MODEL_NAME = '$MODEL_NAME'"
echo "ADAPTER_MODEL_PATH = '$OUTPUT_DIR'"
echo "PROMPT = '$instance_prompt'"
