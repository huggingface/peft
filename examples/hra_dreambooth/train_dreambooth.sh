
CLASS_IDX=$1

# Define the UNIQUE_TOKEN, CLASS_TOKENs, and SUBJECT_NAMES
UNIQUE_TOKEN="qwe"

SUBJECT_NAMES=(
    "backpack" "backpack_dog" "bear_plushie" "berry_bowl" "can"
    "candle" "cat" "cat2" "clock" "colorful_sneaker"
    "dog" "dog2" "dog3" "dog5" "dog6"
    "dog7" "dog8" "duck_toy" "fancy_boot" "grey_sloth_plushie"
    "monster_toy" "pink_sunglasses" "poop_emoji" "rc_car" "red_cartoon"
    "robot_toy" "shiny_sneaker" "teapot" "vase" "wolf_plushie"
)

CLASS_TOKENs=(
    "backpack" "backpack" "stuffed animal" "bowl" "can"
    "candle" "cat" "cat" "clock" "sneaker"
    "dog" "dog" "dog" "dog" "dog"
    "dog" "dog" "toy" "boot" "stuffed animal"
    "toy" "glasses" "toy" "toy" "cartoon"
    "toy" "sneaker" "teapot" "vase" "stuffed animal"
)

CLASS_TOKEN=${CLASS_TOKENs[$CLASS_IDX]}
SELECTED_SUBJECT=${SUBJECT_NAMES[$CLASS_IDX]}

if [[ $CLASS_IDX =~ ^(0|1|2|3|4|5|8|9|17|18|19|20|21|22|23|24|25|26|27|28|29)$ ]]; then
  PROMPT_LIST=(
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} in the jungle."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} in the snow."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} on the beach."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} on a cobblestone street."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} on top of pink fabric."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} on top of a wooden floor."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} with a city in the background."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} with a mountain in the background."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} with a blue house in the background."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} on top of a purple rug in a forest."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} with a wheat field in the background."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} with a tree and autumn leaves in the background."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} with the Eiffel Tower in the background."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} floating on top of water."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} floating in an ocean of milk."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} on top of green grass with sunflowers around it."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} on top of a mirror."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} on top of the sidewalk in a crowded street."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} on top of a dirt road."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} on top of a white rug."
    "a red ${UNIQUE_TOKEN} ${CLASS_TOKEN}."
    "a purple ${UNIQUE_TOKEN} ${CLASS_TOKEN}."
    "a shiny ${UNIQUE_TOKEN} ${CLASS_TOKEN}."
    "a wet ${UNIQUE_TOKEN} ${CLASS_TOKEN}."
    "a cube shaped ${UNIQUE_TOKEN} ${CLASS_TOKEN}."
  )

  prompt_test_list=(
    "a ${CLASS_TOKEN} in the jungle"
    "a ${CLASS_TOKEN} in the snow"
    "a ${CLASS_TOKEN} on the beach"
    "a ${CLASS_TOKEN} on a cobblestone street"
    "a ${CLASS_TOKEN} on top of pink fabric"
    "a ${CLASS_TOKEN} on top of a wooden floor"
    "a ${CLASS_TOKEN} with a city in the background"
    "a ${CLASS_TOKEN} with a mountain in the background"
    "a ${CLASS_TOKEN} with a blue house in the background"
    "a ${CLASS_TOKEN} on top of a purple rug in a forest"
    "a ${CLASS_TOKEN} with a wheat field in the background"
    "a ${CLASS_TOKEN} with a tree and autumn leaves in the background"
    "a ${CLASS_TOKEN} with the Eiffel Tower in the background"
    "a ${CLASS_TOKEN} floating on top of water"
    "a ${CLASS_TOKEN} floating in an ocean of milk"
    "a ${CLASS_TOKEN} on top of green grass with sunflowers around it"
    "a ${CLASS_TOKEN} on top of a mirror"
    "a ${CLASS_TOKEN} on top of the sidewalk in a crowded street"
    "a ${CLASS_TOKEN} on top of a dirt road"
    "a ${CLASS_TOKEN} on top of a white rug"
    "a red ${CLASS_TOKEN}"
    "a purple ${CLASS_TOKEN}"
    "a shiny ${CLASS_TOKEN}"
    "a wet ${CLASS_TOKEN}"
    "a cube shaped ${CLASS_TOKEN}"
  )

else
  PROMPT_LIST=(
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} in the jungle."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} in the snow."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} on the beach."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} on a cobblestone street."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} on top of pink fabric."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} on top of a wooden floor."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} with a city in the background."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} with a mountain in the background."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} with a blue house in the background."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} on top of a purple rug in a forest."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} wearing a red hat."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} wearing a santa hat."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} wearing a rainbow scarf."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} wearing a black top hat and a monocle."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} in a chef outfit."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} in a firefighter outfit."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} in a police outfit."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} wearing pink glasses."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} wearing a yellow shirt."
    "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} in a purple wizard outfit."
    "a red ${UNIQUE_TOKEN} ${CLASS_TOKEN}."
    "a purple ${UNIQUE_TOKEN} ${CLASS_TOKEN}."
    "a shiny ${UNIQUE_TOKEN} ${CLASS_TOKEN}."
    "a wet ${UNIQUE_TOKEN} ${CLASS_TOKEN}."
    "a cube shaped ${UNIQUE_TOKEN} ${CLASS_TOKEN}."
  )

  prompt_test_list=(
    "a ${CLASS_TOKEN} in the jungle"
    "a ${CLASS_TOKEN} in the snow"
    "a ${CLASS_TOKEN} on the beach"
    "a ${CLASS_TOKEN} on a cobblestone street"
    "a ${CLASS_TOKEN} on top of pink fabric"
    "a ${CLASS_TOKEN} on top of a wooden floor"
    "a ${CLASS_TOKEN} with a city in the background"
    "a ${CLASS_TOKEN} with a mountain in the background"
    "a ${CLASS_TOKEN} with a blue house in the background"
    "a ${CLASS_TOKEN} on top of a purple rug in a forest"
    "a ${CLASS_TOKEN} wearing a red hat"
    "a ${CLASS_TOKEN} wearing a santa hat"
    "a ${CLASS_TOKEN} wearing a rainbow scarf"
    "a ${CLASS_TOKEN} wearing a black top hat and a monocle"
    "a ${CLASS_TOKEN} in a chef outfit"
    "a ${CLASS_TOKEN} in a firefighter outfit"
    "a ${CLASS_TOKEN} in a police outfit"
    "a ${CLASS_TOKEN} wearing pink glasses"
    "a ${CLASS_TOKEN} wearing a yellow shirt"
    "a ${CLASS_TOKEN} in a purple wizard outfit"
    "a red ${CLASS_TOKEN}"
    "a purple ${CLASS_TOKEN}"
    "a shiny ${CLASS_TOKEN}"
    "a wet ${CLASS_TOKEN}"
    "a cube shaped ${CLASS_TOKEN}"
  )
fi

VALIDATION_PROMPT=${PROMPT_LIST[@]}
INSTANCE_PROMPT="a photo of ${UNIQUE_TOKEN} ${CLASS_TOKEN}"
CLASS_PROMPT="a photo of ${CLASS_TOKEN}"

export MODEL_NAME="stabilityai/stable-diffusion-2-1"

PEFT_TYPE="hra"
HRA_R=8

export PROJECT_NAME="dreambooth_${PEFT_TYPE}"
export RUN_NAME="${SELECTED_SUBJECT}_${PEFT_TYPE}_${HRA_R}"
export INSTANCE_DIR="./data/dreambooth/dataset/${SELECTED_SUBJECT}"
export CLASS_DIR="./data/class_data/${CLASS_TOKEN}"
export OUTPUT_DIR="./data/output/${PEFT_TYPE}"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir="$CLASS_DIR" \
  --output_dir=$OUTPUT_DIR \
  --project_name=$PROJECT_NAME \
  --run_name=$RUN_NAME \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="$INSTANCE_PROMPT" \
  --validation_prompt="$VALIDATION_PROMPT" \
  --class_prompt="$CLASS_PROMPT" \
  --resolution=512 \
  --train_batch_size=1 \
  --num_dataloader_workers=2 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --use_hra \
  --hra_r=$HRA_R \
  --hra_bias="hra_only" \
  --learning_rate=5e-3 \
  --max_train_steps=510 \
  --checkpointing_steps=200 \
  --validation_steps=200 \
  --enable_xformers_memory_efficient_attention \
  --report_to="none" \