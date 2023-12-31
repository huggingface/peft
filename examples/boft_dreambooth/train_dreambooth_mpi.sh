# Define the UNIQUE_TOKEN, CLASS_TOKENs, and SUBJECT_NAMES
UNIQUE_TOKEN="qwe"
CLASS_TOKEN="two-story black modern institute building"
SELECTED_SUBJECT="mpi-is-building"

PROMPT_LIST=(
  "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} near a historic monument."
  "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} in a vineyard."
  "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} on a cliffside."
  "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} in a historic district."
  "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} on the lakeside."
  "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} in the jungle."
  "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} in the snow."
  "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} on the beach."
  "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} near the sea."
  "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} in the desert."
  "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} in the rainforest."
  "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} in a redwood forest."
  "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} on a cobblestone street."
  "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} with a city in the background."
  "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} with a mountain in the background."
  "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} with a wheat field in the background."
  "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} with a tree and autumn leaves in the background."
  "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} with the Eiffel Tower in the background."
  "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} on the moon, with a view of space."
  "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} under the northern lights in a polar region."
  "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} amidst skyscrapers in a futuristic city."
  "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} on a floating island in the sky."
  "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} on a Mars colony, with a red desert landscape."
  "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} in a dystopian world, amidst ruins and overgrown nature."
  "a ${UNIQUE_TOKEN} ${CLASS_TOKEN} in a cyberpunk cityscape, with neon lights and high-tech surroundings."
  "A ${UNIQUE_TOKEN} ${CLASS_TOKEN} in a cyberpunk city."
  "A ${UNIQUE_TOKEN} ${CLASS_TOKEN} in oil painting style."
  "A ${UNIQUE_TOKEN} ${CLASS_TOKEN} in watercolor style."
  "A ${UNIQUE_TOKEN} ${CLASS_TOKEN} in digital art style."
  "A ${UNIQUE_TOKEN} ${CLASS_TOKEN} in pencil sketch style."
  "A ${UNIQUE_TOKEN} ${CLASS_TOKEN} in charcoal drawing style."
  "A ${UNIQUE_TOKEN} ${CLASS_TOKEN} in pastel colors style."
  "A ${UNIQUE_TOKEN} ${CLASS_TOKEN} in acrylic painting style."
  "A ${UNIQUE_TOKEN} ${CLASS_TOKEN} in ink drawing style."
  "A ${UNIQUE_TOKEN} ${CLASS_TOKEN} in collage style."
  "A ${UNIQUE_TOKEN} ${CLASS_TOKEN} in graffiti style."
  "A ${UNIQUE_TOKEN} ${CLASS_TOKEN} in impressionist painting style."
  "A ${UNIQUE_TOKEN} ${CLASS_TOKEN} in cubist style."
  "A ${UNIQUE_TOKEN} ${CLASS_TOKEN} in abstract expressionist style."
  "A ${UNIQUE_TOKEN} ${CLASS_TOKEN} in art nouveau style."
  "A ${UNIQUE_TOKEN} ${CLASS_TOKEN} in surrealism style."
)

# VALIDATION_PROMPT="a ${UNIQUE_TOKEN} ${CLASS_TOKEN} in winter with snow"
VALIDATION_PROMPT=${PROMPT_LIST[@]}

INSTANCE_PROMPT="a photo of ${UNIQUE_TOKEN} ${CLASS_TOKEN}"
CLASS_PROMPT="a photo of ${CLASS_TOKEN}"

export CUDA_HOME="/is/software/nvidia/cuda-11.7"

export MODEL_NAME="stabilityai/stable-diffusion-2-1" 
# export MODEL_NAME="runwayml/stable-diffusion-v1-5"

export PEFT_TYPE="boft"
BLOCK_NUM=8
BLOCK_SIZE=0
N_BUTTERFLY_FACTOR=0
export PROJECT_NAME="dreambooth_${PEFT_TYPE}"
export RUN_NAME="${SELECTED_SUBJECT}_${PEFT_TYPE}_${BLOCK_NUM}${BLOCK_SIZE}${N_BUTTERFLY_FACTOR}"
export INSTANCE_DIR="./data/${SELECTED_SUBJECT}"
export CLASS_DIR="./data/class_data/${CLASS_TOKEN}"
export OUTPUT_DIR="./data/output/${PEFT_TYPE}"

. /home/zqiu/miniconda3/etc/profile.d/conda.sh
conda activate peft

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir="$CLASS_DIR" \
  --output_dir=$OUTPUT_DIR \
  --wandb_project_name=$PROJECT_NAME \
  --wandb_run_name=$RUN_NAME \
  --instance_prompt="$INSTANCE_PROMPT" \
  --validation_prompt="$VALIDATION_PROMPT" \
  --class_prompt="$CLASS_PROMPT" \
  --resolution=512 \
  --train_batch_size=4 \
  --num_dataloader_workers=2 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --use_boft \
  --boft_block_num=$BLOCK_NUM \
  --boft_block_size=$BLOCK_SIZE \
  --boft_n_butterfly_factor=$N_BUTTERFLY_FACTOR \
  --boft_dropout=0.1 \
  --boft_bias="boft_only" \
  --learning_rate=3e-5 \
  --max_train_steps=1050 \
  --checkpointing_steps=2000 \
  --validation_steps=200 \
  --enable_xformers_memory_efficient_attention \
  --report_to="wandb" \

  # --with_prior_preservation --prior_loss_weight=1.0 \