OUTPUT=$1

CUDA_VISIBLE_DEVICES=0 python inference/gsm8k_inference.py --model $OUTPUT
CUDA_VISIBLE_DEVICES=0 python inference/MATH_inference.py --model $OUTPUT
