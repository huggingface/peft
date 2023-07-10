
bash setup.sh casual_language_modeling
export MLFLOW_TRACKING_URI="http://127.0.0.1:5001"
python mlflow_peft_lora_clm_accelerate_ds_zero3_offload.py 2>&1 | tee task.log
