
bash setup.sh casual_language_modeling
python mlflow_peft_lora_clm_accelerate_ds_zero3_offload.py 2>&1 | tee task.log
