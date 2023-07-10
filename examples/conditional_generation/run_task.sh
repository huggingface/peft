
bash setup.sh conditional_generation
export MLFLOW_TRACKING_URI="http://127.0.0.1:5001"
python mlflow_peft_adalora_seq2seq.py 2>&1 | tee peft_adalora_seq2seq.log
python mlflow_peft_lora_seq2seq_accelerate_ds_zero3_offload.py 2>&1 | tee peft_lora_seq2seq_accelerate_ds_zero3_offload.log
python mlflow_peft_lora_seq2seq_accelerate_fsdp.py 2>&1 | tee peft_lora_seq2seq_accelerate_fsdp.log
