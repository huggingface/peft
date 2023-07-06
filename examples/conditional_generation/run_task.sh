
bash setup.sh conditional_generation
python peft_adalora_seq2seq.py 2>&1 | tee peft_adalora_seq2seq.log
python peft_lora_seq2seq_accelerate_ds_zero3_offload.py 2>&1 | tee peft_lora_seq2seq_accelerate_ds_zero3_offload.log
python peft_lora_seq2seq_accelerate_fsdp.py 2>&1 | tee peft_lora_seq2seq_accelerate_fsdp.log
