1. Setup env - train all models (the technique is specialize and this task is run with example model name user can explore more, see more in code). There are some example .py in this task folder.
```
bash setup.sh
bash p_tuning_peft.sh
bash peft_no_lora_accelerate.sh
```
* To track top GPU vRAM usage by device, open a new terminal and run this script, pass device ID, for ex device 0:
```
bash ../memory_record_moreh.sh 0
```
