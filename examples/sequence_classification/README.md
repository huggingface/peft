1. Setup env - train all models (the technique is specialize and this task is run with example model name user can explore more, see more in code). There are some example .py in this task folder.
```
sh setup.sh
sh p_tuning_peft.sh
sh peft_no_lora_accelerate.sh
```
* To track top GPU vRAM usage by device, open a new terminal and run this script, pass device ID, for ex device 0:
```
sh ../memory_record_moreh.sh 0
```