1. Setup env - train all models (the technique is specialize and this task is run with example model name "bigscience/bloomz-7b1", dataset "twitter_complaints", user can explore more, see more in code)
```
sh run_task.sh
```
* To track top GPU vRAM usage by device, open a new terminal and run this script, pass device ID, for ex device 2:
```
sh ../memory_record_moreh.sh 2
```
