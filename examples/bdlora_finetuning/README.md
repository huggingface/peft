# BD-LoRA Finetuning

Block-Diagonal LoRA (BD-LoRA) is a LoRA variant in which some LoRA factors are constrained to be block-diagonal. 
This allows faster serving by eliminating communication overheads when running inference on multiple GPU, at the same finetuning performance as vanilla LoRA. 

To get an overview on how to use BD-LoRA, please view the Python notebook at `peft/examples/bdlora_finetuning/bdlora_peft_demo.ipynb`. 

To benefit from inference speed-ups, you need an inference engine that is compatible with BD-LoRA. At the moment, there is an experimental PR at https://github.com/vllm-project/vllm/pull/28136 which allows you to use BD-LoRA in vLLM. If you find this work useful, consider leaving a comment there. 

To install, you can clone the GitHub repository connected to the fork at https://github.com/Conzel/vllm/tree/bdlora-bk. Then, install vLLM following the usual instructions: https://docs.vllm.ai/en/stable/getting_started/installation/. We assume that you have a hardware setup with at least 2 available GPUs. 

This example folder contains 3 scripts:
- `bdlora_peft_demo.ipynb` Showcases how to instantiate a BD-LoRA model, train it, and save/reload the weights. 
- `vllm_server.bash` Spins up a BD-LoRA compatible vLLM server. To use it, you need to run the notebook once to create adapters with the correct format. 
- `chat.py` Can be used to query the vLLM server after it has finished booting up. Usage example: `python3 chat.py --target lora1`.