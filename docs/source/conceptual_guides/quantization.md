# Quantization 


Quantization can be defined as the process of mapping values from a large set of real numbers to values in a small discrete set. Typically, this involves transforming continuous inputs into fixed values at the output. 

In simpler terms,  quantization in neural networks is the process of reducing the precision of weights , biases and activations of a neural networks in order to compressed or reduce the model size but also to reduce the computational requirements of the model  , 

without significantlly impacting the accuracy of the models !!!😲 

Types of Quantization Strategies

* Post-Training Quantization

* Quantization-Aware Training

**Post-training Quantization is more common these days with individuals such as [THE-Bloke](https://huggingface.co/TheBloke) with their GGML or GPTQ models**


## Quantizaton  has a number of advantages: 

*  Trimmed Memory Consumption: Quantized models require significantly less memory storage, a crucial advantage for deployment on devices with restricted memory capacity.

*  Reduced Energy Consumption: Theoretically, quantized models may consume less energy due to reduced data movement and storage operations, contributing to sustainability.

* Speed Boost: When your model is quantized, it can perform computations more quickly. This means faster predictions and responses, making it ideal for applications that require speedy performance.

  
# Motivation 

Regardless of the quantization method chosen, our main motivation is to achieve the following:

* Improve the Inference speed :  Training deep neural networks is a computationally intensive task, involving complex mathematical operations across numerous parameters. The 
computational expense becomes particularly apparent during the inference phase, where the trained model makes predictions on new data. Inference speed is crucial for real-time applications, such as image recognition in mobile apps or voice assistants.

  For instance, imagine a deep learning model responsible for identifying objects in a live video stream. The model needs to process each frame quickly to provide timely predictions. This rapid processing becomes challenging when dealing with large models that have a high number of parameters, such as state-of-the-art language models.

  Here's where quantization comes to the rescue. By reducing the precision of the model's parameters during inference, we can significantly cut down on the computational resources required. Specifically, moving from the standard 32-bit floating-point representation (float32) to 8-bit integer representation (int8) allows us to perform computations with fewer bits, resulting in faster and more efficient inference without sacrificing essential information.


* The need for quantized models : There is an increasing demands to run these huge models on our laptops or small devices such as  mobile phone . 
  
  None of the above is possible without **Quantization**   


## Supported Tuners to perform quantization in 🤗 Hugging face PEFT

🤗 PEFT provides different fine-tuning techniques  to perform quantization:

* The `LoRA` techniques allows us to
[quantize and fine-tune models ](https://huggingface.co/docs/peft/main/en/package_reference/lora)by reducing the number of trainable parameters.

* The `AdaLora` techniques also allows us to [quantize and fine-tune models](https://huggingface.co/docs/peft/main/en/package_reference/adalora) but unlike LoRA, which distributes parameters evenly across all modules. More parameters are budgeted for important weight matrices and layers while less important ones receive fewer parameters.

* The `ia3` allows us to [quantize and fine-tune models](https://huggingface.co/docs/peft/main/en/package_reference/ia3) it is a method that adds three learned vectors to rescale the keys and values of the self-attention and encoder-decoder attention layers, and the intermediate activation of the position-wise feed-forward network.


## References

- [All about Quantization ](https://www.youtube.com/watch?v=UQlsqdwCQdc&t=478s) you-tube video
- [LLM Series — Parameter Efficient Fine Tuning](https://medium.com/@abonia/llm-series-parameter-efficient-fine-tuning-e9839fae44ac)
- [🤗 transformers quantization documentation page](https://huggingface.co/docs/transformers/main_classes/quantization) 