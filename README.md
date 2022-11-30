# 🤗 PET
Parameter-Efficient Tuning. Intergrated with 🤗 Accelerate to scale seamlessly to large models using PyTorch FSDP. 

Supported methods:

1. Prefix Tuning
2. P-Tuning
3. Prompt Tuning
4. LoRA [in backlog]

## Models support matrix

### Sequence Classification
|            | Prefix Tuning | P-Tuning  | Prompt Tuning | LoRA  | 
| --------- | ---- | ---- | ---- | ---- |
| BERT           | ✅  | ✅  | ✅  |   |  
| RoBERTa        | ✅  | ✅  | ✅  |   |
| GPT-2          | ✅  | ✅  | ✅  |   | 
| Bloom          | ✅  | ✅  | ✅  |   |   
| OPT            | ✅  | ✅  | ✅  |   |
| GPT-Neo        | ✅  | ✅  | ✅  |   |
| GPT-J          | ✅  | ✅  | ✅  |   |
| Deberta        |   |   |   |   | 
| Deberta-v2     |   |   |   |   |

### Causal Language Modeling
|            | Prefix Tuning | P-Tuning  | Prompt Tuning | LoRA  | 
| --------- | ---- | ---- | ---- | ---- |
| GPT-2          | ✅  | ✅  | ✅  |   |
| Bloom          | ✅  | ✅  | ✅  |   |
| OPT            | ✅  | ✅  | ✅  |   |
| GPT-Neo        | ✅  | ✅  | ✅  |   |
| GPT-J          | ✅  | ✅  | ✅  |   |

### Conditional Generation
|            | Prefix Tuning | P-Tuning  | Prompt Tuning | LoRA  | 
| --------- | ---- | ---- | ---- | ---- |
| T5        | ✅  | ✅  | ✅  |   |
| BART      | ✅  | ✅  | ✅  |   |


## Caveats:
1. Doesn't work currently with DeeSpeed ZeRO Stage-3. Extending support with DeeSpeed ZeRO Stage-3 is in backlog.


