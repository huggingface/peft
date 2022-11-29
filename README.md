# 🤗 pets
Parameter-Efficient Tuning at Scale with 🤗 Accelerate

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



