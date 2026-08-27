# KappaTune Experiment

This script compares different fine-tuning strategies on a downstream task (gsm8k) while measuring **catastrophic forgetting** on a general-knowledge control dataset (WikiText). For further details see the [KappaTune paper](https://arxiv.org/abs/2506.16289).

- **Baseline**: No adaptation
- **LoRA_Global**: Classic LoRA on common projections (`q_proj`, `k_proj`, `o_proj`, `v_proj`, `gate_proj`, `up_proj`, `down_proj` )
- **KappaTune_LoRA**: The new `KappaTuneSelector` with relative selection (`top_p=0.2`)

The goal is to show that KappaTune achieves similar task adaptation **while forgetting less** of the original pre-trained knowledge.

KappaTune is recommended when catastrophic forgetting is a concern. If your fine-tuning data is closely aligned with the model's pretraining distribution, it can even decrease general/Wiki perplexity. This is the case of fine-tuning on math data like GSM8K (adopted in this experiment) for some models. Then unrestricted LoRA over all layers may yield better results, since it reinforces pre-training.

### Key hyperparameters to play with

| Hyperparameter                | Location                          | Default          | What it controls                                      | Recommendation                                      |
|-------------------------------|-----------------------------------|------------------|-------------------------------------------------------|-----------------------------------------------------|
| `top_p`                       | KappaTune block                   | `0.2`            | Fraction of best (lowest κ) modules selected          | 0.1–0.3 (lower = more conservative)                 |
| `num_modules`                 | KappaTune block (alternative)     | `None`           | Fixed number of modules                               | Use instead of `top_p` for strict budget            |
| `r` (rank)                    | Both LoRA configs                 | `16` / `190`     | LoRA rank (controls trainable parameters)             | Keep total trainable params similar between runs    |
| `LR` (learning rate)          | Top of script                     | `2e-4`           | Training speed and stability                          | 1e-4 – 5e-4                                         |
| `num_train_epochs`            | Top of script                     | `10`             | Total training steps                                  | Increase for stronger adaptation                    |
| `MODEL_ID`                    | Top of script                     | DeepSeek-V2-Lite | Base model                                            | Try Mistral, Qwen, etc. MoE yields the best results |
| `max_dim_size_to_analyze`     | `KappaTuneSelector`               | `16384`          | Max matrix size for SVD (memory / speed trade-off)    | Increase only if you have very high VRAM            |

## Expected results

Running the script with default parameters produces the following behavior.

<details>
<summary><strong>KappaTune (training log)</strong></summary>

```text
========================================
>>> EXPERIMENT: KappaTune_LoRA
========================================
[KappaTune] Selecting target modules using PEFT KappaTuneSelector...
trainable params: 219,188,480 (0.47%)
#trainable tensors: 370
#trainable params: 219,188,480
{'loss': 1.2208, 'grad_norm': 0.15972940623760223, 'learning_rate': 0.00019666666666666666, 'epoch': 0.87}
{'loss': 1.07, 'grad_norm': 0.26267319917678833, 'learning_rate': 0.00019250000000000002, 'epoch': 1.7}
{'loss': 1.0124, 'grad_norm': 0.2723085284233093, 'learning_rate': 0.00018833333333333335, 'epoch': 2.52}
{'loss': 0.9436, 'grad_norm': 0.29701513051986694, 'learning_rate': 0.00018416666666666665, 'epoch': 3.35}
{'loss': 0.9114, 'grad_norm': 0.3046768009662628, 'learning_rate': 0.00018, 'epoch': 4.17}
{'loss': 0.8915, 'grad_norm': 0.47676295042037964, 'learning_rate': 0.00017583333333333334, 'epoch': 5.0}
{'loss': 0.8728, 'grad_norm': 0.24124978482723236, 'learning_rate': 0.00017166666666666667, 'epoch': 5.87}
{'loss': 0.8385, 'grad_norm': 0.20988021790981293, 'learning_rate': 0.0001675, 'epoch': 6.7}
{'loss': 0.8389, 'grad_norm': 0.21968263387680054, 'learning_rate': 0.00016333333333333334, 'epoch': 7.52}
{'loss': 0.8224, 'grad_norm': 0.04499625042080879, 'learning_rate': 0.00015916666666666667, 'epoch': 8.35}
{'loss': 0.7997, 'grad_norm': 0.05307907983660698, 'learning_rate': 0.000155, 'epoch': 9.17}
{'loss': 0.8057, 'grad_norm': 0.05258958786725998, 'learning_rate': 0.00015083333333333333, 'epoch': 10.0}
{'loss': 0.7907, 'grad_norm': 0.05647141858935356, 'learning_rate': 0.00014666666666666666, 'epoch': 10.87}
{'loss': 0.7804, 'grad_norm': 0.07924238592386246, 'learning_rate': 0.00014250000000000002, 'epoch': 11.7}
{'loss': 0.7705, 'grad_norm': 0.03199317678809166, 'learning_rate': 0.00013833333333333333, 'epoch': 12.52}
{'loss': 0.76, 'grad_norm': 0.030626816675066948, 'learning_rate': 0.00013416666666666666, 'epoch': 13.35}
{'loss': 0.7456, 'grad_norm': 0.03376801684498787, 'learning_rate': 0.00013000000000000002, 'epoch': 14.17}
{'loss': 0.7352, 'grad_norm': 0.045162301510572433, 'learning_rate': 0.00012583333333333335, 'epoch': 15.0}
{'loss': 0.7221, 'grad_norm': 0.04105797037482262, 'learning_rate': 0.00012166666666666667, 'epoch': 15.87}
{'loss': 0.7118, 'grad_norm': 0.04653630033135414, 'learning_rate': 0.00011750000000000001, 'epoch': 16.7}
{'loss': 0.7071, 'grad_norm': 0.0477643646299839, 'learning_rate': 0.00011333333333333334, 'epoch': 17.52}
{'loss': 0.67, 'grad_norm': 0.05408667400479317, 'learning_rate': 0.00010916666666666666, 'epoch': 18.35}
{'loss': 0.6753, 'grad_norm': 0.05562206730246544, 'learning_rate': 0.000105, 'epoch': 19.17}
{'loss': 0.6529, 'grad_norm': 0.08778411149978638, 'learning_rate': 0.00010083333333333334, 'epoch': 20.0}
{'loss': 0.6356, 'grad_norm': 0.07903064042329788, 'learning_rate': 9.666666666666667e-05, 'epoch': 20.87}
{'loss': 0.6198, 'grad_norm': 0.08349727094173431, 'learning_rate': 9.250000000000001e-05, 'epoch': 21.7}
{'loss': 0.604, 'grad_norm': 0.08645208925008774, 'learning_rate': 8.833333333333333e-05, 'epoch': 22.52}
{'loss': 0.5897, 'grad_norm': 0.09194190055131912, 'learning_rate': 8.416666666666668e-05, 'epoch': 23.35}
{'loss': 0.5649, 'grad_norm': 0.10126981139183044, 'learning_rate': 8e-05, 'epoch': 24.17}
{'loss': 0.5483, 'grad_norm': 0.13687381148338318, 'learning_rate': 7.583333333333334e-05, 'epoch': 25.0}
{'loss': 0.5323, 'grad_norm': 0.13106191158294678, 'learning_rate': 7.166666666666667e-05, 'epoch': 25.87}
{'loss': 0.5147, 'grad_norm': 0.1281006783246994, 'learning_rate': 6.750000000000001e-05, 'epoch': 26.7}
{'loss': 0.4941, 'grad_norm': 0.1377001851797104, 'learning_rate': 6.333333333333333e-05, 'epoch': 27.52}
{'loss': 0.4845, 'grad_norm': 0.14247003197669983, 'learning_rate': 5.916666666666667e-05, 'epoch': 28.35}
{'loss': 0.4669, 'grad_norm': 0.14682190120220184, 'learning_rate': 5.500000000000001e-05, 'epoch': 29.17}
{'loss': 0.454, 'grad_norm': 0.19749927520751953, 'learning_rate': 5.0833333333333333e-05, 'epoch': 30.0}
{'loss': 0.4323, 'grad_norm': 0.1622404158115387, 'learning_rate': 4.666666666666667e-05, 'epoch': 30.87}
{'loss': 0.4274, 'grad_norm': 0.15755769610404968, 'learning_rate': 4.25e-05, 'epoch': 31.7}
{'loss': 0.4086, 'grad_norm': 0.1696886271238327, 'learning_rate': 3.8333333333333334e-05, 'epoch': 32.52}
{'loss': 0.4006, 'grad_norm': 0.14865559339523315, 'learning_rate': 3.4166666666666666e-05, 'epoch': 33.35}
{'loss': 0.3934, 'grad_norm': 0.14790022373199463, 'learning_rate': 3e-05, 'epoch': 34.17}
{'loss': 0.3777, 'grad_norm': 0.20866677165031433, 'learning_rate': 2.5833333333333336e-05, 'epoch': 35.0}
{'loss': 0.3712, 'grad_norm': 0.17520156502723694, 'learning_rate': 2.1666666666666667e-05, 'epoch': 35.87}
{'loss': 0.3668, 'grad_norm': 0.16120171546936035, 'learning_rate': 1.75e-05, 'epoch': 36.7}
{'loss': 0.3583, 'grad_norm': 0.16624851524829865, 'learning_rate': 1.3333333333333333e-05, 'epoch': 37.52}
{'loss': 0.357, 'grad_norm': 0.15067771077156067, 'learning_rate': 9.166666666666666e-06, 'epoch': 38.35}
{'loss': 0.3524, 'grad_norm': 0.15819120407104492, 'learning_rate': 5e-06, 'epoch': 39.17}
{'loss': 0.3466, 'grad_norm': 0.1855546236038208, 'learning_rate': 8.333333333333333e-07, 'epoch': 40.0}
{'train_runtime': 2871.3103, 'train_samples_per_second': 12.538, 'train_steps_per_second': 0.084, 'train_loss': 0.6427203471461932, 'epoch': 40.0}

========================================
>>> EXPERIMENT: Baseline
========================================

========================================
>>> EXPERIMENT: LoRA_Global
========================================
trainable params: 218,103,808 || all params: 46,920,896,512 || trainable%: 0.4648
{'loss': 1.2122, 'grad_norm': 0.03368454799056053, 'learning_rate': 0.00019666666666666666, 'epoch': 0.87}
{'loss': 1.0674, 'grad_norm': 0.04771586135029793, 'learning_rate': 0.00019250000000000002, 'epoch': 1.7}
{'loss': 0.9889, 'grad_norm': 0.030208367854356766, 'learning_rate': 0.00018833333333333335, 'epoch': 2.52}
{'loss': 0.9269, 'grad_norm': 0.0202629417181015, 'learning_rate': 0.00018416666666666665, 'epoch': 3.35}
{'loss': 0.9146, 'grad_norm': 0.01595970056951046, 'learning_rate': 0.00018, 'epoch': 4.17}
{'loss': 0.8983, 'grad_norm': 0.017851779237389565, 'learning_rate': 0.00017583333333333334, 'epoch': 5.0}
{'loss': 0.8914, 'grad_norm': 0.01525798998773098, 'learning_rate': 0.00017166666666666667, 'epoch': 5.87}
{'loss': 0.8733, 'grad_norm': 0.01363384909927845, 'learning_rate': 0.0001675, 'epoch': 6.7}
{'loss': 0.8712, 'grad_norm': 0.014126025140285492, 'learning_rate': 0.00016333333333333334, 'epoch': 7.52}
{'loss': 0.8673, 'grad_norm': 0.01614651270210743, 'learning_rate': 0.00015916666666666667, 'epoch': 8.35}
{'loss': 0.8461, 'grad_norm': 0.014323701150715351, 'learning_rate': 0.000155, 'epoch': 9.17}
{'loss': 0.8519, 'grad_norm': 0.022168157622218132, 'learning_rate': 0.00015083333333333333, 'epoch': 10.0}
{'loss': 0.8326, 'grad_norm': 0.017714861780405045, 'learning_rate': 0.00014666666666666666, 'epoch': 10.87}
{'loss': 0.8258, 'grad_norm': 0.01950528658926487, 'learning_rate': 0.00014250000000000002, 'epoch': 11.7}
{'loss': 0.8142, 'grad_norm': 0.021654563024640083, 'learning_rate': 0.00013833333333333333, 'epoch': 12.52}
{'loss': 0.803, 'grad_norm': 0.027227576822042465, 'learning_rate': 0.00013416666666666666, 'epoch': 13.35}
{'loss': 0.7892, 'grad_norm': 0.0281345397233963, 'learning_rate': 0.00013000000000000002, 'epoch': 14.17}
{'loss': 0.7759, 'grad_norm': 0.04052634909749031, 'learning_rate': 0.00012583333333333335, 'epoch': 15.0}
{'loss': 0.7614, 'grad_norm': 0.03630959987640381, 'learning_rate': 0.00012166666666666667, 'epoch': 15.87}
{'loss': 0.7474, 'grad_norm': 0.04881247878074646, 'learning_rate': 0.00011750000000000001, 'epoch': 16.7}
{'loss': 0.7449, 'grad_norm': 0.04792051389813423, 'learning_rate': 0.00011333333333333334, 'epoch': 17.52}
{'loss': 0.707, 'grad_norm': 0.059059303253889084, 'learning_rate': 0.00010916666666666666, 'epoch': 18.35}
{'loss': 0.713, 'grad_norm': 0.05515185743570328, 'learning_rate': 0.000105, 'epoch': 19.17}
{'loss': 0.6907, 'grad_norm': 0.09590236097574234, 'learning_rate': 0.00010083333333333334, 'epoch': 20.0}
{'loss': 0.6736, 'grad_norm': 0.07875961065292358, 'learning_rate': 9.666666666666667e-05, 'epoch': 20.87}
{'loss': 0.663, 'grad_norm': 0.08937060832977295, 'learning_rate': 9.250000000000001e-05, 'epoch': 21.7}
{'loss': 0.6445, 'grad_norm': 0.0950784757733345, 'learning_rate': 8.833333333333333e-05, 'epoch': 22.52}
{'loss': 0.6387, 'grad_norm': 0.08285810798406601, 'learning_rate': 8.416666666666668e-05, 'epoch': 23.35}
{'loss': 0.6182, 'grad_norm': 0.1019740179181099, 'learning_rate': 8e-05, 'epoch': 24.17}
{'loss': 0.6045, 'grad_norm': 0.1708088219165802, 'learning_rate': 7.583333333333334e-05, 'epoch': 25.0}
{'loss': 0.5959, 'grad_norm': 0.12375958263874054, 'learning_rate': 7.166666666666667e-05, 'epoch': 25.87}
{'loss': 0.5877, 'grad_norm': 0.1316744089126587, 'learning_rate': 6.750000000000001e-05, 'epoch': 26.7}
{'loss': 0.5698, 'grad_norm': 0.11958763003349304, 'learning_rate': 6.333333333333333e-05, 'epoch': 27.52}
{'loss': 0.5653, 'grad_norm': 0.11063854396343231, 'learning_rate': 5.916666666666667e-05, 'epoch': 28.35}
{'loss': 0.5502, 'grad_norm': 0.11866016685962677, 'learning_rate': 5.500000000000001e-05, 'epoch': 29.17}
{'loss': 0.5458, 'grad_norm': 0.1533481925725937, 'learning_rate': 5.0833333333333333e-05, 'epoch': 30.0}
{'loss': 0.5316, 'grad_norm': 0.13474972546100616, 'learning_rate': 4.666666666666667e-05, 'epoch': 30.87}
{'loss': 0.5269, 'grad_norm': 0.13297690451145172, 'learning_rate': 4.25e-05, 'epoch': 31.7}
{'loss': 0.5175, 'grad_norm': 0.13090570271015167, 'learning_rate': 3.8333333333333334e-05, 'epoch': 32.52}
{'loss': 0.5108, 'grad_norm': 0.1196967139840126, 'learning_rate': 3.4166666666666666e-05, 'epoch': 33.35}
{'loss': 0.5062, 'grad_norm': 0.13541598618030548, 'learning_rate': 3e-05, 'epoch': 34.17}
{'loss': 0.4965, 'grad_norm': 0.202300563454628, 'learning_rate': 2.5833333333333336e-05, 'epoch': 35.0}
{'loss': 0.492, 'grad_norm': 0.14462313055992126, 'learning_rate': 2.1666666666666667e-05, 'epoch': 35.87}
{'loss': 0.4898, 'grad_norm': 0.1338396966457367, 'learning_rate': 1.75e-05, 'epoch': 36.7}
{'loss': 0.4833, 'grad_norm': 0.11928340792655945, 'learning_rate': 1.3333333333333333e-05, 'epoch': 37.52}
{'loss': 0.4805, 'grad_norm': 0.1199464276432991, 'learning_rate': 9.166666666666666e-06, 'epoch': 38.35}
{'loss': 0.4801, 'grad_norm': 0.1222926527261734, 'learning_rate': 5e-06, 'epoch': 39.17}
{'loss': 0.4729, 'grad_norm': 0.14248400926589966, 'learning_rate': 8.333333333333333e-07, 'epoch': 40.0}
{'train_runtime': 2779.2046, 'train_samples_per_second': 12.953, 'train_steps_per_second': 0.086, 'train_loss': 0.7012442946434021, 'epoch': 40.0}

======================================================================
METHOD          | gsm8k PPL (Task train) |  gsm8k PPL (Task test) | Wiki PPL (General/control)
----------------------------------------------------------------------
KappaTune       | 1.4410             | 3.2826             | 13.7780           
Baseline        | 3.6899             | 3.4668             | 13.9841           
LoRA_Global     | 1.6593             | 3.5648             | 26.6836           
======================================================================


```

</details>

## Dense model with varying training effort

Since we are using small datasets, the adaptation effort is small, as is the risk of forgetting. To evaluate catastrophic forgetting during intensive fine-tuning we need big datasets or overfitting a small dataset. Therefore, I ran a set of experiments that track the performance of a dense LLM (Llama 8B) trained on the IMDB dataset over extended epochs, proxying the heavy gradient updates typical of massive datasets. The experiments use the same Python script, changing just these parameters:

```python
MODEL_ID = "unsloth/Meta-Llama-3.1-8B-Instruct"
imdb_ds = load_dataset("imdb", split="train[:1000]").train_test_split(test_size=0.1)
imdb_tokenized = imdb_ds.map(format_imdb).map(
    lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=256), 
    batched=True, remove_columns=imdb_ds["train"].column_names
)
    if method_name == "LoRA_Global":
        Target_modules = [
        "q_proj", 
        "k_proj", 
        "v_proj", 
        "o_proj", 
        "gate_proj", 
        "up_proj", 
        "down_proj"
        ]
        lora_config = LoraConfig(
            r=12, 
            target_modules=Target_modules, 
            task_type=TaskType.CAUSAL_LM, lora_dropout=0.05
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        LR=2e-4
        STP= #VAR <from 4 to 20>
    elif method_name == "KappaTune_LoRA":
        stable_modules_dic = find_kappa_target_modules(model, top_p=0.2)
        lora_config = LoraConfig(
            r=64,
            target_modules = stable_modules_dic["target_modules"], 
            target_parameters = stable_modules_dic["target_parameters"] if stable_modules_dic["target_parameters"] else None, 
            task_type=TaskType.CAUSAL_LM,
            lora_dropout=0.05,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        trainable = [(n, p.shape, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
        LR = 2e-4
        STP = # VAR <from 6 to 60>

```

The figure below plots task-specific adaptation (IMDB perplexity) against general knowledge retention (control Wiki perplexity). The results reveal a distinct divergence: while both methods perform comparably under light training loads, pushing into deeper convergence exposes KappaTune's structural advantage. As the model tightly fits the target data, standard LoRA exhibits a steep degradation in general knowledge (higher Wiki PPL), whereas KappaTune maintains a significantly flatter trajectory. This demonstrates its superior ability to isolate new learning and mitigate catastrophic forgetting even under sustained training pressure.


<img width="778" height="536" alt="image" src="https://github.com/user-attachments/assets/4cbbcaa2-e433-48cf-8764-67498462f686" />


In case of using this test framework for different experiments, it's worth highlighting that size matters.
KappaTune shows the strongest gains on larger models (≥7B) and especially on MoE architectures (many independent expert modules). In small, dense models, the benefit is reduced because there is a limited variety of independent tensors to choose from. A fair comparison of catastrophic forgetting should make both methods reach roughly the same level of adaptation to the new task (similar training PPL). Matching on test PPL is not sufficient, because the same test PPL can be achieved through overfitting (more forgetting) or underfitting (less forgetting).
