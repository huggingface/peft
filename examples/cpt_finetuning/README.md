
# Context-aware Prompt Tuning: Advancing In-Context Learning with Adversarial Methods
## Introduction ([Paper](https://arxiv.org/abs/2410.17222), [Code](https://github.com/tsachiblau/Context-aware-Prompt-Tuning-Advancing-In-Context-Learning-with-Adversarial-Methods), [Notebook](cpt_train_and_inference.ipynb), [Colab](https://colab.research.google.com/drive/1UhQDVhZ9bDlSk1551SuJV8tIUmlIayta?usp=sharing))
Large Language Models (LLMs) can perform few-shot learning using either optimization-based approaches or In-Context Learning (ICL). Optimization-based methods often suffer from overfitting, as they require updating a large number of parameters with limited data. In contrast, ICL avoids overfitting but typically underperforms compared to optimization-based methods and is highly sensitive to the selection, order, and format of demonstration examples.

To overcome these challenges, we introduce Context-aware Prompt Tuning (CPT), a method inspired by ICL, Prompt Tuning (PT), and adversarial attacks. 
CPT builds on the ICL strategy of concatenating examples before the input, extending it by incorporating PT-like learning to refine the context embedding through iterative optimization, extracting deeper insights from the training examples. Our approach carefully modifies specific context tokens, considering the unique structure of the examples within the context.

In addition to updating the context with PT-like optimization, CPT draws inspiration from adversarial attacks, adjusting the input based on the labels present in the context while preserving the inherent value of the user-provided data. 
To ensure robustness and stability during optimization, we employ a projected gradient descent algorithm, constraining token embeddings to remain close to their original values and safeguarding the quality of the context.
Our method has demonstrated superior accuracy across multiple classification tasks using various LLM models, outperforming existing baselines and effectively addressing the overfitting challenge in few-shot learning.


<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/cpt.png"/>
</div>
<small>CPT optimizing only specific token embeddings while keeping the rest of the model frozen <a href="https://huggingface.co/papers/2410.17222">(image source)</a>.</small>

---

## Dataset Creation and Collation for CPT

This document explains how to prepare datasets for **Context-Aware Prompt Tuning (CPT)** and align these processes with the CPT paper.

---

### Template-Based Tokenization

#### Purpose
Templates define the structure of the input-output pairs, enabling the model to interpret the task within a unified context.

- **Input Templates**:
  Templates such as `"input: {sentence}"` format the raw input sentences. The `{sentence}` placeholder is replaced with the actual input text.

- **Output Templates**:
  Similarly, templates like `"output: {label}"` format the labels (`positive`, `negative`, etc.).

- **Separator Tokens**:
  Separators are used to distinguish between different parts of the input (e.g., input text and labels) and between examples.

#### Paper Reference
- Refer to **Section 3.1** of the paper, where template-based tokenization is described as a critical step in structuring inputs for CPT.

#### How it Helps
Templates provide context-aware structure, ensuring the model does not overfit by utilizing structured input-output formats. Using cpt_tokens_type_mask, we gain fine-grained information about the roles of different tokens in the input-output structure. This enables the model to:

1. Refrain from Updating Label Tokens: Prevent overfitting to label tokens by excluding their gradients during training.
2. Apply Different Projection Norms: Use type-specific projections for different parts of the input during Projected Gradient Descent (PGD), enhancing robustness and generalization.


#### Paper Reference

These steps are directly informed by the principles outlined in the CPT paper, particularly in Sections **3.1**, **3.2**, and **3.3**.





## Citation
```bib
@article{   
    blau2025cpt, 
    title={Context-Aware Prompt Tuning: Advancing In-Context Learning with Adversarial Methods}, 
    author={Tsachi Blau, Moshe Kimhi, Yonatan Belinkov, Alexander Bronstein, Chaim Baskin}, 
    journal={arXiv preprint arXiv:2410.17222}}, 
    year={2025} 
}
```
