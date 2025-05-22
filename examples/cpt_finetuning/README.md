
# Context-aware Prompt Tuning: Advancing In-Context Learning with Adversarial Methods
## Introduction ([Paper](https://huggingface.co/papers/2410.17222), [Code](https://github.com/tsachiblau/Context-aware-Prompt-Tuning-Advancing-In-Context-Learning-with-Adversarial-Methods), [Notebook](cpt_train_and_inference.ipynb), [Colab](https://colab.research.google.com/drive/1UhQDVhZ9bDlSk1551SuJV8tIUmlIayta?usp=sharing))

> Large Language Models (LLMs) can perform few-shot learning using either optimization-based approaches or In-Context Learning (ICL). Optimization-based methods often suffer from overfitting, as they require updating a large number of parameters with limited data. In contrast, ICL avoids overfitting but typically underperforms compared to optimization-based methods and is highly sensitive to the selection, order, and format of demonstration examples. To overcome these challenges, we introduce Context-aware Prompt Tuning (CPT), a method inspired by ICL, Prompt Tuning (PT), and adversarial attacks. CPT builds on the ICL strategy of concatenating examples before the input, extending it by incorporating PT-like learning to refine the context embedding through iterative optimization, extracting deeper insights from the training examples. Our approach carefully modifies specific context tokens, considering the unique structure of the examples within the context. In addition to updating the context with PT-like optimization, CPT draws inspiration from adversarial attacks, adjusting the input based on the labels present in the context while preserving the inherent value of the user-provided data. To ensure robustness and stability during optimization, we employ a projected gradient descent algorithm, constraining token embeddings to remain close to their original values and safeguarding the quality of the context. Our method has demonstrated superior accuracy across multiple classification tasks using various LLM models, outperforming existing baselines and effectively addressing the overfitting challenge in few-shot learning.



<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/cpt.png"/>
</div>
<small>CPT optimizing only specific token embeddings while keeping the rest of the model frozen <a href="https://huggingface.co/papers/2410.17222">(image source)</a>.</small>

---

## Dataset Creation and Collation for CPT

This document explains how to prepare datasets for CPT, linking the dataset preparation processes in the code to the methods and principles described in the CPT paper, specifically in **Sections 3.1**, **3.2**, and **3.3**.

---

### Template-Based Tokenization

#### The Role of Templates
Templates define the structure of the input-output pairs, enabling the model to interpret the task within a unified context.

- **Input Templates**:  
  Templates like `"input: {sentence}"` structure raw input sentences. The `{sentence}` placeholder is replaced with the actual input text.

- **Output Templates**:  
  Templates such as `"output: {label}"` format the labels (e.g., `positive`, `negative`, etc.).

- **Separator Tokens**:  
  Separators distinguish different parts of the input, such as the input text and labels, as well as separate examples within the context.


#### How CPT Utilizes Context Structure

CPT leverages the context structure, encoded within the `cpt_tokens_type_mask`, to optimize the context effectively. to optimize the context effectively. By treating different token types based on their roles, the model updates some tokens while using others solely for optimization:

1. **Refrain from Updating Label Tokens**:  
   Some context tokens represent label tokens, which contain valuable, unmodifiable information. By excluding these tokens from updates during training, CPT ensures that the labels remain fixed, preserving their integrity.

2. **Apply Type-Specific Projection Norms**:  
   CPT employs Projected Gradient Descent (PGD) to update context embeddings, applying tailored norms to different context parts. This approach reduces overfitting while maintaining robustness and generalization by preserving the integrity of user-provided examples.



#### Limitations 
CPT is designed for few-shot scenarios, as concatenating more examples increases memory usage due to the self-attention mechanism and additional loss terms. For larger datasets, users can limit the number of context examples and use the remaining samples solely for optimization to manage memory efficiently.




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
