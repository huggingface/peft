<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<h1 align="center"> <p>🤗 PEFT</p></h1>
<h3 align="center">
    <p>前沿参数高效微调（Parameter-Efficient Fine-Tuning）方法库</p>
</h3>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

微调超大规模预训练模型通常因其庞大的参数量而成本高昂。**参数高效微调（PEFT）** 方法通过仅微调少量额外（新增）的模型参数，而非更新全部预训练权重，从而能够以极高效率将大型模型适配到各类下游任务中。这极大降低了显存算力与存储成本。最新前沿 PEFT 技术在保持极低开销的同时，可取得与全量微调（Full Fine-Tuning）相媲美的出色性能。

🤗 PEFT 深度无缝集成了 Hugging Face 核心生态：
- **Transformers**：开箱即用的轻量模型微调与高效推理；
- **Diffusers**：便捷管理与组合多种文生图/扩散模型适配器；
- **Accelerate**：针对超大模型的分布式并行训练与异构算力调度。

> [!TIP]
> 欢迎访问官方 [PEFT 机构主页](https://huggingface.co/PEFT)，查阅库内已实现的完整 PEFT 算法族与各类下游任务实战 Jupyter Notebook 示例。点击主页上的 “Watch repos” 按钮，即可第一时间接收新方法与新教程的发布通知！

更多技术实现细节，请查阅 PEFT 适配器 API 参考文档，以及 [Adapters 适配器](https://huggingface.co/docs/peft/en/conceptual_guides/adapter)、[Soft prompts 软提示](https://huggingface.co/docs/peft/en/conceptual_guides/prompting) 和 [IA3](https://huggingface.co/docs/peft/en/conceptual_guides/ia3) 等核心概念指南。

---

## 🚀 快速上手 (Quickstart)

通过 pip 直接安装 PEFT：

```bash
pip install peft
```

### 1. 模型训练准备
以 **LoRA** 为例，只需使用 `get_peft_model` 包装基础模型与 PEFT 配置，即可快速就绪。对于 Qwen2.5-3B 模型，您只需要微调约 **0.12%** 的参数量！

```python
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model

device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
model_id = "Qwen/Qwen2.5-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    task_type=TaskType.CAUSAL_LM,
    # target_modules=["q_proj", "v_proj", ...]  # 可按需指定注入的目标注意力模块
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# 输出打印：trainable params: 3,686,400 || all params: 3,089,625,088 || trainable%: 0.1193

# 随后在您的数据集上进行训练（例如搭配 transformers Trainer），训练完成后保存适配器权重：
model.save_pretrained("qwen2.5-3b-lora")
```

### 2. 适配器加载与推理
加载已微调的 PEFT 适配器进行模型推理：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
model_id = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device)
# 在基础模型之上挂载微调好的 LoRA 权重
model = PeftModel.from_pretrained(model, "qwen2.5-3b-lora")

inputs = tokenizer("Preheat the oven to 350 degrees and place the cookie dough", return_tensors="pt")
outputs = model.generate(**inputs.to(device), max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# 输出类似内容: Preheat the oven to 350 degrees and place the cookie dough in a baking dish [...]
```

---

## 💡 为什么选择 PEFT？(Why you should use PEFT)

PEFT 最核心的优势在于**极致削减算力与显存/磁盘开销**，让超大模型在丰富多样的边缘与消费级硬件上真正落地。

### 1. 消费级硬件上的卓越表现
在拥有 A100 80GB GPU 与 64GB+ 内存的环境下微调以下模型时的显存占用对比（测试数据集：`twitter_complaints`）：

| 模型 | 全量微调 (Full FT) | PEFT-LoRA (PyTorch) | PEFT-LoRA (DeepSpeed + CPU Offload) |
| :--- | :--- | :--- | :--- |
| **bigscience/T0_3B** (3B 参数) | 47.14GB GPU / 2.96GB CPU | 14.4GB GPU / 2.96GB CPU | **9.8GB GPU** / 17.8GB CPU |
| **bigscience/mt0-xxl** (12B 参数) | 💥 显存溢出 (OOM) | 56GB GPU / 3GB CPU | **22GB GPU** / 52GB CPU |
| **bigscience/bloomz-7b1** (7B 参数) | 💥 显存溢出 (OOM) | 32GB GPU / 3.8GB CPU | **18.1GB GPU** / 35GB CPU |

借助 LoRA，您可以在单张 80GB GPU 上轻松训练原本会 OOM 的 12B 大模型，也能在更小显存的设备上轻松微调 3B-7B 模型。在下游任务准确率上，PEFT 展现出媲美全量微调的高水准：

| 评估基准 | 准确率 (Accuracy) |
| :--- | :---: |
| **人类基准线 (众包测试)** | 0.897 |
| **Flan-T5 全量微调** | 0.892 |
| **lora-t0-3b** | **0.863** |

> [!TIP]
> 上表中 `bigscience/T0_3B` 的微调性能尚未达到极致，通过调节指令模板格式、LoRA 超参数（如秩 `r` 与缩放因子 `alpha`），还能进一步榨取更强性能。该模型的最终 Checkpoint 权重体积**仅有 19MB**（而原始全量模型高达 11GB）！更多实战分析请参阅这篇 [技术博文](https://www.philschmid.de/fine-tune-flan-t5-peft)。

### 2. 量化技术深度结合 (Quantization & QLoRA)
量化能够通过将权重转换为更低位宽（如 8-bit、4-bit）来显著降低显存门槛。PEFT 可与量化技术无缝融合，让消费级单卡训练超大 LLM 成为现实：
* **QLoRA 实战**：在 16GB 消费级显卡上使用 QLoRA 与 TRL 微调 Llama-2-7B，详见 PyTorch 官方博客 [消费级硬件大模型微调指南](https://pytorch.org/blog/finetune-llms/)。
* **语音模型微调**：通过 8-bit 量化搭配 LoRA 微调 Whisper-large-v2 多语种语音识别，详见 [Jupyter 实战教程](https://colab.research.google.com/drive/1DOkD_5OUjFa0r5Ik3SgywJLJtEo2qLxO?usp=sharing)。

### 3. 极大节省存储与杜绝灾难性遗忘
在传统全量微调流程中，每个下游业务任务都需要完整备份一份数十 GB 的模型文件；而使用 PEFT，每个任务沉淀下来的适配器 Checkpoint 仅有**几兆字节（MB）**。更关键的是，冻结原始基座模型有效避免了对通用知识的“灾难性遗忘（Catastrophic Forgetting）”，让一个通用底座模型同时服务成百上千个定制化业务场景。

---

## 🌐 生态系统无缝集成 (PEFT Integrations)

### 🎨 Diffusers (文生图与扩散模型)
扩散模型的迭代去噪推理与反向传播极耗显存。PEFT 不仅大幅降低显存开销，还将最终微调 Checkpoint 缩小至极限。例如微调 Stable Diffusion v1-4，最终产物**仅需 8.8MB**！

| 模型 | 全量微调 | PEFT-LoRA | PEFT-LoRA + 梯度检查点 (Gradient Checkpointing) |
| :--- | :--- | :--- | :--- |
| **CompVis/stable-diffusion-v1-4** | 27.5GB GPU / 3.97GB CPU | 15.5GB GPU / 3.84GB CPU | **8.12GB GPU** / 3.77GB CPU |

> [!TIP]
> 体验训练脚本 [examples/lora_dreambooth/train_dreambooth.py](examples/lora_dreambooth/train_dreambooth.py) 使用 LoRA 定制个性化 Stable Diffusion，也可访问 Hugging Face Space 上的 [在线 DreamBooth Demo](https://huggingface.co/spaces/smangrul/peft-lora-sd-dreambooth) 进行试玩。更多技术指引详见 [Diffusers 集成教程](https://huggingface.co/docs/peft/main/en/tutorial/peft_integrations#diffusers)。

### 🤗 Transformers
PEFT 原生深度内置于 [Transformers](https://huggingface.co/docs/transformers/main/en/peft)。加载模型后，可通过原语进行多适配器无缝管理与热切换：

```python
from peft import LoraConfig
model = ...  # Transformers 基础模型
peft_config = LoraConfig(...)

# 1. 挂载新适配器
model.add_adapter(peft_config, adapter_name="lora_1")

# 2. 从本地或 Hugging Face Hub 加载已训练好的适配器
model.load_adapter("<path-to-adapter>", adapter_name="lora_2")

# 3. 零停机动态切换活跃适配器
model.set_adapter("lora_2")
```

### ⚡ Accelerate
[Accelerate](https://huggingface.co/docs/accelerate/index) 为多硬件平台（GPU、TPU、Apple Silicon 等）提供统一的分布式训练与推理接口。PEFT 原生即插即用兼容 Accelerate，让在资源受限的环境中训练大规模模型变得轻而易举。

### 🎯 TRL (强化学习与偏好对齐)
在 RLHF（人类反馈强化学习）或 DPO（直接偏好优化）中，微调奖励模型（Reward Model）与策略模型（Policy）：
* **DPO 对齐实战**：阅读 [使用 DPO 与 TRL 微调 Mistral-7B](https://towardsdatascience.com/fine-tune-a-mistral-7b-model-with-direct-preference-optimization-708042745aac)；
* **单卡 24GB 微调 20B 级模型**：阅读博客 [在消费级单卡上进行 20B LLM 强化学习微调](https://huggingface.co/blog/trl-peft)，并试玩 [情感分析实战 Notebook](https://github.com/huggingface/trl/blob/main/examples/notebooks/gpt2-sentiment.ipynb)；
* **StackLLaMA**：查阅 [LLaMA RLHF 实战指南](https://huggingface.co/blog/stackllama)，涵盖监督微调（SFT）、奖励建模与 RL 全流程脚本。

---

## 📋 模型架构支持 (Model Support)

欢迎访问 [PEFT 支持方法交互式空间](https://stevhliu-peft-methods.hf.space) 或查阅 [官方文档](https://huggingface.co/docs/peft/main/en/index)，了解官方开箱即用支持的模型架构清单。即使您的特定模型未在列表中显式列出，也可以通过自定义适配层快速启用，详见 [自定义架构集成指南](https://huggingface.co/docs/peft/main/en/developer_guides/custom_models#new-transformers-architectures)。

---

## 🤝 参与贡献 (Contribute)

如果您希望参与 PEFT 的开发演进，欢迎查阅官方 [贡献指南 (Contribution Guide)](https://huggingface.co/docs/peft/developer_guides/contributing)。

---

## 📑 论文与学术引用 (Citing 🤗 PEFT)

如果您在学术科研或工业应用中使用了 🤗 PEFT，请引用以下 BibTeX 条目：

```bibtex
@Misc{peft,
  title =        {{PEFT}: State-of-the-art Parameter-Efficient Fine-Tuning methods},
  author =       {Sourab Mangrulkar and Sylvain Gugger and Lysandre Debut and Younes Belkada and Sayak Paul and Benjamin Bossan and Marian Tietz},
  howpublished = {\url{https://github.com/huggingface/peft}},
  year =         {2022}
}
```

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年9月4日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
