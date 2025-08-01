import argparse
import json
import os
import torch
import numpy as np
import random
import logging
import torch.distributed as dist
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from datasets import Dataset
from sklearn.model_selection import train_test_split

import sys
# 定义PEFT库的路径（修改为你的实际路径）
peft_path = "/data1/lihanzuo/peft_memorization_test/peft/src/"  # <-- 替换为实际路径
# 规范化路径（处理斜杠、符号链接等）
peft_path = os.path.abspath(peft_path)
# 检查路径是否已在sys.path中，避免重复添加
if peft_path in sys.path:
    sys.path.remove(peft_path)
sys.path.append(peft_path)
print(f"已添加PEFT路径：{peft_path}")

try:
    import peft
    from peft import (
        LoraConfig,
        PromptTuningConfig,
        BottleneckConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        set_peft_model_state_dict,
    )
except ImportError as e:
    print(f"导入peft失败：{e}")
    print("请检查：")
    print(f" PEFT路径是否正确：{peft_path}")
import wandb


from qwen2_custom import CustomQwen2ForCausalLM

class LoggingCallback(TrainerCallback):
    """自定义回调类，用于记录训练过程中的所有指标"""
    def __init__(self, logger, local_rank):
        self.logger = logger
        self.local_rank = local_rank
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        # 只在主进程记录日志
        if self.local_rank in [-1, 0]:
            _ = logs.pop("total_flos", None)  # 移除不必要的指标
            self.logger.info(f"Step {state.global_step}: {json.dumps(logs, indent=2)}")


def setup_logging(seed, output_dir):
    """配置日志记录"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 创建日志目录
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    
    # 创建文件处理器
    log_file = os.path.join(output_dir, "logs", f"log_{seed}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器并添加到处理器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def set_seed(seed):
    """设置所有随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 固定torch卷积算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"


# class AdapterSaveCallback(TrainerCallback):
#     """自定义回调类，只保存适配器权重"""
#     def __init__(self, logger):
#         self.logger = logger
        
#     def on_save(self, args, state, control, **kwargs):
#         # 只在主进程执行保存
#         if state.is_world_process_zero:
#             checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            
#             # 获取模型（处理可能的分布式封装）
#             model = kwargs.get('model')
#             if model is None:
#                 return
                
#             # 解包分布式模型
#             if hasattr(model, "module"):
#                 model_to_save = model.module
#             else:
#                 model_to_save = model
                
#             # 确保只保存适配器权重
#             if hasattr(model_to_save, "save_pretrained"):
#                 # 创建检查点目录
#                 os.makedirs(checkpoint_folder, exist_ok=True)
                
#                 # 保存适配器权重
#                 model_to_save.save_pretrained(checkpoint_folder)
                
#                 # 保存tokenizer
#                 tokenizer = kwargs.get("tokenizer")
#                 if tokenizer is not None:
#                     tokenizer.save_pretrained(checkpoint_folder)
                    
#                 self.logger.info(f"Saved adapter checkpoint to {checkpoint_folder}")
        
#         # 阻止Trainer保存完整模型权重
#         control.should_save = False
#         return control


def main():
    parser = argparse.ArgumentParser(description="Qwen LoRA微调脚本")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen-32B", help="预训练模型名称或路径")
    parser.add_argument("--data_path", type=str, required=True, help="训练数据路径 (JSON格式)")
    parser.add_argument("--output_dir", type=str, default="./qwen_lora_finetuned", help="输出目录")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="每设备训练批次大小")
    parser.add_argument("--per_device_eval_batch_size",type=int, default=1, help="每设备验证批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--eval_ratio", type=float, default=0.2, help="评估集比例")
    parser.add_argument("--eval_and_save_steps", type=int, default=100, help="验证和保存的步数节")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA秩大小")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha参数")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str,
                        default="q_proj,k_proj,v_proj", help="应用LoRA的模块")
    parser.add_argument("--use_4bit", action="store_true", default=False, help="使用4位量化")
    parser.add_argument("--use_8bit", action="store_true", default=False, help="使用8位量化")
    parser.add_argument("--bf16", action="store_true", default=False, help="使用bfloat16精度")
    parser.add_argument("--fp16", action="store_true", default=False, help="使用float16精度")
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed配置文件路径")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B项目名称")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="从检查点恢复训练")
    parser.add_argument("--local_rank", type=int, default=-1, help="用于分布式训练的本地进程排名，-1表示自动获取")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--tuning_method", type=str, required=True, help="微调方法选择，从LoRA，PrefixTuning，Bottleneck，DoRA中选择")
    parser.add_argument("--num_virtual_tokens", type=int, default=20, help="prefixtuning的超参数配置")
    # bottleneck
    parser.add_argument("--bottleneck_size", type=int, default=128, help="bottleneck size")
    parser.add_argument("--target_modules", type=str, default="up_proj,down_proj", help="bottleneck插入的位置")
    # DoRA特有参数
    parser.add_argument("--use_dora", action="store_true", default=False, help="启用DoRA（仅在tuning_method=LoRA时有效）")
    args = parser.parse_args()

    # 从环境变量获取local_rank（优先于命令行参数）
    if "LOCAL_RANK" in os.environ and args.local_rank == -1:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    # 设置随机种子
    set_seed(args.seed)
    
    # 设置日志
    logger = setup_logging(args.seed, args.output_dir)
    
    # 在主进程记录所有命令行参数
    if args.local_rank in [-1, 0]:
        logger.info("===== 命令行参数 =====")
        for arg, value in sorted(vars(args).items()):
            logger.info(f"{arg}: {value}")
    
    # 初始化分布式环境
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        logger.info(f"进程 local_rank: {args.local_rank}, 物理 GPU: {torch.cuda.current_device()}")
    
    # 初始化W&B
    if args.local_rank in [-1, 0] and args.wandb_project:
        wandb.init(project=args.wandb_project, config=vars(args))

    # 量化配置 - 修复量化问题
    bnb_config = None
    if args.use_4bit:
        logger.info(f"进程 {args.local_rank}: 使用4bit量化")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16 if args.bf16 else torch.float16
        )
    elif args.use_8bit:
        logger.info(f"进程 {args.local_rank}: 使用8bit量化")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )

    # 初始化tokenizer
    logger.info(f"进程 {args.local_rank}: 初始化tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="left",
        use_fast=False
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载并预处理数据
    logger.info(f"进程 {args.local_rank}: 加载数据: {args.data_path}")
    with open(args.data_path, "r") as f:
        data = json.load(f)

    def format_example(example):
        instruction = example["instruction"]
        input_text = example.get("input", "")
        output_text = example["output"]

        # 构建模型输入格式
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        return {"prompt": prompt, "response": output_text}

    formatted_data = [format_example(ex) for ex in data]
    dataset = Dataset.from_list(formatted_data)
    
    # 使用sklearn分割训练集和评估集（所有进程使用相同种子，结果一致）
    logger.info(f"进程 {args.local_rank}: 分割数据集，评估集比例: {args.eval_ratio}, 随机种子: {args.seed}")
    train_data, eval_data = train_test_split(
        formatted_data,
        test_size=args.eval_ratio,
        random_state=args.seed
    )
    
    # 转换为Dataset对象
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
    logger.info(f"进程 {args.local_rank}: 训练集大小: {len(train_dataset)}, 评估集大小: {len(eval_dataset)}")

    # 数据编码函数 - 确保在CPU上进行
    def tokenize_function(examples):
        texts = [p + r + tokenizer.eos_token for p, r in zip(examples["prompt"], examples["response"])]
        tokenized = tokenizer(
            texts,
            max_length=args.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # 设置labels：prompt部分为-100，response部分为实际token
        labels = tokenized["input_ids"].clone()
        for i, text in enumerate(texts):
            prompt_len = len(tokenizer(
                examples["prompt"][i],
                truncation=True,
                max_length=args.max_seq_length,
                return_tensors="pt"
            )["input_ids"][0])

            # 设置prompt部分的labels为-100
            labels[i, :prompt_len] = -100

        tokenized["labels"] = labels
        return tokenized

    # 处理训练集
    num_proc = 4 if args.local_rank in [-1, 0] else 1
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["prompt", "response"],
        num_proc=num_proc
    )
    
    # 处理评估集
    tokenized_eval = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["prompt", "response"],
        num_proc=num_proc
    )

    # 加载模型 - 使用DeepSpeed自动设备映射
    logger.info(f"进程 {args.local_rank}: 加载模型...")
    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    if "qwen" in args.model_name.lower() and args.tuning_method == "PromptTuning":
        logger.info(f"使用CustomQwen")
        model = CustomQwen2ForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto" if args.local_rank == -1 else None  # 分布式训练时不使用device_map
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto" if args.local_rank == -1 else None  # 分布式训练时不使用device_map
        )
    
    # 模型配置禁用缓存
    model.config.use_cache = False

    # 准备模型用于量化训练
    if args.use_4bit or args.use_8bit:
        logger.info(f"进程 {args.local_rank}: 准备模型用于量化训练...")
        model = prepare_model_for_kbit_training(model)

    print("==========original============\n",model)
    # 更新LoRA配置部分，增加DoRA支持
    if args.tuning_method == "LoRA" or args.tuning_method == "DoRA":
        logger.info(f"进程 {args.local_rank}: 配置{'DoRA' if args.use_dora else 'LoRA'}...")
        lora_target_modules = args.lora_target_modules.split(",")

        tuning_config = LoraConfig(
            peft_type="LORA",
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=lora_target_modules,
            task_type="CAUSAL_LM",
            bias="none",
            use_dora=args.use_dora or args.tuning_method == "DoRA"  # 如果使用DoRA或明确指定use_dora，则启用DoRA
        )

    elif args.tuning_method == "PromptTuning":
        logger.info(f"进程 {args.local_rank}: PromptTuning...")
        tuning_config = PromptTuningConfig(
            peft_type="PROMPT_TUNING",
            num_virtual_tokens=args.num_virtual_tokens,
            token_dim = model.config.hidden_size,
            task_type="CAUSAL_LM"
        )
    elif args.tuning_method == "Bottleneck":
        logger.info(f"进程 {args.local_rank}: Adapter tuning...")
        target_modules = args.target_modules.split(",")

        tuning_config = BottleneckConfig(
            peft_type="BOTTLENECK",
            bottleneck_size=args.bottleneck_size,
            non_linearity="tanh",
            adapter_dropout=0.05,
            use_adapterp=False,
            target_modules=target_modules,
            scaling=1.0,
            bias="none",
            task_type="CAUSAL_LM"

        )
    model = get_peft_model(model, tuning_config)
    print("==========peft============\n",model)
    if args.local_rank in [-1, 0]:
        model.print_trainable_parameters()
    

    # 如果从检查点恢复: 在某个已有adapter_model基础上继续
    if args.resume_from_checkpoint:
        logger.info(f"进程 {args.local_rank}: 从检查点恢复: {args.resume_from_checkpoint}")
        checkpoint_path = os.path.join(args.resume_from_checkpoint, "adapter_model.bin")
        adapters_weights = torch.load(checkpoint_path, map_location="cpu")
        set_peft_model_state_dict(model, adapters_weights)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        weight_decay=0.01,
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=args.eval_and_save_steps,
        save_steps=args.eval_and_save_steps,
        save_total_limit=2,
        deepspeed=args.deepspeed,
        report_to="wandb" if args.wandb_project else "none",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,  # 必须禁用
        local_rank=args.local_rank,
        seed=args.seed,  # 设置训练参数中的随机种子
    )
    
    # 在主进程记录所有TrainingArguments参数
    if args.local_rank in [-1, 0]:
        logger.info("===== Trainer训练参数 =====")
        for arg, value in sorted(training_args.to_dict().items()):
            logger.info(f"{arg}: {value}")
    

    # 数据整理器 - 保持在CPU上
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )

    # 初始化Trainer，添加自定义日志回调和适配器保存回调
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[
            LoggingCallback(logger, args.local_rank),
            # AdapterSaveCallback(logger)  # 使用自定义适配器保存回调
        ]
    )
    

    # 开始训练
    logger.info(f"进程 {args.local_rank}: 开始训练...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    

    # 保存最终适配器 - 使用回调函数相同的逻辑
    if args.local_rank in [-1, 0]:
        logger.info(f"进程 {args.local_rank}: 保存最终适配器...")
        
        # 处理分布式模型
        if hasattr(model, "module"):
            model_to_save = model.module
        else:
            model_to_save = model
            
        # 保存适配器权重
        model_to_save.save_pretrained(args.output_dir)
        
        # 保存tokenizer
        tokenizer.save_pretrained(args.output_dir)
        
        # 保存训练配置
        with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

        logger.info(f"进程 {args.local_rank}: 训练完成! 适配器保存至: {args.output_dir}")


if __name__ == "__main__":
    main()