"""

Implementing LoRA+ https://arxiv.org/abs/2402.12354


@dataclass
class LoraPlusTrainingArguments(TrainingArguments):
    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    do_eval: bool = field(
        default=True, metadata={"help": "Whether to run eval on the dev set."}
    )
    keep_checkpoints: str = field(
        default="all",
        metadata={"help": "keep all, eval, or none checkpoints after end of training"},
    )
    lora_rank: int = field(default=8, metadata={"help": "LoRA rank r"})
    lora_alpha: float = field(default=16, metadata={"help": "LoRA alpha parameter"})
    lora_dropout: float = field(
        default=0.1, metadata={"help": "dropout rate for LoRA modules"}
    )
    target_modules: Optional[str] = field(
        default=None, metadata={"help": "which modules to add LoRA layer to"}
    )
    use_lora: bool = field(
        default=True, metadata={"help": "whether to finetune using LoRA"}
    )
    lora_use_original_init: bool = field(
        default=False,
        metadata={"help": "whether to use the original LoRA initialization"},
    )
    bf16: bool = field(default=False, metadata={"help": "use bfloat16"})
    fp16: bool = field(default=False, metadata={"help": "use bfloat16"})
    gradient_checkpointing: bool = field(
        default=False, metadata={"help": "use gradient checkpointing"}
    )
    loraplus_lr_ratio: Optional[float] = field(
        default=None, metadata={"help": "loraplus learning rate ratio lr_B / lr_A."}
    )
    loraplus_lr_embedding: Optional[float] = field(
        default=1e-6, metadata={"help": "loraplus learning rate for lora embedding layers."}
    )


def get_module(name, opt_model):
    parent_idx = 2 if "lora" in name else 1
    module_names = name.split(sep=".")[:-parent_idx]
    module = reduce(getattr, module_names, opt_model)
    return module

def create_loraplus_optimizer(
    opt_model,
    optimizer_cls,
    optimizer_kwargs,
    loraplus_lr_ratio,
    loraplus_lr_embedding=None,
):

    assert loraplus_lr_ratio is not None, "loraplus_lr_ratio must be provided."

    if loraplus_lr_embedding is None:
        loraplus_lr_embedding = 1e-6

    decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    param_groups = {
        "groupA": {},
        "groupB": {},
        "groupB_no_decay": {},
        "embedding": {},
    }

    for name, param in opt_model.named_parameters():
        if not param.requires_grad:
            continue

        module = get_module(name, opt_model)
        if isinstance(module, lora.Embedding):
            param_groups["embedding"][name] = param
        elif "lora_B" in name or param.ndim == 1:
            if name in decay_parameters:
                param_groups["groupB"][name] = param
            else:
                param_groups["groupB_no_decay"][name] = param
        else:
            param_groups["groupA"][name] = param

    assigned_param_groups = ""
    for group in param_groups:
        assigned_param_groups += f"{group}\n {list(param_groups[group].keys())}\n\n"
    logger.debug(assigned_param_groups)

    lr = optimizer_kwargs["lr"]
    weight_decay = optimizer_kwargs.get("weight_decay", 0.0)

    optimizer_grouped_parameters = [
        {
            "params": list(param_groups["groupA"].values()),
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": list(param_groups["embedding"].values()),
            "weight_decay": weight_decay,
            "lr": loraplus_lr_embedding,
        },
        {
            "params": list(param_groups["groupB"].values()),
            "weight_decay": weight_decay,
            "lr": lr * loraplus_lr_ratio,
        },
        {
            "params": list(param_groups["groupB_no_decay"].values()),
            "weight_decay": 0.0,
            "lr": lr * loraplus_lr_ratio,
        },
    ]

    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    if optimizer_cls.__name__ == "Adam8bit":
        import bitsandbytes

        manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

        skipped = 0
        for module in opt_model.modules():
            if isinstance(module, nn.Embedding):
                skipped += sum(
                    {p.data_ptr(): p.numel() for p in module.parameters()}.values()
                )
                logger.info(f"skipped {module}: {skipped/2**20}M params")
                manager.register_module_override(module, "weight", {"optim_bits": 32})
                logger.debug(f"bitsandbytes: will optimize {module} in fp32")
        logger.info(f"skipped: {skipped/2**20}M params")

    return optimizer

class LoraPlusTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: LoraPlusTrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        assert isinstance(args, LoraPlusTrainingArguments), "args must be of type LoraPlusTrainingArguments"
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )

    def create_optimizer(self):
        if self.args.loraplus_lr_ratio is None:
            return super().create_optimizer()

        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            loraplus_lr_ratio = getattr(self.args, 'loraplus_lr_ratio', None)
            loraplus_lr_embedding = getattr(self.args, 'loraplus_lr_embedding', None)
            self.optimizer = create_loraplus_optimizer(
                opt_model,
                optimizer_cls,
                optimizer_kwargs,
                loraplus_lr_ratio,
                loraplus_lr_embedding,
            )

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
"""

from peft.import_utils import is_bnb_4bit_available, is_bnb_available

from ..lora.config import LoftQConfig, LoraConfig
from ..lora.gptq import QuantLinear
from ..lora.layer import Conv2d, Embedding, Linear
from ..lora.layer import LoraLayer as LoraPlusLayer
from ..lora.model import LoraModel


__all__ = ["LoraConfig", "LoftQConfig", "Conv2d", "Embedding", "LoraPlusLayer", "Linear", "LoraModel", "QuantLinear"]


def __getattr__(name):
    if (name == "Linear8bitLt") and is_bnb_available():
        from ..lora.bnb import Linear8bitLt

        return Linear8bitLt

    if (name == "Linear4bit") and is_bnb_4bit_available():
        from ..lora.bnb import Linear4bit

        return Linear4bit

    raise AttributeError(f"module {__name__} has no attribute {name}")
