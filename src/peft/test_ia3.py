from peft import IA3Config, LoraConfig
from peft import TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm
if __name__ == "__main__":
    config = IA3Config(task_type=TaskType.SEQ_2_SEQ_LM,feedforward_modules=[])
    # config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=8, lora_alpha=32, lora_dropout=0.1)
    model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-large")
    model = get_peft_model(model, config)
    import pdb; pdb.set_trace()
    model.print_trainable_parameters()