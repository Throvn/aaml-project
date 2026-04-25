import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model

# 1. Load dataset
df = pd.read_csv("../preprocessing/cleaned/train_data.csv")
assert "title" in df.columns and "text" in df.columns
dataset = Dataset.from_pandas(df)

# 3. Load model + tokenizer
model_name = "HuggingFaceTB/SmolLM-135M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. Prompt formatting
def format_example(example):
    return {
        "text": f"{example['text']}\n\nTitle: {example['title']}{tokenizer.eos_token}",
        "title": example["title"],
    }
dataset = dataset.map(format_example)

# 4. Apply LoRA (paper-style)
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    # target_modules=['q_proj', 'v_proj', 'o_proj',
    #                  'up_proj', 'down_proj']
)

model = get_peft_model(model, lora_config)

# 6. Training arguments
training_args = TrainingArguments(
    output_dir="../../runs/lora-own",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    learning_rate=4e-5,
    logging_steps=100,
    save_steps=200,
    save_total_limit=2,
    
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# 7. Trainer
dataset = dataset.train_test_split(test_size=0.01, seed=42)
train = dataset["train"]

# We need to mask the Title for our validation dataset,
# as we want to generate the title ourselves.
validate = dataset["test"]

trainer = SFTTrainer(
    model=model,
    train_dataset=train,
    args=training_args,
    dataset_text_field='text',
    packing=False,
)

# 8. Train
trainer.train(resume_from_checkpoint=True)

trainer.save_model("../../runs/lora-own")
tokenizer.save_pretrained("../../runs/lora-own")