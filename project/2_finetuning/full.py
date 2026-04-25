import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from transformers import Trainer

MAX_SEQ_LENGTH = 512
CHECKPOINT_PATH = "../../runs/fft-cspubsum"

# 1. Load dataset
df = pd.read_csv("../1_preprocessing/cleaned/train_CSPubSum.csv")
assert "title" in df.columns and "text" in df.columns
dataset = Dataset.from_pandas(df)


# 3. Load model + tokenizer
model_name = "HuggingFaceTB/SmolLM-135M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(model_name)

def tokenize(example):
    prompt = f"{example['text']}\n\nTitle:"
    target = f" {example['title']}{tokenizer.eos_token}"

    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids

    # truncate
    input_ids = input_ids[:MAX_SEQ_LENGTH]
    labels = labels[:MAX_SEQ_LENGTH]

    attention_mask = [1] * len(input_ids)

    # padding
    pad_length = MAX_SEQ_LENGTH - len(input_ids)
    if pad_length > 0:
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_length
        attention_mask = attention_mask + [0] * pad_length
        labels = labels + [-100] * pad_length

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

dataset = dataset.map(tokenize, remove_columns=dataset.column_names)


# 6. Training arguments
training_args = TrainingArguments(
    output_dir=CHECKPOINT_PATH,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    learning_rate=4e-5,
    logging_steps=250,
    save_steps=500,
    save_total_limit=3,

    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    
    fp16=torch.cuda.is_available(),
    report_to="none"
)

trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)


# 8. Train

resume_flag = CHECKPOINT_PATH if os.path.exists(CHECKPOINT_PATH) and os.listdir(CHECKPOINT_PATH) else None
trainer.train(resume_from_checkpoint=resume_flag)

trainer.save_model(CHECKPOINT_PATH)
tokenizer.save_pretrained(CHECKPOINT_PATH)