import os
import torch
import mlflow
from getpass import getpass

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)

from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

print("\n==============================")
print(" QLoRA TRAINING CONFIGURATION")
print("==============================\n")

# --------------------------------
# PROMPTS
# --------------------------------

hf_token = getpass("Enter HuggingFace API Token (optional): ")

model_id = input(
    "Model to train [mistralai/Mistral-7B-Instruct-v0.2]: "
) or "mistralai/Mistral-7B-Instruct-v0.2"

dataset_path = input(
    "Dataset path [datasets/train.json]: "
) or "datasets/train.json"

epochs = int(input("Epochs [3]: ") or "3")
batch_size = int(input("Batch size [1]: ") or "1")
grad_accum = int(input("Gradient accumulation [8]: ") or "8")
learning_rate = float(input("Learning rate [2e-4]: ") or "2e-4")

output_dir = input("Checkpoint directory [checkpoints]: ") or "checkpoints"

# --------------------------------
# HF TOKEN
# --------------------------------

if hf_token:
    os.environ["HF_TOKEN"] = hf_token

# --------------------------------
# MLflow
# --------------------------------

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mistral-qlora")

# --------------------------------
# QLoRA config
# --------------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# --------------------------------
# TOKENIZER
# --------------------------------

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --------------------------------
# MODEL
# --------------------------------

print("Loading model in 4bit QLoRA...")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# --------------------------------
# LORA
# --------------------------------

print("Attaching LoRA adapters...")

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# --------------------------------
# DATASET
# --------------------------------

print("\nLoading dataset:", dataset_path)

dataset = load_dataset("json", data_files=dataset_path)

# --------------------------------
# TRAINING CONFIG
# --------------------------------

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
    num_train_epochs=epochs,
    learning_rate=learning_rate,
    bf16=True,
    logging_steps=10,
    save_steps=200,
    save_total_limit=3,
    report_to="none"
)

# --------------------------------
# TRAINER
# --------------------------------

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    args=training_args,
    processing_class=tokenizer
)

# --------------------------------
# TRAIN
# --------------------------------

print("\nStarting training...\n")

with mlflow.start_run():

    mlflow.log_param("model", model_id)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("grad_accum", grad_accum)
    mlflow.log_param("learning_rate", learning_rate)

    trainer.train()

    trainer.save_model(output_dir)

print("\nTraining complete.")
print("Model saved to:", output_dir)
