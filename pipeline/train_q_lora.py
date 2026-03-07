"""Real QLoRA training implementation with HuggingFace model download.

This script implements actual QLoRA training using transformers + bitsandbytes + peft.
Configuration is read from environment variables or config.json.
"""
import os
import json
import logging
import torch
from pathlib import Path
from typing import Optional, Dict, Any

# ML dependencies
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        TrainingArguments,
        DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer
    from datasets import load_dataset, Dataset
    import mlflow
    ML_DEPENDENCIES = True
except ImportError as e:
    logging.warning(f"Missing ML dependencies: {e}")
    ML_DEPENDENCIES = False

logging.basicConfig(level=logging.INFO)

def load_config() -> Dict[str, Any]:
    """Load configuration from config.json or environment."""
    config_file = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    
    # Fallback to environment variables
    return {
        'base_model': os.environ.get('BASE_MODEL', 'mistralai/Mistral-7B-Instruct-v0.2'),
        'hf_token': os.environ.get('HF_TOKEN', ''),
        'batch_size': int(os.environ.get('BATCH_SIZE', '1')),
        'learning_rate': float(os.environ.get('LEARNING_RATE', '2e-4')),
        'train_steps': int(os.environ.get('TRAIN_STEPS', '500')),
        'r': int(os.environ.get('QLORA_R', '64')),
        'alpha': int(os.environ.get('QLORA_ALPHA', '16')),
        'dropout': float(os.environ.get('QLORA_DROPOUT', '0.05')),
        'enable_mlflow': os.environ.get('ENABLE_MLFLOW', 'true').lower() == 'true'
    }

def setup_huggingface_token(token: str):
    """Setup HuggingFace token for model access."""
    if token:
        os.environ['HF_TOKEN'] = token
        from huggingface_hub import login
        try:
            login(token=token)
            logging.info("HuggingFace login successful")
        except Exception as e:
            logging.error(f"HuggingFace login failed: {e}")
            return False
    return True

def load_model_and_tokenizer(model_name: str, hf_token: str = None):
    """Load model and tokenizer with QLoRA configuration."""
    if not ML_DEPENDENCIES:
        raise ImportError("ML dependencies not available")
    
    # Setup token
    setup_huggingface_token(hf_token)
    
    logging.info(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token if hf_token else None,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logging.info(f"Loading model {model_name} with 4-bit quantization...")
    
    # QLoRA configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        token=hf_token if hf_token else None,
        device_map="auto",
        trust_remote_code=True
    )
    
    return model, tokenizer

def setup_lora(model, config: Dict[str, Any]):
    """Setup LoRA configuration and apply to model."""
    if not ML_DEPENDENCIES:
        raise ImportError("ML dependencies not available")
    
    lora_config = LoraConfig(
        r=config['r'],
        lora_alpha=config['alpha'],
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=config['dropout'],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def load_training_data(train_file: str, tokenizer):
    """Load and prepare training dataset."""
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    
    try:
        dataset = load_dataset('json', data_files=train_file, split='train')
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        raise
    
    def format_example(example):
        """Format training example for instruction following."""
        if 'instruction' in example and 'response' in example:
            context = example.get('context', '')
            if context:
                text = f"### Context:\n{context}\n\n### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
            else:
                text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
        else:
            # Fallback for different format
            text = str(example)
        
        return {'text': text}
    
    dataset = dataset.map(format_example)
    return dataset

def train_model(model, tokenizer, dataset, config: Dict[str, Any], output_dir: str):
    """Train the model using SFTTrainer."""
    if not ML_DEPENDENCIES:
        raise ImportError("ML dependencies not available")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config['batch_size'],
        gradient_accumulation_steps=max(1, 8 // config['batch_size']),
        learning_rate=config['learning_rate'],
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        max_steps=config['train_steps'],
        bf16=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="mlflow" if config.get('enable_mlflow') else "none",
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        max_seq_length=2048,
        dataset_text_field="text",
        packing=False,
    )
    
    logging.info("Starting training...")
    trainer.train()
    
    # Save model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logging.info(f"Training completed. Model saved to {output_dir}")
    return trainer

def real_train(train_file: str, output_dir: str):
    """Main training function."""
    if not ML_DEPENDENCIES:
        logging.error("ML dependencies not available. Please install: torch transformers peft trl accelerate bitsandbytes datasets")
        return False
    
    try:
        # Load configuration
        config = load_config()
        logging.info(f"Training with config: {config}")
        
        # Setup MLflow if enabled
        if config.get('enable_mlflow'):
            mlflow.set_experiment("qlora-continuous-training")
            mlflow.start_run(run_name=f"train-{Path(output_dir).name}")
            mlflow.log_params(config)
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(
            config['base_model'], 
            config.get('hf_token')
        )
        
        # Setup LoRA
        model = setup_lora(model, config)
        
        # Load data
        dataset = load_training_data(train_file, tokenizer)
        logging.info(f"Loaded {len(dataset)} training examples")
        
        # Train
        trainer = train_model(model, tokenizer, dataset, config, output_dir)
        
        # Log final metrics
        if config.get('enable_mlflow'):
            final_loss = trainer.state.log_history[-1].get('train_loss', 0)
            mlflow.log_metric("final_train_loss", final_loss)
            mlflow.log_artifacts(output_dir, artifact_path="model")
            mlflow.end_run()
        
        return True
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        if config.get('enable_mlflow'):
            mlflow.end_run()
        return False

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python train_q_lora.py <train_file> <output_dir>")
        sys.exit(1)
    
    train_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    success = real_train(train_file, output_dir)
    sys.exit(0 if success else 1)
