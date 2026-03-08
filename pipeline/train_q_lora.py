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
import inspect

# Constants
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
AI_LAB = os.path.join(ROOT, 'data', 'ai-lab')

# ML dependencies
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        TrainingArguments,
        DataCollatorForLanguageModeling,
        TrainerCallback,
        EarlyStoppingCallback
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
        'grad_accum': int(os.environ.get('GRAD_ACCUM', '8')),
        'learning_rate': float(os.environ.get('LEARNING_RATE', '2e-4')),
        'train_steps': int(os.environ.get('TRAIN_STEPS', '500')),
        'eval_steps': int(os.environ.get('EVAL_STEPS', '50')),
        'save_steps': int(os.environ.get('SAVE_STEPS', '50')),
        'max_seq_length': int(os.environ.get('MAX_SEQ_LENGTH', '2048')),
        'r': int(os.environ.get('QLORA_R', '64')),
        'alpha': int(os.environ.get('QLORA_ALPHA', '16')),
        'dropout': float(os.environ.get('QLORA_DROPOUT', '0.05')),
        'enable_mlflow': os.environ.get('ENABLE_MLFLOW', 'true').lower() == 'true',
        'val_set_size': float(os.environ.get('VAL_SET_SIZE', '0.05'))
    }

def get_device_info():
    """Determine GPU capabilities and select optimal dtype."""
    if not torch.cuda.is_available():
        return "cpu", torch.float32, False
    
    gpu_name = torch.cuda.get_device_name(0)
    # A100, H100, L40, RTX 30/40 series support bf16
    # 2080 Ti (Turing) does NOT support bf16 efficiently
    supports_bf16 = torch.cuda.is_bf16_supported()
    
    # User's 2080 Ti is Turing (Compute 7.5), No native bfloat16
    # A100/H100 are Compute 8.0+
    if supports_bf16 and ("A100" in gpu_name or "H100" in gpu_name or "A800" in gpu_name):
        return gpu_name, torch.bfloat16, True
    else:
        return gpu_name, torch.float16, False

def preflight_checks(config: Dict[str, Any], train_file: str):
    """Verify system readiness before starting training."""
    logging.info("Running preflight checks...")
    
    # 1. Dataset existence
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Dataset not found at {train_file}")
    
    # 2. Dataset row count
    try:
        with open(train_file, 'r') as f:
            data = json.load(f)
            if not isinstance(data, list) or len(data) == 0:
                raise ValueError("Dataset is empty or not a list")
            logging.info(f"Verified dataset has {len(data)} samples")
    except Exception as e:
        raise ValueError(f"Failed to read dataset: {e}")

    # 3. GPU/VRAM check
    if not torch.cuda.is_available():
        logging.warning("No GPU detected! This will be extremely slow.")
    else:
        gpu_name, dtype, is_bf16 = get_device_info()
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logging.info(f"Found GPU: {gpu_name} with {vram_gb:.1f}GB VRAM")
        logging.info(f"Using dtype: {dtype} (bf16={is_bf16})")
        
        if vram_gb < 8:
            logging.warning("Low VRAM detected. Training might fail.")
            
    # 4. Disk space check
    import shutil
    usage = shutil.disk_usage(AI_LAB)
    free_gb = usage.free / 1e9
    if free_gb < 5:
        raise RuntimeError(f"Low disk space: {free_gb:.1f}GB remaining in {AI_LAB}")
    
    logging.info("Preflight checks passed.")

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
    
    logging.info(f"Loading model {model_name}...")

    gpu_name, compute_dtype, use_bf16 = get_device_info()
    
    try:
        logging.info(f"Attempting 4-bit quantized load with {compute_dtype}...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            token=hf_token if hf_token else None,
            device_map="auto",
            trust_remote_code=True
        )
        logging.info("4-bit quantized load successful")
    except Exception as e:
        logging.error(f"4-bit load failed: {e}")
        logging.error("Falling back to full non-quantized load is DISABLED to prevent VRAM explosion.")
        raise RuntimeError(f"Failed to load quantized model: {e}")
    
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

def load_training_data(train_file: str, tokenizer, config: Dict[str, Any]):
    """Load and prepare training dataset with validation split."""
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    
    try:
        dataset = load_dataset('json', data_files=train_file, split='train')
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        raise
    
    # Split into train/validation
    val_size = config.get('val_set_size', 0.05)
    if val_size > 0:
        dataset = dataset.train_test_split(test_size=val_size)
        train_data = dataset['train']
        eval_data = dataset['test']
    else:
        train_data = dataset
        eval_data = None

    def format_example(example):
        """Format training example for instruction following."""
        # Normalize whitespace and ensure it ends with EOS
        instruction = example.get('instruction', '').strip()
        response = example.get('response', '').strip()
        context = example.get('context', '').strip()

        if not instruction or not response:
            return {'text': ''} # Filtered later

        if context:
            text = f"### Context:\n{context}\n\n### Instruction:\n{instruction}\n\n### Response:\n{response}"
        else:
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        
        if not text.endswith(tokenizer.eos_token):
            text += tokenizer.eos_token
            
        return {'text': text}
    
    train_data = train_data.map(format_example)
    if eval_data:
        eval_data = eval_data.map(format_example)

    # Filter out broken samples
    train_data = train_data.filter(lambda x: len(x['text']) > 0)
    if eval_data:
        eval_data = eval_data.filter(lambda x: len(x['text']) > 0)

    return train_data, eval_data

def train_model(model, tokenizer, train_data, eval_data, config: Dict[str, Any], output_dir: str):
    """Train the model using SFTTrainer."""
    if not ML_DEPENDENCIES:
        raise ImportError("ML dependencies not available")
    
    # setup_lora(model, config) # REMOVED: Called in real_train once.

    _, _, use_bf16 = get_device_info()
    
    resume_from_checkpoint = os.environ.get('RESUME_FROM_CHECKPOINT', None)
    if resume_from_checkpoint == "True":
        resume_from_checkpoint = True
    elif resume_from_checkpoint == "False":
        resume_from_checkpoint = False

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['grad_accum'],
        learning_rate=config['learning_rate'],
        logging_steps=10,
        save_steps=config['save_steps'],
        eval_steps=config['eval_steps'],
        save_total_limit=3,
        max_steps=config['train_steps'],
        bf16=use_bf16,
        fp16=not use_bf16,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to=["tensorboard", "mlflow"] if config.get('enable_mlflow') else "tensorboard",
        logging_dir=os.path.join(output_dir, 'logs'), # Run-scoped
        remove_unused_columns=False,
        resume_from_checkpoint=resume_from_checkpoint,
        gradient_checkpointing=True,
        group_by_length=True,
        evaluation_strategy="steps" if eval_data else "no",
        load_best_model_at_end=True if eval_data else False,
        metric_for_best_model="eval_loss" if eval_data else None,
        greater_is_better=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    trainer = None
    try:
        params = set(inspect.signature(SFTTrainer.__init__).parameters.keys())
        logging.info(f"SFTTrainer accepts parameters: {params}")
    except Exception as e:
        logging.warning(f"Failed to inspect SFTTrainer: {e}")
        params = set()

    common_kwargs = {
        "model": model,
        "train_dataset": train_data,
        "eval_dataset": eval_data,
        "args": training_args,
        "data_collator": data_collator,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=3)] if eval_data else [],
    }

    # Only add parameters that are actually supported
    supported_params = set(inspect.signature(SFTTrainer.__init__).parameters.keys())
    
    # Handle tokenizer/processing_class parameter compatibility
    if "processing_class" in supported_params:
        common_kwargs["processing_class"] = tokenizer
        logging.info("Using processing_class parameter for tokenizer")
    elif "tokenizer" in supported_params:
        common_kwargs["tokenizer"] = tokenizer
        logging.info("Using tokenizer parameter")
    else:
        logging.warning("Neither tokenizer nor processing_class found in SFTTrainer parameters")

    # Add optional parameters only if supported
    optional_params = {
        "max_seq_length": 2048,
        "dataset_text_field": "text",
        "packing": False,
    }
    
    for param, value in optional_params.items():
        if param in supported_params:
            common_kwargs[param] = value
            logging.info(f"Using {param} parameter")
        else:
            logging.info(f"Skipping unsupported parameter: {param}")

    if "dataset_text_field" not in supported_params:
        try:
            cols = getattr(dataset, "column_names", None)
            if cols and "text" in cols:
                max_len = int(os.environ.get("MAX_SEQ_LENGTH", "2048"))

                def _tok(batch):
                    return tokenizer(
                        batch["text"],
                        truncation=True,
                        max_length=max_len,
                    )

                dataset = dataset.map(
                    _tok,
                    batched=True,
                    remove_columns=list(cols),
                    desc="Tokenizing dataset",
                )
                common_kwargs["train_dataset"] = dataset
        except Exception:
            logging.exception("Failed to tokenize dataset")

    try:
        trainer = SFTTrainer(**common_kwargs)
        logging.info("SFTTrainer created successfully")
    except TypeError as e:
        logging.error(f"Failed to create SFTTrainer: {e}")
        # Try with minimal parameters
        minimal_kwargs = {
            "model": model,
            "train_dataset": train_data,
            "eval_dataset": eval_data,
            "args": training_args,
        }
        if "processing_class" in supported_params:
            minimal_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in supported_params:
            minimal_kwargs["tokenizer"] = tokenizer
        
        try:
            trainer = SFTTrainer(**minimal_kwargs)
            logging.info("SFTTrainer created with minimal parameters")
        except TypeError as e2:
            logging.error(f"Failed to create SFTTrainer with minimal parameters: {e2}")
            raise e2
    
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
        
        # Preflight checks
        preflight_checks(config, train_file)
        
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
        train_data, eval_data = load_training_data(train_file, tokenizer, config)
        logging.info(f"Loaded {len(train_data)} training and {len(eval_data) if eval_data else 0} eval examples")
        
        # Train
        trainer = train_model(model, tokenizer, train_data, eval_data, config, output_dir)
        
        # Save run metadata
        gpu_name, dtype, _ = get_device_info()
        metadata = {
            "base_model": config['base_model'],
            "dataset_path": train_file,
            "sample_count": len(train_data),
            "train_steps": config['train_steps'],
            "learning_rate": config['learning_rate'],
            "lora_r": config['r'],
            "lora_alpha": config['alpha'],
            "gpu_name": gpu_name,
            "dtype": str(dtype),
            "final_loss": trainer.state.log_history[-1].get('train_loss', 0) if trainer.state.log_history else 0,
            "best_eval_loss": trainer.state.best_metric if eval_data else None
        }
        
        with open(os.path.join(output_dir, 'run_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
            
        # Log final metrics
        if config.get('enable_mlflow'):
            final_loss = metadata['final_loss']
            mlflow.log_metric("final_train_loss", final_loss)
            if eval_data:
                mlflow.log_metric("best_eval_loss", metadata['best_eval_loss'])
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
