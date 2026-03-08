#!/usr/bin/env python3
"""Interactive environment setup for QLoRA Training Pipeline."""

import os
import sys
from pathlib import Path

# Constants
ROOT = Path(__file__).parent.resolve()
ENV_FILE = ROOT / '.env'
ENV_TEMPLATE = ROOT / '.env.template'

def setup_environment():
    """Interactive environment setup."""
    print("\n" + "="*60)
    print(" QLoRA Training Pipeline Environment Setup")
    print("="*60)
    
    # Load existing environment if exists
    existing_env = {}
    if ENV_FILE.exists():
        with open(ENV_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    existing_env[key.strip()] = value.strip()
        print(f"✅ Loaded existing configuration from {ENV_FILE}")
    
    # Get user input
    config = {}
    
    print("\n🤖 Teacher API Configuration:")
    config['TEACHER_API_KEY'] = input(f"Teacher API Key [{existing_env.get('TEACHER_API_KEY', 'YOUR_TEACHER_API_KEY_HERE')}]: ").strip() or existing_env.get('TEACHER_API_KEY', 'YOUR_TEACHER_API_KEY_HERE')
    config['TEACHER_MODEL'] = input(f"Primary Teacher model [{existing_env.get('TEACHER_MODEL', 'github-copilot/gpt-5.3-codex')}]: ").strip() or existing_env.get('TEACHER_MODEL', 'github-copilot/gpt-5.3-codex')
    config['TEACHER_FAILBACK_MODEL'] = input(f"Failback Teacher model [{existing_env.get('TEACHER_FAILBACK_MODEL', 'xai/grok-4-1-fast')}]: ").strip() or existing_env.get('TEACHER_FAILBACK_MODEL', 'xai/grok-4-1-fast')

    print("\n💰 Pricing Configuration:")
    print("Configure pricing for primary model (per 1K tokens):")
    
    # Get current pricing or use defaults
    if config['TEACHER_MODEL'] == 'github-copilot/gpt-5.3-codex':
        default_input = '0.00'
        default_output = '0.00'
    elif config['TEACHER_MODEL'] == 'xai/grok-4-1-fast':
        default_input = '0.20'
        default_output = '0.50'
    else:
        default_input = '0.20'
        default_output = '0.50'
    
    config['PRIMARY_INPUT_PRICE'] = input(f"Input Price per 1K tokens [${default_input}]: ").strip() or default_input
    config['PRIMARY_OUTPUT_PRICE'] = input(f"Output Price per 1K tokens [${default_output}]: ").strip() or default_output
    
    print("\n🤗 HuggingFace Configuration:")
    config['HF_TOKEN'] = input(f"HuggingFace Token (optional) [{existing_env.get('HF_TOKEN', 'YOUR_HUGGINGFACE_TOKEN_HERE')}]: ").strip() or existing_env.get('HF_TOKEN', 'YOUR_HUGGINGFACE_TOKEN_HERE')
    
    print("\n⚙️ Training Configuration:")
    config['LANGUAGES'] = input(f"Languages [{existing_env.get('LANGUAGES', 'python,javascript,typescript,cpp')}]: ").strip() or existing_env.get('LANGUAGES', 'python,javascript,typescript,cpp')
    config['ROUNDS'] = input(f"Training rounds [{existing_env.get('ROUNDS', '10')}]: ").strip() or existing_env.get('ROUNDS', '10')
    config['SAMPLES_PER_ROUND'] = input(f"Samples per round [{existing_env.get('SAMPLES_PER_ROUND', '100')}]: ").strip() or existing_env.get('SAMPLES_PER_ROUND', '100')
    config['TRAIN_STEPS'] = input(f"Training steps [{existing_env.get('TRAIN_STEPS', '500')}]: ").strip() or existing_env.get('TRAIN_STEPS', '500')
    config['BATCH_SIZE'] = input(f"Batch size [{existing_env.get('BATCH_SIZE', '1')}]: ").strip() or existing_env.get('BATCH_SIZE', '1')
    config['LEARNING_RATE'] = input(f"Learning rate [{existing_env.get('LEARNING_RATE', '2e-4')}]: ").strip() or existing_env.get('LEARNING_RATE', '2e-4')
    
    gen_only = input(f"Run in Data Generation Only mode? (y/N) [{existing_env.get('GENERATE_ONLY', 'false')}]: ").strip().lower()
    config['GENERATE_ONLY'] = 'true' if gen_only in ['y', 'yes'] else 'false'
    
    print("\n🏗️ Infrastructure Configuration:")
    enable_mlflow = input(f"Enable MLflow? (Y/n) [{existing_env.get('ENABLE_MLFLOW', 'true')}]: ").strip().lower()
    config['ENABLE_MLFLOW'] = 'true' if enable_mlflow in ['', 'y', 'yes'] else 'false'
    
    enable_tensorboard = input(f"Enable TensorBoard? (Y/n) [{existing_env.get('ENABLE_TENSORBOARD', 'true')}]: ").strip().lower()
    config['ENABLE_TENSORBOARD'] = 'true' if enable_tensorboard in ['', 'y', 'yes'] else 'false'
    
    enable_dashboard = input(f"Enable Dashboard? (Y/n) [{existing_env.get('ENABLE_DASHBOARD', 'true')}]: ").strip().lower()
    config['ENABLE_DASHBOARD'] = 'true' if enable_dashboard in ['', 'y', 'yes'] else 'false'
    
    print("\n🎯 Advanced Configuration:")
    config['BASE_MODEL'] = input(f"Base model [{existing_env.get('BASE_MODEL', 'mistralai/Mistral-7B-Instruct-v0.2')}]: ").strip() or existing_env.get('BASE_MODEL', 'mistralai/Mistral-7B-Instruct-v0.2')
    config['LORA_R'] = input(f"LoRA rank [{existing_env.get('LORA_R', '64')}]: ").strip() or existing_env.get('LORA_R', '64')
    config['LORA_ALPHA'] = input(f"LoRA alpha [{existing_env.get('LORA_ALPHA', '16')}]: ").strip() or existing_env.get('LORA_ALPHA', '16')
    config['LORA_DROPOUT'] = input(f"LoRA dropout [{existing_env.get('LORA_DROPOUT', '0.1')}]: ").strip() or existing_env.get('LORA_DROPOUT', '0.1')
    
    # Write environment file
    with open(ENV_FILE, 'w') as f:
        f.write("# QLoRA Training Pipeline Environment Configuration\n")
        f.write("# Generated by setup_env.py\n")
        f.write("# " + str(os.path.basename(__file__)) + "\n\n")
        
        f.write("# =============================================================================\n")
        f.write("# TEACHER API CONFIGURATION\n")
        f.write("# =============================================================================\n")
        f.write(f"TEACHER_API_KEY={config['TEACHER_API_KEY']}\n")
        f.write(f"TEACHER_MODEL={config['TEACHER_MODEL']}\n")
        f.write(f"TEACHER_FAILBACK_MODEL={config['TEACHER_FAILBACK_MODEL']}\n\n")
        
        f.write("# =============================================================================\n")
        f.write("# PRICING CONFIGURATION\n")
        f.write("# =============================================================================\n")
        f.write(f"GPT_5_3_CODEX_INPUT_PRICE={config['PRIMARY_INPUT_PRICE']}\n")
        f.write(f"GPT_5_3_CODEX_OUTPUT_PRICE={config['PRIMARY_OUTPUT_PRICE']}\n\n")
        
        f.write("# =============================================================================\n")
        f.write("# HUGGINGFACE CONFIGURATION\n")
        f.write("# =============================================================================\n")
        f.write(f"HF_TOKEN={config['HF_TOKEN']}\n\n")
        
        f.write("# =============================================================================\n")
        f.write("# TRAINING CONFIGURATION\n")
        f.write("# =============================================================================\n")
        f.write(f"LANGUAGES={config['LANGUAGES']}\n")
        f.write(f"ROUNDS={config['ROUNDS']}\n")
        f.write(f"SAMPLES_PER_ROUND={config['SAMPLES_PER_ROUND']}\n")
        f.write(f"TRAIN_STEPS={config['TRAIN_STEPS']}\n")
        f.write(f"BATCH_SIZE={config['BATCH_SIZE']}\n")
        f.write(f"LEARNING_RATE={config['LEARNING_RATE']}\n")
        f.write(f"GENERATE_ONLY={config['GENERATE_ONLY']}\n\n")
        
        f.write("# =============================================================================\n")
        f.write("# INFRASTRUCTURE CONFIGURATION\n")
        f.write("# =============================================================================\n")
        f.write(f"ENABLE_MLFLOW={config['ENABLE_MLFLOW']}\n")
        f.write(f"ENABLE_TENSORBOARD={config['ENABLE_TENSORBOARD']}\n")
        f.write(f"ENABLE_WANDB=false\n")
        f.write(f"ENABLE_DASHBOARD={config['ENABLE_DASHBOARD']}\n\n")
        
        f.write("# =============================================================================\n")
        f.write("# MONITORING CONFIGURATION\n")
        f.write("# =============================================================================\n")
        f.write("DASHBOARD_REFRESH_RATE=2\n")
        f.write("GPU_POLL_INTERVAL=5\n")
        f.write("DASHBOARD_MAX_EVENTS=20\n\n")
        
        f.write("# =============================================================================\n")
        f.write("# ADVANCED CONFIGURATION\n")
        f.write("# =============================================================================\n")
        f.write(f"BASE_MODEL={config['BASE_MODEL']}\n")
        f.write(f"LORA_R={config['LORA_R']}\n")
        f.write(f"LORA_ALPHA={config['LORA_ALPHA']}\n")
        f.write(f"LORA_DROPOUT={config['LORA_DROPOUT']}\n\n")
        
        f.write("# =============================================================================\n")
        f.write("# SECURITY & PERFORMANCE\n")
        f.write("# =============================================================================\n")
        f.write("API_TIMEOUT=30\n")
        f.write("MAX_RETRIES=3\n")
        f.write("RATE_LIMIT_DELAY=1\n")
    
    print(f"\n✅ Environment configuration saved to {ENV_FILE}")
    print("\n🎉 Setup complete! You can now run './run.sh up' to start the pipeline.")
    print(f"\n📝 Edit {ENV_FILE} anytime to change configuration.")

def main():
    """Main entry point."""
    setup_environment()

if __name__ == "__main__":
    main()
