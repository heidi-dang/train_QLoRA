#!/usr/bin/env python3
"""Interactive configuration setup for QLoRA training pipeline."""
import os
import json
from pathlib import Path

ROOT = os.path.abspath(os.path.dirname(__file__))
CONFIG_FILE = os.path.join(ROOT, 'config.json')

def get_user_input():
    """Interactive configuration prompts."""
    config = {}
    
    print("\n" + "="*60)
    print(" QLoRA Training Pipeline Configuration")
    print("="*60)
    
    # Teacher Model Configuration
    print("\n🤖 Teacher Model Configuration:")
    config['teacher_api_key'] = input("Enter Teacher API Key (Grok/OpenAI/Anthropic): ").strip()
    config['teacher_model'] = input("Teacher model [grok-beta]: ").strip() or "grok-beta"
    
    # Training Model Configuration
    print("\n🎯 Training Model Configuration:")
    config['base_model'] = input("Base model to train [mistralai/Mistral-7B-Instruct-v0.2]: ").strip() or "mistralai/Mistral-7B-Instruct-v0.2"
    config['hf_token'] = input("HuggingFace Token (optional): ").strip()
    
    # Training Parameters
    print("\n⚙️ Training Parameters:")
    config['languages'] = input("Languages to train (comma separated) [python,javascript,typescript,cpp]: ").strip() or "python,javascript,typescript,cpp"
    config['rounds'] = int(input("Number of training rounds [10]: ") or "10")
    config['samples_per_round'] = int(input("Samples per round [100]: ") or "100")
    config['train_steps'] = int(input("Training steps per round [500]: ") or "500")
    config['batch_size'] = int(input("Batch size [1]: ") or "1")
    config['learning_rate'] = float(input("Learning rate [2e-4]: ") or "2e-4")
    
    # GitHub Search Configuration
    print("\n🔍 GitHub Search Configuration:")
    config['github_search_enabled'] = input("Enable GitHub auto-search? [y/N]: ").strip().lower() == 'y'
    if config['github_search_enabled']:
        config['min_stars'] = int(input("Minimum stars [100]: ") or "100")
        config['min_forks'] = int(input("Minimum forks [50]: ") or "50")
        config['max_repos'] = int(input("Maximum repos to download [50]: ") or "50")
        config['search_languages'] = input("Search languages (comma separated) [python,javascript]: ").strip() or "python,javascript"
    else:
        config['repo_list_file'] = input("Repository list file path: ").strip()
    
    # Infrastructure Configuration
    print("\n🏗️ Infrastructure Configuration:")
    config['enable_mlflow'] = input("Enable MLflow tracking? [Y/n]: ").strip().lower() != 'n'
    config['enable_tensorboard'] = input("Enable TensorBoard? [Y/n]: ").strip().lower() != 'n'
    config['enable_wandb'] = input("Enable Weights & Biases? [y/N]: ").strip().lower() == 'y'
    
    # Dashboard Configuration
    config['enable_dashboard'] = input("Enable rich monitoring dashboard? [Y/n]: ").strip().lower() != 'n'
    
    if config['enable_wandb']:
        config['wandb_api_key'] = input("W&B API Key: ").strip()
        config['wandb_project'] = input("W&B Project Name [qlora-continuous]: ").strip() or "qlora-continuous"
    
    return config

def save_config(config):
    """Save configuration to file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n✅ Configuration saved to {CONFIG_FILE}")

def load_config():
    """Load existing configuration."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return None

def setup_environment(config):
    """Set up environment variables."""
    env_vars = {
        'TEACHER_API_KEY': config.get('teacher_api_key'),
        'TEACHER_MODEL': config.get('teacher_model'),
        'BASE_MODEL': config.get('base_model'),
        'HF_TOKEN': config.get('hf_token'),
        'LANGUAGES': config.get('languages'),
        'ROUNDS': str(config.get('rounds')),
        'SAMPLES_PER_ROUND': str(config.get('samples_per_round')),
        'TRAIN_STEPS': str(config.get('train_steps')),
        'BATCH_SIZE': str(config.get('batch_size')),
        'LEARNING_RATE': str(config.get('learning_rate')),
        'MIN_STARS': str(config.get('min_stars', 0)),
        'MIN_FORKS': str(config.get('min_forks', 0)),
        'MAX_REPOS': str(config.get('max_repos', 0)),
        'ENABLE_MLFLOW': str(config.get('enable_mlflow', True)).lower(),
        'ENABLE_TENSORBOARD': str(config.get('enable_tensorboard', True)).lower(),
        'ENABLE_WANDB': str(config.get('enable_wandb', False)).lower(),
        'ENABLE_DASHBOARD': str(config.get('enable_dashboard', True)).lower(),
        'SEARCH_LANGUAGES': config.get('search_languages', ''),
    }
    
    if config.get('enable_mlflow'):
        os.environ['MLFLOW_TRACKING_URI'] = f'sqlite:///{ROOT}/state/mlflow.db'
    
    if config.get('enable_wandb'):
        os.environ['WANDB_API_KEY'] = config.get('wandb_api_key')
        os.environ['WANDB_PROJECT'] = config.get('wandb_project')
    
    # Write to .env file
    env_file = os.path.join(ROOT, '.env')
    with open(env_file, 'w') as f:
        for key, value in env_vars.items():
            if value:
                f.write(f"{key}={value}\n")
    
    print(f"✅ Environment variables written to {env_file}")

def main():
    """Main setup function."""
    # Check if config exists
    existing_config = load_config()
    if existing_config:
        print(f"Found existing configuration in {CONFIG_FILE}")
        overwrite = input("Overwrite existing config? [y/N]: ").strip().lower() == 'y'
        if not overwrite:
            setup_environment(existing_config)
            return
    
    # Get new configuration
    config = get_user_input()
    
    # Save and setup
    save_config(config)
    setup_environment(config)
    
    print("\n🎉 Setup complete! You can now run './run.sh up' to start the pipeline.")

if __name__ == '__main__':
    main()
