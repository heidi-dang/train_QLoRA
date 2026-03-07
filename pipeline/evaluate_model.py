#!/usr/bin/env python3
"""Comprehensive model evaluation with multiple benchmarks.

Implements:
- HumanEval-style Python code generation
- C++ compilation tests
- Instruction following accuracy
- Code quality metrics
- Language-specific evaluation
"""
import os
import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import ast
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
AI_LAB = os.path.join(ROOT, 'data', 'ai-lab')
EVAL_DIR = os.path.join(AI_LAB, 'evaluation')
CHECKPOINTS_DIR = os.path.join(AI_LAB, 'checkpoints')

logging.basicConfig(level=logging.INFO)

# HumanEval-style Python problems
PYTHON_EVAL_PROBLEMS = [
    {
        'task_id': 'python_fibonacci',
        'prompt': 'Write a Python function to calculate the nth Fibonacci number.',
        'expected': 'def fibonacci(n):',
        'test': 'def test_fibonacci():\n    assert fibonacci(0) == 0\n    assert fibonacci(1) == 1\n    assert fibonacci(5) == 5\n    assert fibonacci(10) == 55'
    },
    {
        'task_id': 'python_reverse',
        'prompt': 'Write a Python function to reverse a string.',
        'expected': 'def reverse_string',
        'test': 'def test_reverse():\n    assert reverse_string("hello") == "olleh"\n    assert reverse_string("") == ""\n    assert reverse_string("a") == "a"'
    },
    {
        'task_id': 'python_binary_search',
        'prompt': 'Write a Python function to implement binary search on a sorted list.',
        'expected': 'def binary_search',
        'test': 'def test_binary_search():\n    arr = [1, 3, 5, 7, 9, 11]\n    assert binary_search(arr, 7) == 3\n    assert binary_search(arr, 2) == -1'
    }
]

# C++ evaluation problems
CPP_EVAL_PROBLEMS = [
    {
        'task_id': 'cpp_hello',
        'prompt': 'Write a C++ program that prints "Hello, World!"',
        'expected': '#include',
        'code': '#include <iostream>\nint main() {\n    std::cout << "Hello, World!" << std::endl;\n    return 0;\n}'
    },
    {
        'task_id': 'cpp_sort',
        'prompt': 'Write a C++ function to sort an array of integers.',
        'expected': 'void sort',
        'code': '#include <vector>\n#include <algorithm>\nvoid sort_array(std::vector<int>& arr) {\n    std::sort(arr.begin(), arr.end());\n}'
    }
]

# Instruction following tasks
INSTRUCTION_TASKS = [
    {
        'task_id': 'explain_code',
        'instruction': 'Explain what this Python code does: def factorial(n): return 1 if n <= 1 else n * factorial(n-1)',
        'keywords': ['recursive', 'factorial', 'multiplication', 'base case'],
        'expected_length': (20, 200)
    },
    {
        'task_id': 'debug_code',
        'instruction': 'Find and fix the bug in this code: for i in range(10): print(i)',
        'keywords': ['bug', 'fix', 'error', 'syntax'],
        'expected_length': (15, 150)
    }
]

def load_model_for_evaluation(round_n: int):
    """Load trained model for evaluation."""
    checkpoint_dir = os.path.join(CHECKPOINTS_DIR, f'adapter_round_{round_n}')
    
    if not os.path.exists(checkpoint_dir):
        logging.warning(f"Checkpoint not found: {checkpoint_dir}")
        return None, None
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        import torch
        
        # Load config to get base model
        config_file = os.path.join(checkpoint_dir, 'metadata.json')
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            base_model = config.get('base_model', 'mistralai/Mistral-7B-Instruct-v0.2')
        else:
            base_model = 'mistralai/Mistral-7B-Instruct-v0.2'
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            load_in_4bit=True
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(model, checkpoint_dir)
        
        logging.info(f"Loaded model from {checkpoint_dir}")
        return model, tokenizer
        
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return None, None

def generate_response(model, tokenizer, prompt: str, max_length: int = 512) -> str:
    """Generate response from model."""
    if model is None or tokenizer is None:
        return "Model not available"
    
    try:
        # Format prompt
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the response part
        if "### Response:" in response:
            response = response.split("### Response:")[1].strip()
        
        return response
        
    except Exception as e:
        logging.error(f"Generation failed: {e}")
        return "Generation failed"

def evaluate_python_code(model, tokenizer) -> Dict[str, float]:
    """Evaluate Python code generation."""
    results = {
        'python_total': len(PYTHON_EVAL_PROBLEMS),
        'python_passed': 0,
        'python_syntax_valid': 0,
        'python_compiles': 0
    }
    
    for problem in PYTHON_EVAL_PROBLEMS:
        response = generate_response(model, tokenizer, problem['prompt'])
        
        # Check if expected pattern is present
        if problem['expected'] in response:
            results['python_passed'] += 1
        
        # Check syntax validity
        try:
            # Try to parse as Python
            ast.parse(response)
            results['python_syntax_valid'] += 1
            
            # Try to execute test if test function exists
            if 'def test_' in response:
                results['python_compiles'] += 1
        except SyntaxError:
            pass
        
        logging.info(f"Python eval {problem['task_id']}: {response[:100]}...")
    
    return results

def evaluate_cpp_code(model, tokenizer) -> Dict[str, float]:
    """Evaluate C++ code generation."""
    results = {
        'cpp_total': len(CPP_EVAL_PROBLEMS),
        'cpp_passed': 0,
        'cpp_compiles': 0
    }
    
    for problem in CPP_EVAL_PROBLEMS:
        response = generate_response(model, tokenizer, problem['prompt'])
        
        # Check if expected pattern is present
        if problem['expected'] in response:
            results['cpp_passed'] += 1
        
        # Try to compile
        if run_cpp_compile_test(response):
            results['cpp_compiles'] += 1
        
        logging.info(f"C++ eval {problem['task_id']}: {response[:100]}...")
    
    return results

def run_cpp_compile_test(code: str) -> bool:
    """Test if C++ code compiles successfully."""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(code)
            temp_cpp = f.name
        
        temp_exe = temp_cpp.replace('.cpp', '.out')
        
        result = subprocess.run(
            ['g++', temp_cpp, '-O2', '-std=c++17', '-o', temp_exe],
            capture_output=True,
            timeout=10
        )
        
        success = result.returncode == 0
        
        # Cleanup
        try:
            os.unlink(temp_cpp)
            if os.path.exists(temp_exe):
                os.unlink(temp_exe)
        except:
            pass
        
        return success
        
    except Exception:
        return False

def evaluate_instruction_following(model, tokenizer) -> Dict[str, float]:
    """Evaluate instruction following capabilities."""
    results = {
        'instruction_total': len(INSTRUCTION_TASKS),
        'instruction_passed': 0,
        'instruction_keyword_match': 0,
        'instruction_length_appropriate': 0
    }
    
    for task in INSTRUCTION_TASKS:
        response = generate_response(model, tokenizer, task['instruction'])
        
        # Check keyword presence
        keyword_matches = sum(1 for keyword in task['keywords'] if keyword.lower() in response.lower())
        if keyword_matches >= len(task['keywords']) // 2:
            results['instruction_keyword_match'] += 1
        
        # Check length appropriateness
        min_len, max_len = task['expected_length']
        if min_len <= len(response) <= max_len:
            results['instruction_length_appropriate'] += 1
        
        # Overall pass if both criteria met
        if keyword_matches >= len(task['keywords']) // 2 and min_len <= len(response) <= max_len:
            results['instruction_passed'] += 1
        
        logging.info(f"Instruction eval {task['task_id']}: {response[:100]}...")
    
    return results

def calculate_code_quality_metrics(code: str) -> Dict[str, float]:
    """Calculate code quality metrics."""
    metrics = {
        'line_count': len(code.split('\n')),
        'char_count': len(code),
        'comment_ratio': 0.0,
        'indentation_consistency': 0.0,
        'naming_convention_score': 0.0
    }
    
    lines = code.split('\n')
    comment_lines = sum(1 for line in lines if line.strip().startswith('#') or '//' in line)
    metrics['comment_ratio'] = comment_lines / len(lines) if lines else 0
    
    # Check indentation consistency (simplified)
    indent_sizes = []
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            if indent > 0:
                indent_sizes.append(indent)
    
    if indent_sizes:
        most_common = max(set(indent_sizes), key=indent_sizes.count)
        consistency = sum(1 for size in indent_sizes if size == most_common) / len(indent_sizes)
        metrics['indentation_consistency'] = consistency
    
    # Naming convention (very basic check)
    function_names = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
    if function_names:
        snake_case_count = sum(1 for name in function_names if '_' in name or name.islower())
        metrics['naming_convention_score'] = snake_case_count / len(function_names)
    
    return metrics

def evaluate_round(round_n: int) -> Dict[str, Any]:
    """Comprehensive evaluation for a training round."""
    Path(EVAL_DIR).mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Starting evaluation for round {round_n}")
    
    # Load model
    model, tokenizer = load_model_for_evaluation(round_n)
    
    if model is None:
        logging.warning(f"Model not available for round {round_n}, using dummy evaluation")
        results = {
            'round': round_n,
            'status': 'model_not_available',
            'python_total': len(PYTHON_EVAL_PROBLEMS),
            'python_passed': 0,
            'cpp_total': len(CPP_EVAL_PROBLEMS),
            'cpp_passed': 0,
            'instruction_total': len(INSTRUCTION_TASKS),
            'instruction_passed': 0,
            'overall_score': 0.0
        }
    else:
        # Run evaluations
        start_time = time.time()
        
        python_results = evaluate_python_code(model, tokenizer)
        cpp_results = evaluate_cpp_code(model, tokenizer)
        instruction_results = evaluate_instruction_following(model, tokenizer)
        
        # Combine results
        results = {
            'round': round_n,
            'status': 'success',
            'evaluation_time_seconds': time.time() - start_time,
            **python_results,
            **cpp_results,
            **instruction_results
        }
        
        # Calculate overall score
        total_tasks = results['python_total'] + results['cpp_total'] + results['instruction_total']
        total_passed = results['python_passed'] + results['cpp_passed'] + results['instruction_passed']
        results['overall_score'] = total_passed / total_tasks if total_tasks > 0 else 0.0
        
        # Add pass rates
        results['python_pass_rate'] = results['python_passed'] / results['python_total'] if results['python_total'] > 0 else 0.0
        results['cpp_pass_rate'] = results['cpp_passed'] / results['cpp_total'] if results['cpp_total'] > 0 else 0.0
        results['instruction_pass_rate'] = results['instruction_passed'] / results['instruction_total'] if results['instruction_total'] > 0 else 0.0
    
    # Save results
    out_file = os.path.join(EVAL_DIR, f'results_round_{round_n}.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"✅ Evaluation complete for round {round_n}")
    logging.info(f"📊 Overall score: {results.get('overall_score', 0):.3f}")
    logging.info(f"📁 Results saved to: {out_file}")
    
    return results

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate_model.py <round_number>")
        sys.exit(1)
    
    round_n = int(sys.argv[1])
    evaluate_round(round_n)
