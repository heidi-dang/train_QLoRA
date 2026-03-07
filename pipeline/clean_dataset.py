#!/usr/bin/env python3
"""Clean raw teacher outputs into a deduplicated, token-limited training file.

Implements advanced data cleaning including:
- Deduplication
- Quality filtering
- Language detection
- Toxicity filtering
- Length filtering
- Format validation
"""
import os
import json
import re
from pathlib import Path
import logging
import hashlib
from typing import Dict, List, Set, Any
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
AI_LAB = os.path.join(ROOT, 'data', 'ai-lab')
RAW_DIR = os.path.join(AI_LAB, 'datasets', 'raw')
CLEAN_DIR = os.path.join(AI_LAB, 'datasets', 'clean')
OUT_FILE = os.path.join(CLEAN_DIR, 'train.json')

logging.basicConfig(level=logging.INFO)

# Configuration
MAX_TOKENS = 2048
MIN_CHARS = 50
MAX_CHARS = 8000
MIN_QUALITY_SCORE = 0.2
MIN_ENTROPY = 0.05

# Toxic words filter (basic)
TOXIC_WORDS = {
    'toxic', 'hate', 'kill', 'die', 'stupid', 'idiot', 'dumb', 'moron',
    'retard', 'loser', 'pathetic', 'worthless', 'garbage', 'trash'
}

# Language patterns
LANG_PATTERNS = {
    'python': r'\b(def |class |import |from |if __name__|print\(|#)',
    'javascript': r'\b(function|const|let|var|=>|console\.|require\()',
    'typescript': r'\b(interface|type|as|namespace|declare|abstract)',
    'cpp': r'\b(#include|int main|std::|cout|cin|->)',
    'java': r'\b(public class|private|protected|static void|System\.out)',
    'go': r'\b(func |package |import \(|fmt\.|go func)',
    'rust': r'\b(fn |use |mut |impl |let |match)',
}

def normalize(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize sample by trimming and cleaning fields."""
    for k in ('instruction', 'context', 'response'):
        if k in sample and isinstance(sample[k], str):
            # Clean whitespace
            sample[k] = re.sub(r'\s+', ' ', sample[k].strip())
            # Remove excessive repetition
            sample[k] = re.sub(r'(.)\1{4,}', r'\1\1\1', sample[k])
            # Length limits
            if len(sample[k]) > MAX_CHARS:
                sample[k] = sample[k][:MAX_CHARS]
    return sample

def calculate_entropy(text: str) -> float:
    """Calculate text entropy for quality assessment."""
    if not text or len(text) < 10:
        return 0.0
    
    # Character-level entropy
    char_counts = Counter(text)
    total_chars = len(text)
    entropy = -sum((count / total_chars) * __import__('math').log2(count / total_chars) 
                   for count in char_counts.values())
    
    # Normalize to 0-1 range
    max_entropy = __import__('math').log2(min(len(char_counts), 256))
    return entropy / max_entropy if max_entropy > 0 else 0.0

def detect_language(text: str) -> str:
    """Detect programming language from code."""
    scores = {}
    for lang, pattern in LANG_PATTERNS.items():
        matches = len(re.findall(pattern, text, re.IGNORECASE))
        scores[lang] = matches
    
    if scores:
        best_lang = max(scores, key=scores.get)
        if scores[best_lang] > 0:
            return best_lang
    return 'unknown'

def contains_toxic_content(text: str) -> bool:
    """Basic toxic content detection."""
    text_lower = text.lower()
    return any(word in text_lower for word in TOXIC_WORDS)

def validate_format(sample: Dict[str, Any]) -> bool:
    """Validate sample format."""
    required_fields = ['instruction', 'response']
    for field in required_fields:
        if field not in sample or not sample[field] or len(sample[field].strip()) < 10:
            return False
    return True

def calculate_quality_score(sample: Dict[str, Any]) -> float:
    """Calculate overall quality score for a sample."""
    score = 0.0
    
    # Base quality from metadata
    score += sample.get('quality_score', 0.0) * 0.3
    
    # Response quality
    response = sample.get('response', '')
    score += min(len(response) / 500, 1.0) * 0.2  # Length component
    score += calculate_entropy(response) * 0.2  # Entropy component
    
    # Instruction clarity
    instruction = sample.get('instruction', '')
    score += min(len(instruction) / 200, 1.0) * 0.1
    
    # Context bonus
    if sample.get('context'):
        score += 0.1
    
    # Language detection bonus
    if detect_language(response) != 'unknown':
        score += 0.1
    
    return min(score, 1.0)

def deduplicate_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate samples based on content hash."""
    seen: Set[str] = set()
    deduped = []
    
    for sample in samples:
        # Create hash from instruction + response
        content = (sample.get('instruction', '') + sample.get('response', '')).encode()
        content_hash = hashlib.sha256(content).hexdigest()
        
        if content_hash not in seen:
            seen.add(content_hash)
            deduped.append(sample)
    
    return deduped

def filter_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply quality filters to samples."""
    filtered = []
    
    for sample in samples:
        # Format validation
        if not validate_format(sample):
            continue
        
        # Toxic content filter
        if contains_toxic_content(sample.get('response', '')):
            continue
        
        # Length filters
        total_chars = len(sample.get('instruction', '')) + len(sample.get('response', ''))
        if total_chars < MIN_CHARS or total_chars > MAX_CHARS:
            continue
        
        # Quality score
        quality = calculate_quality_score(sample)
        if quality < MIN_QUALITY_SCORE:
            continue
        
        # Entropy filter
        response_entropy = calculate_entropy(sample.get('response', ''))
        if response_entropy < MIN_ENTROPY:
            continue
        
        # Update quality score
        sample['quality_score'] = quality
        
        # Add language detection
        sample['detected_language'] = detect_language(sample.get('response', ''))
        
        filtered.append(sample)
    
    return filtered

def balance_by_language(samples: List[Dict[str, Any]], max_per_lang: int = 1000) -> List[Dict[str, Any]]:
    """Balance samples by programming language."""
    lang_samples = {}
    
    for sample in samples:
        lang = sample.get('detected_language', 'unknown')
        if lang not in lang_samples:
            lang_samples[lang] = []
        lang_samples[lang].append(sample)
    
    balanced = []
    for lang, lang_list in lang_samples.items():
        # Sort by quality score and take top samples
        lang_list.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        balanced.extend(lang_list[:max_per_lang])
    
    return balanced

def main():
    """Main data cleaning function."""
    Path(CLEAN_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load all raw samples
    all_samples = []
    raw_files = list(Path(RAW_DIR).glob('*.jsonl'))
    
    logging.info(f"Processing {len(raw_files)} raw files...")
    
    for raw_file in raw_files:
        try:
            with raw_file.open('r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        sample = json.loads(line.strip())
                        sample = normalize(sample)
                        all_samples.append(sample)
                    except json.JSONDecodeError as e:
                        logging.warning(f"Invalid JSON in {raw_file}:{line_num}: {e}")
                        continue
                    except Exception as e:
                        logging.warning(f"Error processing line in {raw_file}:{line_num}: {e}")
                        continue
        except Exception as e:
            logging.error(f"Failed to process {raw_file}: {e}")
    
    logging.info(f"Loaded {len(all_samples)} raw samples")
    
    # Deduplicate
    deduped = deduplicate_samples(all_samples)
    logging.info(f"After deduplication: {len(deduped)} samples")
    
    # Filter by quality
    filtered = filter_samples(deduped)
    logging.info(f"After quality filtering: {len(filtered)} samples")
    
    # Balance by language
    balanced = balance_by_language(filtered)
    logging.info(f"After language balancing: {len(balanced)} samples")
    
    # Sort by quality score
    balanced.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
    
    # Save cleaned dataset
    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(balanced, f, indent=2, ensure_ascii=False)
    
    # Generate statistics
    stats = {
        'total_samples': len(balanced),
        'avg_quality_score': sum(s.get('quality_score', 0) for s in balanced) / len(balanced) if balanced else 0,
        'language_distribution': Counter(s.get('detected_language', 'unknown') for s in balanced),
        'avg_response_length': sum(len(s.get('response', '')) for s in balanced) / len(balanced) if balanced else 0,
    }
    
    stats_file = os.path.join(CLEAN_DIR, 'cleaning_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logging.info(f"✅ Cleaning complete!")
    logging.info(f"📊 Clean samples: {len(balanced)}")
    logging.info(f"📈 Avg quality: {stats['avg_quality_score']:.3f}")
    logging.info(f"🗂️ Languages: {dict(stats['language_distribution'])}")
    logging.info(f"📁 Saved to: {OUT_FILE}")

if __name__ == '__main__':
    main()
