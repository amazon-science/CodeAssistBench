#!/usr/bin/env python3
"""
Convert hf_dataset JSON files to JSONL format for Hugging Face upload

Converts:
- hf_dataset/all_time/{language}/*.json → hf_dataset/jsonl/all_time.jsonl
- hf_dataset/recent/{language}/*.json → hf_dataset/jsonl/recent.jsonl

Each issue will have a 'language' field added.

Usage:
    python script/convert_to_jsonl.py
"""

import os
import json
from datetime import datetime

# Configuration
INPUT_BASE_DIR = "hf_dataset"
OUTPUT_JSONL_DIR = "hf_dataset/jsonl"

def convert_to_jsonl(time_period):
    """Convert JSON files to JSONL format"""
    print(f"\n{'='*80}")
    print(f"[{time_period}] Converting to JSONL")
    print(f"{'='*80}")
    
    input_dir = os.path.join(INPUT_BASE_DIR, time_period)
    output_file = os.path.join(OUTPUT_JSONL_DIR, f"{time_period}.jsonl")
    
    if not os.path.exists(input_dir):
        print(f"  ⚠️  {input_dir} not found")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_JSONL_DIR, exist_ok=True)
    
    # Collect all issues with language
    all_issues = []
    
    languages = [d for d in os.listdir(input_dir) 
                if os.path.isdir(os.path.join(input_dir, d))]
    
    print(f"\n  Processing {len(languages)} languages...")
    
    for language in sorted(languages):
        lang_dir = os.path.join(input_dir, language)
        json_files = [f for f in os.listdir(lang_dir) if f.endswith('.json')]
        
        lang_issue_count = 0
        
        for filename in json_files:
            filepath = os.path.join(lang_dir, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    issues = json.load(f)
                
                # Add language field to each issue
                for issue in issues:
                    issue['language'] = language
                    all_issues.append(issue)
                    lang_issue_count += 1
            
            except Exception as e:
                print(f"    ⚠️  Error reading {filename}: {e}")
        
        print(f"    [{language:12}] {lang_issue_count:4} issues")
    
    # Write to JSONL
    print(f"\n  Writing to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for issue in all_issues:
            json.dump(issue, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"  ✅ {len(all_issues)} issues written to {output_file}")
    
    # Verify file size
    file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    print(f"  File size: {file_size:.2f} MB")
    
    return {
        'issues': len(all_issues),
        'languages': len(languages),
        'file_size_mb': file_size
    }

def verify_jsonl(time_period):
    """Verify JSONL file structure"""
    print(f"\n{'='*80}")
    print(f"[{time_period}] Verification")
    print(f"{'='*80}")
    
    jsonl_file = os.path.join(OUTPUT_JSONL_DIR, f"{time_period}.jsonl")
    
    if not os.path.exists(jsonl_file):
        print(f"  ⚠️  {jsonl_file} not found")
        return
    
    # Read first few lines
    print(f"\n  Checking first 3 lines...")
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            
            issue = json.loads(line)
            print(f"\n  Line {i+1}:")
            print(f"    Language: {issue.get('language')}")
            print(f"    Number: {issue.get('number')}")
            print(f"    Title: {issue.get('title', '')[:50]}...")
            print(f"    Keys: {list(issue.keys())}")
    
    # Count total lines
    print(f"\n  Counting total lines...")
    with open(jsonl_file, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"  Total lines: {total_lines}")
    
    # Language distribution
    print(f"\n  Language distribution:")
    lang_counts = {}
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            issue = json.loads(line)
            lang = issue.get('language', 'unknown')
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
    
    for lang in sorted(lang_counts.keys()):
        print(f"    {lang:12}: {lang_counts[lang]:4} issues")
    
    return {
        'total_lines': total_lines,
        'languages': lang_counts
    }

def main():
    """Main function"""
    print("\n" + "="*80)
    print("Convert to JSONL for Hugging Face")
    print("="*80)
    
    stats = {}
    
    # Convert all_time
    stats['all_time'] = convert_to_jsonl('all_time')
    
    # Convert recent
    stats['recent'] = convert_to_jsonl('recent')
    
    # Verify all_time
    verify_jsonl('all_time')
    
    # Verify recent
    verify_jsonl('recent')
    
    # Final summary
    print("\n" + "="*80)
    print("✅ Conversion Complete!")
    print("="*80)
    
    print(f"\nOutput directory: {OUTPUT_JSONL_DIR}/")
    
    print(f"\nAll Time:")
    print(f"  Issues: {stats['all_time']['issues']}")
    print(f"  Languages: {stats['all_time']['languages']}")
    print(f"  File size: {stats['all_time']['file_size_mb']:.2f} MB")
    print(f"  File: {OUTPUT_JSONL_DIR}/all_time.jsonl")
    
    print(f"\nRecent:")
    print(f"  Issues: {stats['recent']['issues']}")
    print(f"  Languages: {stats['recent']['languages']}")
    print(f"  File size: {stats['recent']['file_size_mb']:.2f} MB")
    print(f"  File: {OUTPUT_JSONL_DIR}/recent.jsonl")
    
    print(f"\nNext steps:")
    print(f"  1. Review the JSONL files in {OUTPUT_JSONL_DIR}/")
    print(f"  2. Upload to Hugging Face using:")
    print(f"     - huggingface-cli login")
    print(f"     - huggingface-cli upload <dataset-name> {OUTPUT_JSONL_DIR}/all_time.jsonl")
    print(f"     - huggingface-cli upload <dataset-name> {OUTPUT_JSONL_DIR}/recent.jsonl")

if __name__ == "__main__":
    main()

