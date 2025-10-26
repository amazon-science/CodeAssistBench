#!/usr/bin/env python3
"""
Merge HF dataset + CAB issue data to create a complete dataset

Strategy:
1. Use HF dataset (all_time, recent_time) as the base
2. Fill missing fields from CAB issue data:
   - _classification (100%)
   - closed_at (100%)
   - labels (100%)
   - author (100%)
3. All fields have keys; values are null if missing

Output: hf_dataset/all_time/, hf_dataset/recent/

Usage:
    python script/merge_hf_cab_dataset.py
"""

import os
import json
import re
from datasets import load_dataset
from collections import defaultdict
from datetime import datetime

# Configuration
HF_CACHE_DIR = "hf_dataset"
CAB_ISSUE_DIR = "issue"
OUTPUT_BASE_DIR = "hf_dataset"

def extract_issue_identifier(url):
    """Extract owner/repo#number from GitHub URL"""
    match = re.search(r'github\.com/([^/]+)/([^/]+)/(issues|pull)/(\d+)', url)
    if match:
        owner = match.group(1)
        repo = match.group(2)
        number = int(match.group(4))
        return f"{owner}/{repo}#{number}", owner, repo, number
    return None, None, None, None

def format_datetime(dt):
    """Convert datetime to ISO format string"""
    if isinstance(dt, str):
        return dt
    if dt:
        iso_str = dt.isoformat()
        if not iso_str.endswith('Z') and '+' not in iso_str:
            iso_str += 'Z'
        return iso_str
    return None

def load_cab_issue_data(time_period):
    """Load CAB issue data (all or recent)"""
    print(f"\n  [{time_period}] Loading CAB data...")
    
    cab_data = {}  # {identifier: {closed_at, labels, author, _classification}}
    
    docker_filter_dir = os.path.join(CAB_ISSUE_DIR, time_period, 'docker_filter')
    
    if not os.path.exists(docker_filter_dir):
        print(f"    ⚠️  {docker_filter_dir} not found")
        return cab_data
    
    languages = [d for d in os.listdir(docker_filter_dir) 
                if os.path.isdir(os.path.join(docker_filter_dir, d))]
    
    for language in languages:
        lang_dir = os.path.join(docker_filter_dir, language)
        
        for category in ['need_docker', 'no_need_docker', 'need_docker_but_cannot']:
            category_dir = os.path.join(lang_dir, category)
            if not os.path.exists(category_dir):
                continue
            
            json_files = [f for f in os.listdir(category_dir) 
                        if f.endswith('.json') and f != 'processed_issues.json']
            
            for filename in json_files:
                filepath = os.path.join(category_dir, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        issues = json.load(f)
                    
                    for issue in issues:
                        url = issue.get('url', '')
                        identifier, _, _, _ = extract_issue_identifier(url)
                        
                        if identifier:
                            cab_data[identifier] = {
                                'closed_at': issue.get('closed_at'),
                                'labels': issue.get('labels', []),
                                'author': issue.get('author'),
                                '_classification': issue.get('_classification', {})
                            }
                except Exception as e:
                    print(f"    ⚠️  Error reading {filename}: {e}")
    
    print(f"    ✅ {len(cab_data)} issues loaded")
    return cab_data

def merge_hf_and_cab(hf_example, cab_data):
    """Merge HF data and CAB data"""
    url = hf_example.get('commit_info', {}).get('repository', '')
    identifier, owner, repo, number = extract_issue_identifier(url)
    
    if not identifier or not owner or not repo or not number:
        return None, None, None, None, None
    
    # Extract Commit ID (convert empty string to None)
    commit_sha = hf_example.get('commit_info', {}).get('latest_commit', {}).get('sha', None)
    if commit_sha == '':  # Convert empty string to None
        commit_sha = None
    
    # Skip issues without commit_id
    if commit_sha is None:
        return None, None, None, None, url  # Return URL for logging
    
    # Base data (from HF)
    merged = {
        'number': number,
        'title': hf_example.get('first_question', {}).get('title', ''),
        'created_at': format_datetime(hf_example.get('created_at')),
        'closed_at': None,  # Filled from CAB
        'commit_id': commit_sha,  # Can be None
        'labels': [],  # Filled from CAB
        'url': url,
        'body': hf_example.get('first_question', {}).get('body', ''),
        'comments_url': f"https://api.github.com/repos/{owner}/{repo}/issues/{number}/comments",
        'author': None,  # Filled from CAB
        'comments': [],
        'satisfaction_conditions': hf_example.get('user_satisfaction_condition', []),
        '_classification': None,  # Filled from CAB
        'dockerfile': hf_example.get('dockerfile', None)  # From HF (can be None)
    }
    
    # Convert comments
    for comment in hf_example.get('comments', []):
        merged['comments'].append({
            'user': comment.get('user', ''),
            'created_at': format_datetime(comment.get('created_at')),
            'body': comment.get('body', '')
        })
    
    # Fill with CAB data
    if identifier in cab_data:
        cab_issue = cab_data[identifier]
        merged['closed_at'] = format_datetime(cab_issue.get('closed_at'))
        merged['labels'] = cab_issue.get('labels', [])
        merged['author'] = cab_issue.get('author')
        merged['_classification'] = cab_issue.get('_classification', {})
    else:
        # Default value if not in CAB
        merged['_classification'] = {
            'category': 'Unknown',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    language = hf_example.get('language', 'unknown')
    
    return owner, repo, language, merged, None  # Last None for skipped URL

def process_dataset(dataset_name, time_period_name):
    """Process HF dataset"""
    print(f"\n{'='*80}")
    print(f"[{time_period_name}] Processing")
    print(f"{'='*80}")
    
    # 1. Load HF data
    print(f"\n  Loading HF {dataset_name}...")
    hf_data = load_dataset("codingsoo/CAB", data_files=f"{dataset_name}.jsonl", 
                          cache_dir=HF_CACHE_DIR, split="train")
    print(f"    ✅ {len(hf_data)} issues loaded")
    
    # 2. Load CAB data
    cab_time_period = 'all' if dataset_name == 'all_time' else 'recent'
    cab_data = load_cab_issue_data(cab_time_period)
    
    # 3. Merge and group
    print(f"\n  Merging and grouping data...")
    repo_issues = defaultdict(list)  # {(owner, repo, language): [issues]}
    
    matched = 0
    unmatched = 0
    skipped_issues = []  # Track skipped issues
    
    for i, example in enumerate(hf_data):
        if (i + 1) % 500 == 0:
            print(f"    Progress: {i+1}/{len(hf_data)}")
        
        owner, repo, language, merged, skipped_url = merge_hf_and_cab(example, cab_data)
        
        # Log skipped issues (no commit_id)
        if skipped_url:
            skipped_issues.append(skipped_url)
            continue
        
        if owner and repo and merged:
            key = (owner, repo, language)
            repo_issues[key].append(merged)
            
            # Statistics
            if merged['_classification'] and merged['_classification'].get('category') != 'Unknown':
                matched += 1
            else:
                unmatched += 1
    
    print(f"\n  Merge complete:")
    print(f"    Repositories: {len(repo_issues)}")
    print(f"    CAB matched: {matched}")
    print(f"    CAB not found: {unmatched}")
    print(f"    Skipped (no commit_id): {len(skipped_issues)}")
    
    # Write skipped issues to log file
    if skipped_issues:
        log_file = os.path.join(OUTPUT_BASE_DIR, f"skipped_issues_{time_period_name}.log")
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"# Skipped issues (no commit_id) for {time_period_name}\n")
            f.write(f"# Total: {len(skipped_issues)} issues\n\n")
            for url in skipped_issues:
                f.write(f"{url}\n")
        print(f"    Log file: skipped_issues_{time_period_name}.log")
    
    # 4. Save as JSON files
    print(f"\n  Creating JSON files...")
    
    output_dir = os.path.join(OUTPUT_BASE_DIR, time_period_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create language directories
    languages = set(lang for _, _, lang in repo_issues.keys())
    for language in languages:
        os.makedirs(os.path.join(output_dir, language), exist_ok=True)
    
    saved_files = 0
    total_issues = 0
    
    for (owner, repo, language), issues in repo_issues.items():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"github_issues_{owner}_{repo}_{timestamp}.json"
        filepath = os.path.join(output_dir, language, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(issues, f, indent=2, ensure_ascii=False)
        
        saved_files += 1
        total_issues += len(issues)
        
        if saved_files <= 3:
            print(f"    Created: {time_period_name}/{language}/{filename} ({len(issues)} issues)")
    
    print(f"\n  ✅ {saved_files} files, {total_issues} issues created")
    
    return {
        'files': saved_files,
        'issues': total_issues,
        'matched': matched,
        'unmatched': unmatched,
        'skipped': len(skipped_issues)
    }

def verify_structure():
    """Verify the generated data structure"""
    print(f"\n{'='*80}")
    print("[Verification] Data structure check")
    print(f"{'='*80}")
    
    for time_period in ['all_time', 'recent']:
        output_dir = os.path.join(OUTPUT_BASE_DIR, time_period)
        
        if not os.path.exists(output_dir):
            continue
        
        print(f"\n{'-'*80}")
        print(f"{time_period.replace('_', ' ').title()}")
        print(f"{'-'*80}")
        
        languages = [d for d in os.listdir(output_dir) 
                    if os.path.isdir(os.path.join(output_dir, d))]
        
        for language in sorted(languages)[:3]:  # First 3 languages only
            lang_dir = os.path.join(output_dir, language)
            json_files = [f for f in os.listdir(lang_dir) if f.endswith('.json')]
            
            if json_files:
                sample_file = os.path.join(lang_dir, json_files[0])
                with open(sample_file, 'r') as f:
                    issues = json.load(f)
                
                if issues:
                    sample = issues[0]
                    
                    print(f"\n  [{language}] ({len(json_files)} files)")
                    print(f"    Keys: {list(sample.keys())}")
                    print(f"    commit_id: {sample.get('commit_id') is not None}")
                    print(f"    _classification: {sample.get('_classification', {}).get('category', 'None')}")
                    print(f"    closed_at: {sample.get('closed_at') is not None}")
                    print(f"    labels: {len(sample.get('labels', []))} items")
                    print(f"    author: {sample.get('author') is not None}")
                    print(f"    dockerfile key: {'dockerfile' in sample}")

def main():
    """Main function"""
    print("\n" + "="*80)
    print("HF + CAB Dataset Merge")
    print("="*80)
    
    stats = {}
    
    # Process All Time
    stats['all_time'] = process_dataset('all_time', 'all_time')
    
    # Process Recent Time
    stats['recent'] = process_dataset('recent_time', 'recent')
    
    # Verify structure
    verify_structure()
    
    # Final summary
    print("\n" + "="*80)
    print("✅ Merge Complete!")
    print("="*80)
    
    print(f"\nOutput directories:")
    print(f"  - {OUTPUT_BASE_DIR}/all_time/")
    print(f"  - {OUTPUT_BASE_DIR}/recent/")
    
    print(f"\nAll Time:")
    print(f"  Files: {stats['all_time']['files']}")
    print(f"  Issues: {stats['all_time']['issues']}")
    print(f"  CAB matched: {stats['all_time']['matched']} ({stats['all_time']['matched']/stats['all_time']['issues']*100:.1f}%)")
    print(f"  Skipped (no commit_id): {stats['all_time']['skipped']}")
    
    print(f"\nRecent:")
    print(f"  Files: {stats['recent']['files']}")
    print(f"  Issues: {stats['recent']['issues']}")
    print(f"  CAB matched: {stats['recent']['matched']} ({stats['recent']['matched']/stats['recent']['issues']*100:.1f}%)")
    print(f"  Skipped (no commit_id): {stats['recent']['skipped']}")
    
    print(f"\nKeys in each issue:")
    print(f"  ✅ commit_id: Always present (issues without it are excluded)")
    print(f"  ✅ dockerfile: Always present (null if missing)")
    print(f"  ✅ _classification: Filled from CAB")
    print(f"  ✅ closed_at, labels, author: Filled from CAB")

if __name__ == "__main__":
    main()

