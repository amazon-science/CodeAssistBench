#!/usr/bin/env python3
"""
Filter repositories by first commit date.
Keep only repos created between Jan 1, 2025 and June 30, 2025.
"""

import json
import os
from datetime import datetime
from pathlib import Path

# Date range for filtering (repos created in first half of 2025)
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 6, 30, 23, 59, 59)

def get_first_commit_date(commit_file: str) -> tuple:
    """Get the first commit date from a commit file."""
    try:
        with open(commit_file, 'r') as f:
            data = json.load(f)
        
        commits = data.get('commits', [])
        if not commits:
            return None, data.get('repository', 'unknown')
        
        # First commit is the last in the list (oldest)
        first_commit = commits[-1]
        date_str = first_commit.get('date', '')
        
        if date_str:
            # Parse ISO format date
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return date.replace(tzinfo=None), data.get('repository', 'unknown')
        
        return None, data.get('repository', 'unknown')
    except Exception as e:
        print(f"Error reading {commit_file}: {e}")
        return None, 'unknown'

def main():
    commits_dir = Path("github_commits")
    
    # Track repos by language
    repos_in_range = {}
    repos_out_of_range = {}
    repos_no_commits = {}
    
    # Process all commit files
    for commit_file in commits_dir.glob("commits_*_issues_*.json"):
        # Extract language from filename
        parts = commit_file.name.split('_')
        if len(parts) >= 2:
            language = parts[1]  # e.g., "python", "javascript"
        else:
            language = "unknown"
        
        first_date, repo_url = get_first_commit_date(commit_file)
        
        if first_date is None:
            repos_no_commits.setdefault(language, []).append(repo_url)
        elif START_DATE <= first_date <= END_DATE:
            repos_in_range.setdefault(language, []).append({
                'repo': repo_url,
                'first_commit': first_date.isoformat(),
                'commit_file': commit_file.name
            })
        else:
            repos_out_of_range.setdefault(language, []).append({
                'repo': repo_url,
                'first_commit': first_date.isoformat(),
                'commit_file': commit_file.name
            })
    
    # Print summary
    print("=" * 60)
    print(f"FILTER SUMMARY: Repos created between {START_DATE.date()} and {END_DATE.date()}")
    print("=" * 60)
    
    total_in = 0
    total_out = 0
    total_no = 0
    
    for lang in sorted(set(list(repos_in_range.keys()) + list(repos_out_of_range.keys()) + list(repos_no_commits.keys()))):
        in_count = len(repos_in_range.get(lang, []))
        out_count = len(repos_out_of_range.get(lang, []))
        no_count = len(repos_no_commits.get(lang, []))
        total_in += in_count
        total_out += out_count
        total_no += no_count
        print(f"{lang}: {in_count} in range, {out_count} out of range, {no_count} no commits")
    
    print("-" * 60)
    print(f"TOTAL: {total_in} in range, {total_out} out of range, {total_no} no commits")
    print("=" * 60)
    
    # Save results
    results = {
        'date_range': {
            'start': START_DATE.isoformat(),
            'end': END_DATE.isoformat()
        },
        'repos_in_range': repos_in_range,
        'repos_out_of_range': repos_out_of_range,
        'repos_no_commits': repos_no_commits,
        'summary': {
            'total_in_range': total_in,
            'total_out_of_range': total_out,
            'total_no_commits': total_no
        }
    }
    
    output_file = "github_commits/filter_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    # Print repos in range
    print("\n" + "=" * 60)
    print("REPOS IN RANGE (Jan 1, 2025 - Jun 30, 2025):")
    print("=" * 60)
    for lang in sorted(repos_in_range.keys()):
        print(f"\n{lang.upper()}:")
        for repo in repos_in_range[lang]:
            print(f"  - {repo['repo']} (first commit: {repo['first_commit'][:10]})")

if __name__ == "__main__":
    main()
