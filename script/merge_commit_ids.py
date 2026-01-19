#!/usr/bin/env python3
"""
Step 6b: Merge commit IDs from github_commits/ into issue files.
Creates the final dataset with commit_id populated.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

def load_commits_for_repo(repo_url: str, commits_dir: Path) -> Optional[Dict]:
    """Find and load commits file for a repository."""
    # Extract owner/repo from URL
    # https://github.com/owner/repo -> owner_repo
    parts = repo_url.rstrip('/').split('/')
    if len(parts) >= 2:
        owner_repo = f"{parts[-2]}_{parts[-1]}"
    else:
        return None
    
    # Search for matching commits file
    for commit_file in commits_dir.glob(f"commits_*_{owner_repo}.json"):
        try:
            with open(commit_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {commit_file}: {e}")
    
    return None

def get_commit_before_date(commits_data: Dict, issue_date: str) -> Optional[str]:
    """Get the most recent commit SHA before the issue date."""
    if not commits_data or 'commits' not in commits_data:
        return None
    
    try:
        issue_dt = datetime.fromisoformat(issue_date.replace('Z', '+00:00'))
    except:
        return None
    
    # Commits are sorted newest to oldest
    for commit in commits_data['commits']:
        try:
            commit_dt = datetime.fromisoformat(commit['date'].replace('Z', '+00:00'))
            if commit_dt < issue_dt:
                return commit['sha']
        except:
            continue
    
    return None

def process_issue_file(input_file: Path, output_file: Path, commits_dir: Path) -> Dict:
    """Process an issue file and add commit_id to each issue."""
    stats = {'total': 0, 'with_commit': 0, 'without_commit': 0}
    
    with open(input_file, 'r') as f:
        issues = json.load(f)
    
    for issue in issues:
        stats['total'] += 1
        
        # Get repository URL - extract from issue URL
        issue_url = issue.get('url', '')
        if not issue_url:
            stats['without_commit'] += 1
            continue
        
        # Extract repo URL from issue URL: https://github.com/owner/repo/issues/123 -> https://github.com/owner/repo
        repo_url = '/'.join(issue_url.split('/')[:5])
        if not repo_url.startswith('https://github.com/'):
            stats['without_commit'] += 1
            continue
        
        # Load commits for this repo
        commits_data = load_commits_for_repo(repo_url, commits_dir)
        if not commits_data:
            stats['without_commit'] += 1
            continue
        
        # Get issue creation date
        issue_date = issue.get('created_at') or issue.get('first_question', {}).get('created_at')
        if not issue_date:
            stats['without_commit'] += 1
            continue
        
        # Find commit before issue date
        commit_sha = get_commit_before_date(commits_data, issue_date)
        if commit_sha:
            issue['commit_id'] = commit_sha
            stats['with_commit'] += 1
        else:
            stats['without_commit'] += 1
    
    # Save updated issues
    with open(output_file, 'w') as f:
        json.dump(issues, f, indent=2)
    
    return stats

def main():
    # Directories
    input_dir = Path("../recent_v2_June2025_Jan2026_scon")   # From Step 5
    output_dir = Path("../recent_v2_June2025_Jan2026_final")  # Final output
    commits_dir = Path("github_commits")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("MERGING COMMIT IDs INTO ISSUE FILES")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Commits: {commits_dir}")
    print()
    
    total_stats = {'total': 0, 'with_commit': 0, 'without_commit': 0}
    
    # Process each issue file
    for input_file in sorted(input_dir.glob("*_issues.json")):
        output_file = output_dir / input_file.name
        print(f"Processing {input_file.name}...")
        
        stats = process_issue_file(input_file, output_file, commits_dir)
        
        total_stats['total'] += stats['total']
        total_stats['with_commit'] += stats['with_commit']
        total_stats['without_commit'] += stats['without_commit']
        
        print(f"  {stats['total']} issues, {stats['with_commit']} with commit_id, {stats['without_commit']} without")
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total issues: {total_stats['total']}")
    print(f"With commit_id: {total_stats['with_commit']}")
    print(f"Without commit_id: {total_stats['without_commit']}")
    print(f"\nOutput saved to: {output_dir}")

if __name__ == "__main__":
    main()
