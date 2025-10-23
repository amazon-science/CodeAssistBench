#!/usr/bin/env python3
import json
import os
import requests
from datetime import datetime
import time
import re
from typing import Dict, List, Optional
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue

class GitHubCommitUpdater:
    def __init__(self, github_token: Optional[str] = None, max_threads: int = 10):
        self.github_token = github_token
        self.max_threads = max_threads
        self.session = requests.Session()
        if github_token:
            self.session.headers.update({'Authorization': f'token {github_token}'})
        self.api_calls = 0
        self.rate_limit_remaining = 5000
        self.lock = threading.Lock()
        
    def parse_filename(self, filename: str) -> tuple:
        """Parse GitHub issues filename to extract owner and repo."""
        # Format: github_issues_{owner}_{repo}_{timestamp}.json
        match = re.match(r'github_issues_(.+)_([^_]+)_\d{8}_\d{6}\.json$', filename)
        if match:
            owner = match.group(1)
            repo = match.group(2)
            return owner, repo
        return None, None
    
    def get_last_commit_before_date(self, owner: str, repo: str, created_at: str) -> Optional[str]:
        """Get the last commit before the given date (thread-safe)."""
        try:
            # Parse the created_at date
            issue_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            # Format for GitHub API (ISO format)
            until_date = issue_date.strftime('%Y-%m-%dT%H:%M:%SZ')
            
            # GitHub API endpoint for commits
            url = f'https://api.github.com/repos/{owner}/{repo}/commits'
            params = {
                'until': until_date,
                'per_page': 1
            }
            
            # Create session per thread for thread safety
            session = requests.Session()
            if self.github_token:
                session.headers.update({'Authorization': f'token {self.github_token}'})
            
            response = session.get(url, params=params)
            
            # Thread-safe API call counting
            with self.lock:
                self.api_calls += 1
                if 'X-RateLimit-Remaining' in response.headers:
                    self.rate_limit_remaining = int(response.headers['X-RateLimit-Remaining'])
            
            if response.status_code == 200:
                commits = response.json()
                if commits:
                    return commits[0]['sha']
                else:
                    return None
            elif response.status_code == 404:
                return None
            elif response.status_code == 403:
                return None
            else:
                return None
                
        except Exception as e:
            return None

    def fetch_commit_for_issue(self, issue_data: tuple) -> tuple:
        """Fetch commit ID for a single issue (designed for threading)."""
        owner, repo, issue, issue_index = issue_data
        
        # Skip if commit_id already exists and is not None
        if 'commit_id' in issue and issue['commit_id'] is not None:
            return issue_index, issue['commit_id'], True  # already exists
        
        created_at = issue.get('created_at')
        if not created_at:
            return issue_index, None, False
        
        commit_id = self.get_last_commit_before_date(owner, repo, created_at)
        return issue_index, commit_id, False  # newly fetched
    
    def insert_commit_id_in_order(self, issue: dict, commit_id: Optional[str]):
        """Insert commit_id field after closed_at and before labels."""
        # Define desired field order: number, title, created_at, closed_at, commit_id, labels, url, body, comments_url, author, comments, satisfaction_conditions, _classification
        desired_order = [
            'number', 'title', 'created_at', 'closed_at', 'commit_id', 'labels', 
            'url', 'body', 'comments_url', 'author', 'comments', 
            'satisfaction_conditions', '_classification'
        ]
        
        # Create new ordered dictionary
        ordered_issue = {}
        
        # First, add fields in desired order if they exist
        for field in desired_order:
            if field == 'commit_id':
                ordered_issue[field] = commit_id
            elif field in issue:
                ordered_issue[field] = issue[field]
        
        # Add any remaining fields that weren't in the desired order
        for key, value in issue.items():
            if key not in ordered_issue:
                ordered_issue[key] = value
        
        # Replace issue content with ordered content
        issue.clear()
        issue.update(ordered_issue)
    
    def create_updated_dataset(self, source_dir: str = "dataset/recent", target_dir: str = "dataset/updated"):
        """Create updated dataset with commit IDs in new directory using threading."""
        print("=== CREATING UPDATED DATASET WITH COMMIT IDs (THREADED) ===\n")
        print(f"Using {self.max_threads} threads for parallel processing")
        
        # Create target directory structure (preserve existing if it has partial updates)
        if os.path.exists(target_dir):
            print(f"Target directory {target_dir} already exists. Checking for existing updates...")
        else:
            print(f"Creating new target directory {target_dir}...")
        
        os.makedirs(target_dir, exist_ok=True)
        
        total_files = 0
        updated_files = 0
        total_issues = 0
        updated_issues = 0
        
        for language_dir in sorted(os.listdir(source_dir)):
            lang_source_path = os.path.join(source_dir, language_dir)
            if not os.path.isdir(lang_source_path):
                continue
                
            # Create corresponding language directory in target
            lang_target_path = os.path.join(target_dir, language_dir)
            os.makedirs(lang_target_path, exist_ok=True)
            
            print(f"Processing {language_dir} directory...")
            
            for filename in sorted(os.listdir(lang_source_path)):
                if not filename.endswith('.json'):
                    continue
                    
                total_files += 1
                source_file_path = os.path.join(lang_source_path, filename)
                target_file_path = os.path.join(lang_target_path, filename)
                
                # Parse repository information from filename
                owner, repo = self.parse_filename(filename)
                if not owner or not repo:
                    print(f"  Skipped {filename}: Could not parse owner/repo")
                    # Copy file as-is if we can't parse it
                    shutil.copy2(source_file_path, target_file_path)
                    continue
                
                print(f"  Processing {filename} ({owner}/{repo})...")
                
                try:
                    # Check if target file already exists and load it
                    if os.path.exists(target_file_path):
                        print(f"    Target file exists, checking for existing commit IDs...")
                        with open(target_file_path, 'r', encoding='utf-8') as f:
                            issues_data = json.load(f)
                        print(f"    Loaded existing updated file")
                    else:
                        # Load from source if target doesn't exist
                        with open(source_file_path, 'r', encoding='utf-8') as f:
                            issues_data = json.load(f)
                        print(f"    Loaded from source file")
                    
                    # Prepare issues that need commits for threading
                    issues_to_process = []
                    issues_needing_commits = 0
                    
                    for index, issue in enumerate(issues_data):
                        total_issues += 1
                        if 'commit_id' not in issue or issue['commit_id'] is None:
                            issues_needing_commits += 1
                            issues_to_process.append((owner, repo, issue, index))
                    
                    if issues_needing_commits == 0:
                        print(f"    All issues already have commit IDs, skipping API calls")
                        # Still save to ensure file is in target location
                        with open(target_file_path, 'w', encoding='utf-8') as f:
                            json.dump(issues_data, f, indent=2, ensure_ascii=False)
                        continue
                    else:
                        print(f"    {issues_needing_commits} issues need commit IDs, processing with {self.max_threads} threads...")
                    
                    # Process issues in parallel using ThreadPoolExecutor
                    file_updated = False
                    with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                        # Submit all issues for processing
                        future_to_issue = {
                            executor.submit(self.fetch_commit_for_issue, issue_data): issue_data 
                            for issue_data in issues_to_process
                        }
                        
                        # Process results as they complete
                        for future in as_completed(future_to_issue):
                            try:
                                issue_index, commit_id, already_existed = future.result()
                                
                                # Update the issue with the commit ID in proper position
                                self.insert_commit_id_in_order(issues_data[issue_index], commit_id)
                                
                                if not already_existed:
                                    file_updated = True
                                    if commit_id:
                                        updated_issues += 1
                                        print(f"      Issue #{issues_data[issue_index].get('number', 'unknown')}: {commit_id[:8]}...")
                                    else:
                                        print(f"      Issue #{issues_data[issue_index].get('number', 'unknown')}: No commit found")
                                
                            except Exception as e:
                                print(f"      Error processing issue: {e}")
                    
                    # Save updated data to target directory with proper field ordering
                    with open(target_file_path, 'w', encoding='utf-8') as f:
                        json.dump(issues_data, f, indent=2, ensure_ascii=False)
                    
                    if file_updated:
                        updated_files += 1
                        print(f"    Updated and saved {filename}")
                    else:
                        print(f"    No updates needed for {filename}")
                    
                except Exception as e:
                    print(f"  Error processing {filename}: {e}")
                    # Try to copy original file in case of error
                    try:
                        shutil.copy2(source_file_path, target_file_path)
                        print(f"    Copied original {filename} due to error")
                    except:
                        print(f"    Failed to copy {filename}")
        
        print(f"\n=== UPDATE SUMMARY ===")
        print(f"Source directory: {source_dir}")
        print(f"Target directory: {target_dir}")
        print(f"Total files processed: {total_files}")
        print(f"Files with updates: {updated_files}")
        print(f"Total issues: {total_issues}")
        print(f"Issues updated with commit IDs: {updated_issues}")
        print(f"API calls made: {self.api_calls}")
        print(f"Rate limit remaining: {self.rate_limit_remaining}")
        print(f"Threads used: {self.max_threads}")

def main():
    parser = argparse.ArgumentParser(description='Update GitHub issues dataset with commit IDs')
    parser.add_argument('--github-token', '-t', 
                       help='GitHub personal access token for API authentication')
    parser.add_argument('--source-dir', default='dataset/recent',
                       help='Source dataset directory (default: dataset/recent)')
    parser.add_argument('--target-dir', default='dataset/updated',
                       help='Target directory for updated dataset (default: dataset/updated)')
    parser.add_argument('--threads', '-j', type=int, default=10,
                       help='Number of threads for parallel processing (default: 10)')
    
    args = parser.parse_args()
    
    print("GitHub Dataset Commit ID Updater")
    print("================================")
    print(f"Source: {args.source_dir}")
    print(f"Target: {args.target_dir}")
    print(f"Threads: {args.threads}")
    print("This will create a new dataset with commit IDs added.\n")
    
    github_token = args.github_token
    if not github_token:
        # Try environment variable as fallback
        github_token = os.getenv('GITHUB_TOKEN')
    
    if not github_token:
        print("Warning: No GitHub token provided.")
        print("API calls will be limited to 60 per hour without authentication.")
        print("Use --github-token argument or set GITHUB_TOKEN environment variable.")
        print("Proceeding without authentication...\n")
    else:
        print("Using GitHub token for API authentication.\n")
    
    updater = GitHubCommitUpdater(github_token, args.threads)
    updater.create_updated_dataset(args.source_dir, args.target_dir)


if __name__ == "__main__":
    main()
