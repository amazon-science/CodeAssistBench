#!/usr/bin/env python3
"""
Comprehensive GitHub Issue Collector
Collects target number of issues per language from GitHub repositories.
"""

import os
import json
import time
import requests
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN not found in .env file")

HEADERS = {
    'Accept': 'application/vnd.github.v3+json',
    'Authorization': f'token {GITHUB_TOKEN}'
}

# Allowed licenses
ALLOWED_LICENSES = {
    'apache-2.0', 'zlib', 'isc', 'mit', 'cc-by', 'cc0-1.0', 
    'bsd-2-clause', 'bsd-3-clause', 'unlicense', 'lgpl-2.1', 'lgpl-3.0'
}

# Media detection patterns
URL_PATTERN = r'https?://[^\s)]+' 
IMAGE_PATTERN = r'!\[.*?\]\(https?://[^\s)]+\)'
VIDEO_PATTERN = r'<video[^>]*>.*?</video>'

def rate_limit_wait():
    """Check rate limit and wait if necessary."""
    response = requests.get('https://api.github.com/rate_limit', headers=HEADERS)
    if response.status_code == 200:
        data = response.json()
        remaining = data['resources']['core']['remaining']
        reset_time = data['resources']['core']['reset']
        
        if remaining < 10:
            wait_time = max(reset_time - time.time() + 5, 0)
            print(f"Rate limit low ({remaining}). Waiting {wait_time:.0f}s...")
            time.sleep(wait_time)
            return rate_limit_wait()
        return remaining
    return 0

def contains_media(text: str) -> bool:
    """Check if text contains URLs, images, or videos."""
    if text is None:
        return False
    if re.search(IMAGE_PATTERN, text):
        return True
    if re.search(VIDEO_PATTERN, text):
        return True
    return False

def search_repos_with_issues(language: str, min_stars: int = 10, created_after: str = "2024-06-01", 
                             page: int = 1) -> List[Dict]:
    """Search for repositories with issues."""
    rate_limit_wait()
    
    query = f"language:{language} stars:>{min_stars} created:>{created_after}"
    url = f"https://api.github.com/search/repositories"
    params = {
        'q': query,
        'sort': 'stars',
        'order': 'desc',
        'per_page': 100,
        'page': page
    }
    
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code == 200:
        return response.json().get('items', [])
    elif response.status_code == 403:
        print(f"Rate limited on repo search, waiting...")
        time.sleep(60)
        return search_repos_with_issues(language, min_stars, created_after, page)
    else:
        print(f"Error searching repos: {response.status_code}")
        return []

def get_repo_license(repo_name: str) -> Optional[str]:
    """Get repository license."""
    rate_limit_wait()
    url = f"https://api.github.com/repos/{repo_name}/license"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        data = response.json()
        return data.get('license', {}).get('spdx_id', '').lower()
    return None

def get_closed_issues(repo_name: str, label: Optional[str] = None, page: int = 1) -> List[Dict]:
    """Get closed issues from a repository."""
    rate_limit_wait()
    
    url = f"https://api.github.com/repos/{repo_name}/issues"
    params = {
        'state': 'closed',
        'per_page': 100,
        'page': page
    }
    if label:
        params['labels'] = label
    
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code == 200:
        # Filter out pull requests (they come through issues API too)
        issues = [i for i in response.json() if 'pull_request' not in i]
        return issues
    elif response.status_code == 403:
        print(f"Rate limited, waiting...")
        time.sleep(60)
        return get_closed_issues(repo_name, label, page)
    return []

def get_issue_comments(comments_url: str) -> List[Dict]:
    """Get comments for an issue."""
    rate_limit_wait()
    response = requests.get(comments_url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 403:
        time.sleep(60)
        return get_issue_comments(comments_url)
    return []

def process_issue(issue: Dict, repo_name: str) -> Optional[Dict]:
    """Process and validate a single issue."""
    try:
        # Get issue author
        issue_author = issue['user']['login']
        
        # Get comments
        comments = get_issue_comments(issue['comments_url'])
        
        # Check for multiple authors
        comment_authors = {c['user']['login'] for c in comments if c.get('user')}
        all_authors = {issue_author} | comment_authors
        
        if len(all_authors) < 2:
            return None  # Single author, skip
        
        # Check for media content
        if contains_media(issue.get('body', '')):
            return None
        if contains_media(issue.get('title', '')):
            return None
        for comment in comments:
            if contains_media(comment.get('body', '')):
                return None
        
        # Format the issue
        processed = {
            'number': issue['number'],
            'title': issue['title'],
            'created_at': issue['created_at'],
            'closed_at': issue['closed_at'],
            'labels': [label['name'] for label in issue.get('labels', [])],
            'url': issue['html_url'],
            'body': issue.get('body', ''),
            'author': issue_author,
            'repo': repo_name,
            'comments': [
                {
                    'user': c['user']['login'],
                    'created_at': c['created_at'],
                    'body': c.get('body', '')
                }
                for c in comments if c.get('user')
            ]
        }
        
        return processed
        
    except Exception as e:
        print(f"Error processing issue: {e}")
        return None

def collect_issues_for_language(language: str, target_count: int = 1000, 
                                output_dir: str = "../recent_v2_June2025_Jan2026",
                                created_after: str = "2024-06-01") -> int:
    """Collect target number of issues for a language."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{language.lower()}_issues.json"
    
    # Load existing issues if any
    existing_issues = []
    processed_urls = set()
    if output_file.exists():
        try:
            with open(output_file, 'r') as f:
                existing_issues = json.load(f)
                processed_urls = {i['url'] for i in existing_issues}
                print(f"Loaded {len(existing_issues)} existing {language} issues")
        except:
            existing_issues = []
    
    all_issues = existing_issues.copy()
    processed_repos = set()
    repo_page = 1
    min_stars = 10
    
    print(f"\n{'='*60}")
    print(f"Collecting {target_count} issues for {language}")
    print(f"Starting from {len(all_issues)} existing issues")
    print(f"{'='*60}")
    
    while len(all_issues) < target_count:
        # Search for repos
        repos = search_repos_with_issues(language, min_stars, created_after, repo_page)
        
        if not repos:
            print(f"No more repos found. Reducing min_stars to {min_stars//2}")
            min_stars = max(min_stars // 2, 1)
            repo_page = 1
            if min_stars < 1:
                print("Cannot find more repositories")
                break
            continue
        
        for repo in repos:
            if len(all_issues) >= target_count:
                break
                
            repo_name = repo['full_name']
            
            if repo_name in processed_repos:
                continue
            processed_repos.add(repo_name)
            
            # Check license
            license_key = get_repo_license(repo_name)
            if license_key and not any(lic in license_key for lic in ALLOWED_LICENSES):
                continue
            
            print(f"\nProcessing {repo_name} ({len(all_issues)}/{target_count} issues)")
            
            # Collect issues (try both with labels and without)
            issue_page = 1
            repo_issue_count = 0
            
            while len(all_issues) < target_count:
                # Try question label first
                issues = get_closed_issues(repo_name, 'question', issue_page)
                
                # If no more labeled issues, get all closed issues
                if not issues and issue_page == 1:
                    issues = get_closed_issues(repo_name, None, issue_page)
                
                if not issues:
                    break
                
                for issue in issues:
                    if len(all_issues) >= target_count:
                        break
                    
                    # Skip if already processed
                    if issue['html_url'] in processed_urls:
                        continue
                    
                    # Process the issue
                    processed = process_issue(issue, repo_name)
                    
                    if processed:
                        all_issues.append(processed)
                        processed_urls.add(issue['html_url'])
                        repo_issue_count += 1
                        
                        # Save periodically
                        if len(all_issues) % 50 == 0:
                            with open(output_file, 'w') as f:
                                json.dump(all_issues, f, indent=2)
                            print(f"  Saved {len(all_issues)} issues")
                
                if len(issues) < 100:
                    break
                issue_page += 1
                time.sleep(0.5)
            
            if repo_issue_count > 0:
                print(f"  Found {repo_issue_count} valid issues from {repo_name}")
        
        repo_page += 1
        
        # Save after each batch of repos
        with open(output_file, 'w') as f:
            json.dump(all_issues, f, indent=2)
    
    # Final save
    with open(output_file, 'w') as f:
        json.dump(all_issues, f, indent=2)
    
    print(f"\n✅ Collected {len(all_issues)} {language} issues")
    return len(all_issues)

def main():
    """Main function to collect issues for all languages."""
    languages = ['Python', 'JavaScript', 'Java', 'C', 'C++', 'TypeScript', 'Go']
    target_per_language = 1000
    output_dir = "../recent_v2_June2025_Jan2026"
    created_after = "2024-06-01"  # June 2025 cutoff for recent issues
    
    print("="*60)
    print("GitHub Issue Collector")
    print(f"Target: {target_per_language} issues per language")
    print(f"Languages: {', '.join(languages)}")
    print(f"Created after: {created_after}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    # Check rate limit before starting
    remaining = rate_limit_wait()
    print(f"API calls remaining: {remaining}")
    
    results = {}
    for language in languages:
        try:
            count = collect_issues_for_language(
                language, 
                target_per_language, 
                output_dir,
                created_after
            )
            results[language] = count
        except KeyboardInterrupt:
            print(f"\nInterrupted. Progress saved.")
            break
        except Exception as e:
            print(f"Error collecting {language} issues: {e}")
            results[language] = 0
    
    # Print summary
    print("\n" + "="*60)
    print("COLLECTION SUMMARY")
    print("="*60)
    for lang, count in results.items():
        status = "✅" if count >= target_per_language else "⚠️"
        print(f"{status} {lang}: {count} issues")
    print("="*60)

if __name__ == "__main__":
    main()
