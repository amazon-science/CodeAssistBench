import requests
from datetime import datetime
import os
from typing import List, Dict
import time
import json
from pathlib import Path
from dotenv import load_dotenv

class GitHubAPIClient:
    def __init__(self):
        """Initialize GitHub API client using token from .env file."""
        load_dotenv()  # Load environment variables from .env file
        self.token = os.getenv('GITHUB_TOKEN')
        if not self.token:
            raise ValueError("GITHUB_TOKEN not found in .env file")
        
        self.headers = {
            'Accept': 'application/vnd.github.v3+json',
            'Authorization': f'token {self.token}'
        }

    def get_commits_before_date(self, repo_url: str, before_date: str) -> List[Dict]:
        """Get all commits before a specific date."""
        try:
            # Extract owner and repo from URL
            repo_path = repo_url.split('github.com/')[-1].rstrip('/')
            
            api_url = f"https://api.github.com/repos/{repo_path}/commits"
            params = {
                'until': before_date,
                'per_page': 100
            }
            
            all_commits = []
            page = 1
            
            while True:
                params['page'] = page
                response = requests.get(api_url, headers=self.headers, params=params)
                
                if response.status_code == 403 and 'rate limit exceeded' in response.text.lower():
                    reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                    wait_time = max(reset_time - time.time(), 0)
                    print(f"Rate limit exceeded. Waiting {wait_time:.0f} seconds...")
                    time.sleep(wait_time + 1)
                    continue
                
                if response.status_code == 404:
                    print(f"Repository not found: {repo_url}")
                    return []
                    
                response.raise_for_status()
                commits = response.json()
                
                if not commits:
                    break
                    
                for commit in commits:
                    commit_info = {
                        'sha': commit['sha'],
                        'date': commit['commit']['author']['date'],
                        'message': commit['commit']['message'],
                        'author': commit['commit']['author']['name']
                    }
                    all_commits.append(commit_info)
                
                if len(commits) < 100:
                    break
                    
                page += 1
                time.sleep(1)  # Be nice to GitHub API
            
            return all_commits
            
        except Exception as e:
            print(f"Error fetching commits for {repo_url}: {str(e)}")
            return []

def process_repository_file(file_path: Path, output_dir: Path, client: GitHubAPIClient) -> None:
    """Process a single repository JSON file that may contain issues from multiple repositories."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            issues = json.load(f)
        
        # Ensure issues is a list
        if not isinstance(issues, list):
            issues = [issues]
        
        # Group issues by repository
        repo_issues = {}
        for issue in issues:
            if issue.get('url'):
                # Extract repository URL: https://github.com/owner/repo/issues/123 -> https://github.com/owner/repo
                repo_url = '/'.join(issue['url'].split('/')[:5])
                if repo_url not in repo_issues:
                    repo_issues[repo_url] = []
                repo_issues[repo_url].append(issue)
        
        if not repo_issues:
            print(f"No valid repository URLs found in {file_path}")
            return
        
        # Process each repository separately
        for repo_url, repo_issue_list in repo_issues.items():
            # Get the latest issue date for this repository
            latest_date = max(
                datetime.strptime(issue.get('created_at', '0001-01-01T00:00:00Z'), "%Y-%m-%dT%H:%M:%SZ")
                for issue in repo_issue_list
            )
            
            # Create safe filename from repo URL
            # Example: https://github.com/owner/repo -> owner_repo
            repo_name = repo_url.replace('https://github.com/', '').replace('/', '_')
            output_file = output_dir / f"commits_{file_path.stem}_{repo_name}.json"
            
            # Skip if already processed
            if output_file.exists():
                print(f"Skipping {repo_url} - already processed")
                continue
            
            # Get commits before the latest issue date
            commits = client.get_commits_before_date(
                repo_url,
                latest_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            )
            
            # Save results
            result = {
                'repository': repo_url,
                'issue_count': len(repo_issue_list),
                'latest_issue_date': latest_date.isoformat(),
                'total_commits': len(commits),
                'commits': commits
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
                
            print(f"Processed {repo_url}: {len(commits)} commits found (from {len(repo_issue_list)} issues)")
        
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")

def main():
    """Process all JSON files in the input directory."""
    # Setup directories
    input_dir = Path("../recent_v2_June2025_Jan2026_scon")  # Input: from scon_filter.py (Step 5)
    output_dir = Path("github_commits")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize GitHub API client
    client = GitHubAPIClient()
    
    # Get list of all files to process (exclude debug/prompt files)
    all_files = [f for f in input_dir.glob("*_issues.json") if "prompts_responses" not in f.name]
    total_files = len(all_files)
    
    # Count already processed files
    already_processed = sum(1 for file in all_files if (output_dir / f"commits_{file.stem}.json").exists())
    
    print(f"Found {total_files} total files, {already_processed} already processed")
    
    # Process all JSON files
    for i, file_path in enumerate(all_files):
        output_file = output_dir / f"commits_{file_path.stem}.json"
        
        # if output_file.exists():
        #     print(f"[{i+1}/{total_files}] Skipping {file_path.name} - already processed")
        #     continue
            
        print(f"\n[{i+1}/{total_files}] Processing file: {file_path}")
        process_repository_file(file_path, output_dir, client)
        
    print(f"\nProcessing complete. Results saved in {output_dir}")

if __name__ == "__main__":
    main()
