import requests
import pandas as pd
from dotenv import load_dotenv
import os
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from datetime import datetime
import time
from typing import Dict, List
import urllib.parse

# Allowed licenses dictionary
ALLOWED_LICENSES = {
    'apache-2.0': 'Apache License 2.0',
    'zlib': 'Zlib/libpng License',
    'isc': 'ISC License',
    'mit': 'The MIT License',
    'cc-by': 'Creative Commons Attribution',
    'cc0-1.0': 'Creative Commons 1.0 Universal'
}

class GithubRepoAnalyzer:
    def __init__(self):
        load_dotenv()
        self.token = os.getenv('GITHUB_TOKEN')
        if not self.token:
            raise ValueError("GitHub token not found in .env file")
        
        self.headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.base_url = "https://api.github.com"
        
        # Setup session with retry logic
        self.session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        self.session.headers.update(self.headers)

    def print_rate_limit_info(self):
        """Print current rate limit status"""
        limits = self.check_rate_limit()
        core = limits['resources']['core']
        print(f"\nAPI Rate Limit Status:")
        print(f"Remaining calls: {core['remaining']}/{core['limit']}")
        reset_time = datetime.fromtimestamp(core['reset'])
        print(f"Reset time: {reset_time}")

    def check_rate_limit(self) -> Dict:
        """Check and return current rate limits"""
        try:
            response = self.session.get(f"{self.base_url}/rate_limit")
            response.raise_for_status()
            limits = response.json()
            
            core_limits = limits['resources']['core']
            remaining = core_limits['remaining']
            reset_time = datetime.fromtimestamp(core_limits['reset'])
            
            print(f"\nAPI Rate Limits:")
            print(f"Remaining: {remaining}")
            print(f"Reset Time: {reset_time}")
            
            # If we're out of requests, wait until reset
            if remaining == 0:
                wait_time = max(core_limits['reset'] - time.time(), 0)
                print(f"\nRate limit exceeded. Waiting {wait_time:.0f} seconds...")
                time.sleep(wait_time + 1)  # Add 1 second buffer
                return self.check_rate_limit()  # Recursively check again
                
            return limits
        except requests.exceptions.RequestException as e:
            print(f"Error checking rate limits: {e}")
            return {'resources': {'core': {'remaining': 0, 'reset': time.time() + 60}}}

    def make_request_with_retry(self, url: str, params: Dict = None) -> Dict:
        """Make a request with rate limit handling and exponential backoff"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = self.session.get(url, params=params)
                
                # If we hit the rate limit
                if response.status_code == 403 and 'X-RateLimit-Remaining' in response.headers:
                    if int(response.headers['X-RateLimit-Remaining']) == 0:
                        reset_time = int(response.headers['X-RateLimit-Reset'])
                        wait_time = max(reset_time - time.time(), 0)
                        print(f"\nRate limit exceeded. Waiting {wait_time:.0f} seconds...")
                        time.sleep(wait_time + 1)
                        continue
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                retry_count += 1
                if retry_count == max_retries:
                    raise e
                time.sleep(2 ** retry_count)  # Exponential backoff
                continue

    def get_repo_license(self, repo_name: str) -> Dict:
        """Get repository license information"""
        try:
            data = self.make_request_with_retry(f"{self.base_url}/repos/{repo_name}/license")
            return data
        except requests.exceptions.RequestException:
            return None

    def get_issues_metrics_efficient(self, repo_name: str) -> Dict[str, int]:
        """Get issues metrics using direct issues API, focusing on closed issues only"""
        metrics = {
            'question_issues': 0,
            'help_wanted_issues': 0,
            'total_closed_issues': 0
        }
        
        # Get total closed issues count
        try:
            # We need to query directly for closed issues
            url = f"{self.base_url}/repos/{repo_name}/issues"
            params = {
                'state': 'closed',
                'per_page': 100,
                'page': 1
            }
            
            total_closed_count = 0
            while True:
                response = self.make_request_with_retry(url, params)
                count = len(response)
                total_closed_count += count
                
                if not response or len(response) < 100:
                    break
                
                params['page'] += 1
                # Add a small delay to avoid rate limiting
                time.sleep(0.5)
            
            metrics['total_closed_issues'] = total_closed_count
            
        except requests.exceptions.RequestException as e:
            print(f"\nError fetching closed issues for {repo_name}: {e}")
        
        # Get labeled closed issues
        labels = ['question', 'help wanted']
        for label in labels:
            try:
                url = f"{self.base_url}/repos/{repo_name}/issues"
                params = {
                    'state': 'closed',  # Specifically look for closed issues
                    'labels': label,
                    'per_page': 100,
                    'page': 1
                }
                
                total_count = 0
                while True:
                    response = self.make_request_with_retry(url, params)
                    count = len(response)
                    total_count += count
                    
                    if not response or len(response) < 100:
                        break
                    
                    params['page'] += 1
                    time.sleep(0.5)
                
                if label == 'question':
                    metrics['question_issues'] = total_count
                else:
                    metrics['help_wanted_issues'] = total_count
                
            except requests.exceptions.RequestException as e:
                print(f"\nError fetching closed issues for {repo_name} with label {label}: {e}")
        
        return metrics

    def get_top_repos(self, language: str, k: int = 500, created_after: str = "2024-11-01") -> List[Dict]:
        """Get top repositories based on stars and then filter by community metrics, ensuring k final results"""
        search_query = f"language:{language} stars:>10 created:>{created_after}"
        filtered_repos = []
        page = 1
        
        try:
            with tqdm(total=k, desc=f"Finding {k} qualifying {language} repositories") as pbar:
                while len(filtered_repos) < k:
                    # Check if we need to stop due to API limits
                    limits = self.check_rate_limit()
                    if limits['resources']['core']['remaining'] < 10:
                        print("\nApproaching API rate limit, pausing collection")
                        break
                    
                    # Fetch a batch of repositories
                    search_results = self.make_request_with_retry(
                        f"{self.base_url}/search/repositories",
                        params={
                            'q': search_query,
                            'sort': 'stars',
                            'order': 'desc',
                            'per_page': 100,
                            'page': page
                        }
                    )
                    
                    repos = search_results['items']
                    
                    if page == 1:
                        print(f"\nFound {search_results['total_count']} repositories matching the criteria")
                    
                    if not repos:
                        print("\nNo more repositories available matching the criteria")
                        break
                    
                    # Process each repo in the current batch
                    for repo in repos:
                        repo_name = repo['full_name']
                        
                        # Check license
                        license_info = self.get_repo_license(repo_name)
                        if not license_info:
                            continue
                            
                        license_key = license_info.get('license', {}).get('spdx_id', '').lower()
                        
                        if not any(lic.lower() in license_key for lic in ALLOWED_LICENSES.keys()):
                            continue
                        
                        time.sleep(0.5)  # Small delay to avoid rate limiting
                        
                        # Get issues metrics
                        issues_metrics = self.get_issues_metrics_efficient(repo_name)
                        
                        # Add repo info with all metrics
                        repo_info = {
                            'name': repo_name,
                            'url': repo['html_url'],
                            'stars': repo['stargazers_count'],
                            'description': repo['description'],
                            'created_at': datetime.strptime(repo['created_at'], 
                                                    '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d'),
                            'license': ALLOWED_LICENSES.get(license_key, license_info['license']['name']),
                            'question_issues': issues_metrics['question_issues'],
                            'help_wanted_issues': issues_metrics['help_wanted_issues'],
                            'total_closed_issues': issues_metrics['total_closed_issues'],
                            'community_score': issues_metrics['question_issues'] + issues_metrics['help_wanted_issues']
                        }
                        
                        filtered_repos.append(repo_info)
                        pbar.update(1)
                        
                        # Break once we have enough repos
                        if len(filtered_repos) >= k:
                            break
                    
                    page += 1
                    
                    # Add progress update
                    print(f"\nProcessed page {page-1}. Found {len(filtered_repos)}/{k} qualifying repositories so far.")
                    
            return filtered_repos
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching repositories: {e}")
            return filtered_repos

def get_popular_languages() -> List[str]:
    """Returns a list of popular programming languages"""
    return [
        "Python", "JavaScript", "Java", "C", "C++", "C#", "TypeScript"
    ]

def validate_date_format(date_string: str) -> bool:
    """Validate if the date string is in YYYY-MM-DD format"""
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def main():
    try:
        # Get language selection from user
        languages = get_popular_languages()
        print("Available languages:")
        for i, lang in enumerate(languages, 1):
            print(f"{i}. {lang}")
        
        lang_choice = int(input("\nEnter the number of your chosen language: "))
        if not 1 <= lang_choice <= len(languages):
            raise ValueError(f"Please enter a number between 1 and {len(languages)}")
        selected_language = languages[lang_choice - 1]
        
        k = int(input("\nEnter the number of repositories to analyze (recommended: 500 for top repos): "))
        if k <= 0:
            raise ValueError("Please enter a positive number")
        
        while True:
            created_after = input("\nEnter the minimum creation date (YYYY-MM-DD) or press Enter for default (2024-11-01): ") or "2024-11-01"
            if validate_date_format(created_after):
                break
            print("Invalid date format. Please use YYYY-MM-DD format.")
        
        analyzer = GithubRepoAnalyzer()
        analyzer.print_rate_limit_info()
        print(f"\nAnalyzing repositories created after {created_after}...")
        repos = analyzer.get_top_repos(selected_language, k, created_after)
        
        if not repos:
            print("No repositories found matching the criteria.")
            return
        
        df = pd.DataFrame(repos)
        df = df.sort_values(['community_score', 'total_closed_issues', 'stars'], 
                          ascending=[False, False, False])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'{selected_language.lower()}_repos_analysis_{timestamp}.csv'
        df.to_csv(output_file, index=False)
        
        print(f"\nTop 5 {selected_language} Repositories by Community Score and Total Closed Issues (out of {len(repos)} analyzed):")
        for idx, repo in df.head().iterrows():
            print(f"\n{idx + 1}. {repo['name']}")
            print(f"   Created: {repo['created_at']}")
            print(f"   Stars: {repo['stars']:,}")
            print(f"   License: {repo['license']}")
            print(f"   Community Score: {repo['community_score']}")
            print(f"   Total Closed Issues: {repo['total_closed_issues']:,}")
            print(f"   - Question Issues (closed): {repo['question_issues']}")
            print(f"   - Help Wanted Issues (closed): {repo['help_wanted_issues']}")
            print(f"   URL: {repo['url']}")
        
        print(f"\nFull results saved to '{output_file}'")
        
    except (ValueError, requests.exceptions.RequestException) as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")

if __name__ == "__main__":
    main()