import csv
import requests
from urllib.parse import urlparse
import json
from datetime import datetime
import os
from dotenv import load_dotenv
import re
import time

# Load environment variables from .env file
load_dotenv()

# Get GitHub token from environment variables
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

# Regular expressions for detecting URLs, images, and videos
URL_PATTERN = r'https?://[^\s)]+' 
IMAGE_PATTERN = r'!\\$\\$.*?\\$\\$\\$(https?://[^\s)]+)\\$'
VIDEO_PATTERN = r'<video[^>]*>.*?</video>'

def contains_media(text):
    """Check if text contains URLs, images, or videos"""
    if text is None:
        return False

    # Check for URLs
    if re.search(URL_PATTERN, text):
        return True

    # Check for markdown images
    if re.search(IMAGE_PATTERN, text):
        return True

    # Check for video tags
    if re.search(VIDEO_PATTERN, text):
        return True

    return False

def parse_github_url(url):
    """Parse GitHub URL to extract owner and repo name"""
    path_parts = urlparse(url).path.strip('/').split('/')
    if len(path_parts) >= 2:
        return path_parts[0], path_parts[1]
    raise ValueError("Invalid GitHub URL format")

def get_issues(owner, repo, label=None):
    """Fetch closed issues from GitHub API, optionally filtered by label"""
    base_url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    issues = []
    page = 1

    params = {
        'state': 'closed',
        'page': page,
        'per_page': 100
    }
    
    if label:
        params['labels'] = label

    headers = {
        'Accept': 'application/vnd.github.v3+json'
    }

    if GITHUB_TOKEN:
        headers['Authorization'] = f'token {GITHUB_TOKEN}'
    else:
        print("Warning: No GitHub token found in .env file. Rate limits will be restricted.")

    while True:
        params['page'] = page
        response = requests.get(base_url, params=params, headers=headers)
        
        if response.status_code == 403 and 'API rate limit exceeded' in response.json().get('message', ''):
            print("Rate limit exceeded. Waiting for 60 seconds before retrying...")
            time.sleep(60)
            continue
        
        if response.status_code != 200:
            label_info = f" with label '{label}'" if label else ""
            print(f"Error fetching issues{label_info}: {response.status_code}")
            print(response.json())
            break
            
        current_issues = response.json()
        if not current_issues:
            break
            
        issues.extend(current_issues)
        label_info = f" with label '{label}'" if label else ""
        print(f"Processing closed issues{label_info}: page {page}, found {len(current_issues)} issues")
        page += 1

    return issues

def format_issue(issue):
    """Format issue data and check for multiple authors"""
    issue_author = issue['user']['login']

    return {
        'number': issue['number'],
        'title': issue['title'],
        'created_at': issue['created_at'],
        'closed_at': issue['closed_at'],
        'labels': [label['name'] for label in issue['labels']],
        'url': issue['html_url'],
        'body': issue['body'],
        'comments_url': issue['comments_url'],
        'author': issue_author
    }

def get_comments(comments_url):
    """Fetch comments for an issue"""
    headers = {
        'Accept': 'application/vnd.github.v3+json'
    }

    if GITHUB_TOKEN:
        headers['Authorization'] = f'token {GITHUB_TOKEN}'

    while True:
        response = requests.get(comments_url, headers=headers)
        if response.status_code == 403 and 'API rate limit exceeded' in response.json().get('message', ''):
            print("Rate limit exceeded. Waiting for 60 seconds before retrying...")
            time.sleep(60)
            continue
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching comments: {response.status_code}")
            print(response.json())
            return []
        
def has_multiple_authors(issue_data, comments):
    """Check if the issue has multiple authors"""
    issue_author = issue_data['author']
    comment_authors = {comment['user']['login'] for comment in comments}
    all_authors = {issue_author}.union(comment_authors)
    return len(all_authors) > 1

def has_media_content(issue_data, comments):
    """Check if issue or comments contain media content"""
    # Check issue body
    if contains_media(issue_data['body']):
        return True

    # Check issue title
    if contains_media(issue_data['title']):
        return True

    # Check comments
    for comment in comments:
        if contains_media(comment['body']):
            return True

    return False

def save_processed_issues(owner, repo, processed_issues):
    """Save processed issues to a JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"github_issues_{owner}_{repo}_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(processed_issues, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(processed_issues)} processed issues to {filename}")
    return filename

def process_repo(owner, repo, question_issues, help_wanted_issues, use_labels=False):
    all_issues = []
    
    if use_labels:
        # Approach 2: Get issues with specific labels
        labels = ['question', 'help wanted']
        for label in labels:
            print(f"\nFetching issues with label '{label}'...")
            issues = get_issues(owner, repo, label)
            all_issues.extend(issues)
            print(f"Found {len(issues)} issues with label '{label}'")
        
        # De-duplicate issues that might have multiple labels
        unique_issues = list({issue['id']: issue for issue in all_issues}.values())
        print(f"\nFound {len(unique_issues)} unique issues with specified labels")
        all_issues = unique_issues
    else:
        # Approach 1: Get all closed issues
        print(f"\nFetching all closed issues...")
        all_issues = get_issues(owner, repo)
        print(f"Found {len(all_issues)} closed issues")
    
    if not all_issues:
        print("\nNo closed issues found.")
        return 0, []
        
    print(f"\nProcessing {len(all_issues)} issues...")
    
    processed_issues = []
    excluded_single_author = 0
    excluded_media = 0
    
    for i, issue in enumerate(all_issues, 1):
        processed_issue = format_issue(issue)
        
        print(f"Processing issue {i}/{len(all_issues)}: #{issue['number']}")
        comments = get_comments(issue['comments_url'])
        
        # Check for multiple authors and media content
        if not has_multiple_authors(processed_issue, comments):
            print(f"Skipping issue #{issue['number']} - single author")
            excluded_single_author += 1
            continue
            
        if has_media_content(processed_issue, comments):
            print(f"Skipping issue #{issue['number']} - contains media content")
            excluded_media += 1
            continue
        
        processed_issue['comments'] = [
            {
                'user': comment['user']['login'],
                'created_at': comment['created_at'],
                'body': comment['body']
            } for comment in comments
        ]
        processed_issues.append(processed_issue)
        print(f"Added issue #{issue['number']} - valid question/answer interaction")
    
    print(f"\nExcluded {excluded_single_author} issues with single author")
    print(f"Excluded {excluded_media} issues containing media content")
    
    return len(processed_issues), processed_issues

def main():
    csv_file = input("Enter the path to the CSV file: ")
    use_labels_input = input("Use label-based filtering? (y/n): ").lower()
    use_labels = (use_labels_input == 'y' or use_labels_input == 'yes')
    
    print(f"Using {'label-based' if use_labels else 'all closed issues'} approach")

    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            name = row['name']
            url = row['url']
            stars = row['stars']
            description = row['description']
            question_issues = int(row['question_issues'])
            help_wanted_issues = int(row['help_wanted_issues'])
            original_community_score = int(row['community_score'])

            print(f"\nProcessing repository: {name}")
            print(f"URL: {url}")
            print(f"Stars: {stars}")
            print(f"Description: {description}")
            print(f"Original community score: {original_community_score}")

            try:
                owner, repo = parse_github_url(url)
                processed_issues_count, processed_issues = process_repo(
                    owner, repo, question_issues, help_wanted_issues, use_labels
                )
                processed_community_score = processed_issues_count

                print(f"Processed community score: {processed_community_score}")
                print(f"Difference: {original_community_score - processed_community_score}")

                if processed_issues:
                    saved_file = save_processed_issues(owner, repo, processed_issues)
                    print(f"Valid Q&A interactions saved to: {saved_file}")
                else:
                    print("No valid Q&A interactions found for this repository.")

            except ValueError as e:
                print(f"Error: {str(e)}")
            except Exception as e:
                print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()