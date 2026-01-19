import json
import os
import boto3
import glob
from botocore.config import Config
import time
from collections import defaultdict
import re
import shutil
from datetime import datetime

# Configure path
ISSUES_DIR = "../recent_v2_June2025_Jan2026_final"  # Input: issues with commit_id from Step 6b
DOCKER_FILTER_DIR = "../recent_v2_June2025_Jan2026_classified"  # Output: classified issues
NEED_DOCKER_DIR = os.path.join(DOCKER_FILTER_DIR, "need_docker")
NO_NEED_DOCKER_DIR = os.path.join(DOCKER_FILTER_DIR, "no_need_docker")
NEED_DOCKER_BUT_CANNOT_DIR = os.path.join(DOCKER_FILTER_DIR, "need_docker_but_cannot")
LLM_RESPONSES_DIR = os.path.join(DOCKER_FILTER_DIR, "llm_responses")
PROCESSED_ISSUES_FILE = os.path.join(DOCKER_FILTER_DIR, "processed_issues.json")

# Define classification categories
CATEGORIES = [
    "Does not need build environment",  # No need Docker
    "Can be dockerized without any issue",  # Need Docker
    "Requires build environment but hard to be dockerized"  # Need Docker but Cannot
]

def init_bedrock_client():
    """Initialize the Bedrock client"""
    config = Config(
        retries = {
            "max_attempts": 10000,
            "mode": "standard"
        }
    )
    return boto3.client('bedrock-runtime', config=config, region_name='us-west-2')

def get_model_id():
    """Return DeepSeek model ID"""
    return "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

def generate_response(client, user_prompt, max_retries=10000):
    """Generate a response from the LLM"""
    model_id = get_model_id()
    system_prompt = """You are a helpful AI assistant who classifies GitHub issues according to predefined categories. Your goal is to determine if the GitHub issue can be reproduced to verify the provided solution's correctness."""
    
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "max_tokens": 10000,
        "temperature": 0.0
    }
    
    for attempt in range(max_retries):
        try:
            response = client.invoke_model(
                body=json.dumps(body),
                modelId=model_id,
                accept="application/json",
                contentType="application/json"
            )
            
            response_body = json.loads(response.get('body').read())
            content = response_body["content"][0]["text"]  # Updated response parsing
            
            return content, user_prompt, system_prompt
            
        except Exception as e:
            print(f"Error in API call (attempt {attempt+1}/{max_retries}): {str(e)}")
            time.sleep(2)  # Add a delay between retries
    
    return "Failed to get response after maximum retries", user_prompt, system_prompt

def save_llm_response(issue, prompt, response, system_prompt, repo_name, issue_number):
    """Save the LLM prompt and response to a file"""
    # Create directory if it doesn't exist
    os.makedirs(LLM_RESPONSES_DIR, exist_ok=True)
    
    # Create a filename based on repo, issue number, and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{repo_name}_issue{issue_number}_{timestamp}.json"
    filepath = os.path.join(LLM_RESPONSES_DIR, filename)
    
    # Save the prompt and response
    data = {
        "issue": {
            "number": issue_number,
            "title": issue.get("title", "Unknown"),
            "url": issue.get("url", "Unknown"),
        },
        "system_prompt": system_prompt,
        "user_prompt": prompt,
        "llm_response": response,
        "timestamp": timestamp
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    return filepath

def classify_issue(client, issue, json_file):
    """Classify a GitHub issue into one of the simplified exclusion categories with explanation"""
    
    # Collect all comments for context
    comments_text = ""
    if "comments" in issue and issue["comments"]:
        for comment in issue["comments"]:
            comments_text += f"\nComment by {comment.get('user', 'unknown')} at {comment.get('created_at', 'unknown')}:\n{comment.get('body', '')}\n"
    
    prompt = f"""Analyze the following GitHub issue and classify it into one of these categories:

PROJECT GOAL: We aim to verify if the user needs to set up a build environment to resolve the issue:

1. Does not need build environment
   (The solutions/answers can be validated without build environment, e.g., it's a conceptual issue, documentation issue, or can be verified with simple code analysis)

2. Can be dockerized without any issue
   (The solutions must require a build environment to be verified)

3. Requires build environment but hard to be dockerized or cannot be reproduced now
   (The solution must require a build environment, but it is difficult to containerize for reasons such as: hardware dependencies, network dependencies, external API access with authentication, timing-dependent behaviors, requiring MacOS, multiple users interacting simultaneously, licensed software, etc.)

IMPORTANT GUIDELINES:
- Choose category 1 whenever possible. If the LLM can analyze the issue and verify the solution without needing to build or run the code, always select category 1.
- Only classify an issue as requiring a build environment (categories 2 or 3) when it's ABSOLUTELY NECESSARY to actually build and run the code to verify if the solution works.

GitHub Issue:
Post Date: {issue['created_at']}
Current Date: April 4 2025
Title: {issue['title']}
Description: {issue['body']}

You must provide a category number (1-3). Your response must follow this format exactly:

CATEGORY: [category number]
REASONING: [your reasoning here, including specific reasons why an issue would be hard to dockerize if choosing category 3]
"""

    # Extract repo name and issue number for saving LLM response
    repo_name = get_repo_name_from_issue(issue)
    issue_number = issue.get("number", "unknown")
    
    # Make a single API call to get the response
    response, user_prompt, system_prompt = generate_response(client, prompt)
    
    # Save the LLM prompt and response
    response_file = save_llm_response(issue, user_prompt, response, system_prompt, repo_name, issue_number)
    print(f"  LLM response saved to {response_file}")
    
    # Try to extract the category
    category_match = re.search(r"CATEGORY:\s*(\d)", response, re.IGNORECASE)
    if category_match:
        try:
            category_num = int(category_match.group(1))
            if 1 <= category_num <= 3:
                # Try to find the reasoning
                reasoning_match = re.search(r"REASONING:\s*(.*?)(?=\$|\n\n|\Z)", response, re.IGNORECASE | re.DOTALL)
                reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
                
                return CATEGORIES[category_num-1], response, reasoning
        except (ValueError, IndexError) as e:
            print(f"  Error extracting category: {e}")
    
    # If we couldn't extract a proper category, make a conservative choice
    print(f"  Failed to extract valid category for issue {repo_name}#{issue_number}")
    print(f"  Using category 1 (Does not need build environment) as a conservative default")
    
    # Try to still extract any reasoning provided
    reasoning = "Category extraction failed. Making a conservative classification."
    
    # Use a more flexible regex to find anything that looks like reasoning
    reasoning_match = re.search(r"REASON(?:ING)?:\s*(.*?)(?=\$|\n\n|\Z)", response, re.IGNORECASE | re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    
    return CATEGORIES[0], response, reasoning

def get_repo_name_from_issue(issue):
    """Extract repository name from issue"""
    url = issue.get("url", "")
    if "/issues/" in url:
        repo = url.split("/issues/")[0]
        repo = repo.split("github.com/")[-1] if "github.com/" in repo else repo
        return repo.split("/")[-1] if "/" in repo else repo
    return "unknown"

def get_issue_identifier(issue, json_file):
    """Generate a unique identifier for an issue that combines repository and issue number"""
    repo = None
    issue_number = issue.get("number")
    
    # Extract repository from URL
    url = issue.get("url", "")
    if "/issues/" in url:
        repo = url.split("/issues/")[0]
        repo = repo.split("github.com/")[-1] if "github.com/" in repo else repo
    
    if repo and issue_number:
        return f"{repo}#{issue_number}"
    elif issue_number:
        # Use filename as fallback repo identifier
        file_name = os.path.basename(json_file)
        return f"{file_name}#{issue_number}"
    else:
        # Absolute fallback
        file_name = os.path.basename(json_file)
        return f"{file_name}#{hash(str(issue))}"

def load_processed_issues():
    """Load the set of already processed issue identifiers"""
    processed = set()
    
    if os.path.exists(PROCESSED_ISSUES_FILE):
        try:
            with open(PROCESSED_ISSUES_FILE, 'r') as f:
                processed_data = json.load(f)
                processed = set(processed_data.get("processed_issues", []))
            print(f"Loaded {len(processed)} issue identifiers from {PROCESSED_ISSUES_FILE}")
        except Exception as e:
            print(f"Error loading {PROCESSED_ISSUES_FILE}: {e}")
    
    return processed

def save_processed_issues(processed_issues):
    """Save the set of processed issue identifiers"""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(PROCESSED_ISSUES_FILE), exist_ok=True)
    
    with open(PROCESSED_ISSUES_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            "processed_issues": list(processed_issues)
        }, f, indent=2)
    
    print(f"Processed issues saved to {PROCESSED_ISSUES_FILE}")

def save_issue_to_category(issue, json_file, category):
    """Save the issue to the appropriate category directory"""
    file_name = os.path.basename(json_file)
    
    if category == CATEGORIES[0]:  # Does not need build environment
        target_dir = NO_NEED_DOCKER_DIR
    elif category == CATEGORIES[1]:  # Can be dockerized without any issue
        target_dir = NEED_DOCKER_DIR
    elif category == CATEGORIES[2]:  # Requires build environment but hard to be dockerized
        target_dir = NEED_DOCKER_BUT_CANNOT_DIR
    else:
        # This should never happen
        print(f"Error: Unknown category {category}")
        return None
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Define target file path
    target_file = os.path.join(target_dir, file_name)
    
    # Check if target file exists, create or update it
    issues_to_save = []
    
    if os.path.exists(target_file):
        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                issues_to_save = json.load(f)
        except Exception as e:
            print(f"Error loading {target_file}: {e}")
            issues_to_save = []
    
    # Add the current issue to the list with classification info
    issue["_classification"] = {
        "category": category,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    issues_to_save.append(issue)
    
    # Save the updated list to the target file
    with open(target_file, 'w', encoding='utf-8') as f:
        json.dump(issues_to_save, f, indent=2)
    
    return target_file

def main():
    # Initialize Bedrock client
    client = init_bedrock_client()
    
    # Create output directories if they don't exist
    os.makedirs(DOCKER_FILTER_DIR, exist_ok=True)
    os.makedirs(NEED_DOCKER_DIR, exist_ok=True)
    os.makedirs(NO_NEED_DOCKER_DIR, exist_ok=True)
    os.makedirs(NEED_DOCKER_BUT_CANNOT_DIR, exist_ok=True)
    os.makedirs(LLM_RESPONSES_DIR, exist_ok=True)
    
    # Get all JSON files in the directory (but not in subdirectories)
    json_files = [f for f in glob.glob(os.path.join(ISSUES_DIR, "*.json")) 
                 if os.path.isfile(f) and not f.endswith("processed_issues.json")]
    print(f"Found {len(json_files)} JSON files to process.")
    
    # Load processed issues
    processed_issues = load_processed_issues()
    
    # Track statistics
    category_counts = defaultdict(int)
    total_issues = 0
    
    for i, json_file in enumerate(json_files):
        file_name = os.path.basename(json_file)
        print(f"Processing file {i+1}/{len(json_files)}: {file_name}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                issues = json.load(f)
            
            for issue in issues:
                total_issues += 1
                
                # Generate a unique identifier for this issue
                issue_id = get_issue_identifier(issue, json_file)
                
                # Skip already processed issues
                if issue_id in processed_issues:
                    print(f"  Skipping already processed issue {issue_id}")
                    continue
                
                # Classify the issue
                category, explanation, reasoning = classify_issue(client, issue, json_file)
                
                # Update counts
                category_counts[category] += 1
                
                # Save issue to appropriate category folder
                target_file = save_issue_to_category(issue, json_file, category)
                if target_file:
                    print(f"  Saved issue to {target_file}")
                
                # Add to processed issues
                processed_issues.add(issue_id)
                
                # Print progress
                print(f"  Classified issue #{issue.get('number')} as: {category}")
                print(f"  Reason: {reasoning[:100]}..." if len(reasoning) > 100 else f"  Reason: {reasoning}")
                
                # Save intermediate results every 10 issues
                if len(processed_issues) % 10 == 0:
                    save_processed_issues(processed_issues)
                    
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            # Save what we have so far in case of error
            save_processed_issues(processed_issues)
    
    # Final save of processed issues
    save_processed_issues(processed_issues)
    
    # Print final summary
    print("\n--- Classification Summary ---")
    print(f"Total issues processed: {total_issues}")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_issues * 100) if total_issues > 0 else 0
        print(f"{category}: {count} issues ({percentage:.1f}%)")
    
    print(f"\nIssues requiring Docker: {category_counts[CATEGORIES[1]]}")
    print(f"Issues not requiring Docker: {category_counts[CATEGORIES[0]]}")
    print(f"Issues requiring Docker but cannot be dockerized: {category_counts[CATEGORIES[2]]}")
    
    # Create a summary file
    summary_file = os.path.join(DOCKER_FILTER_DIR, "classification_summary.json")
    summary_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_issues_processed": total_issues,
        "categories": {
            "does_not_need_build": {
                "count": category_counts[CATEGORIES[0]],
                "percentage": (category_counts[CATEGORIES[0]] / total_issues * 100) if total_issues > 0 else 0
            },
            "can_be_dockerized": {
                "count": category_counts[CATEGORIES[1]],
                "percentage": (category_counts[CATEGORIES[1]] / total_issues * 100) if total_issues > 0 else 0
            },
            "need_docker_but_cannot": {
                "count": category_counts[CATEGORIES[2]],
                "percentage": (category_counts[CATEGORIES[2]] / total_issues * 100) if total_issues > 0 else 0
            }
        },
        "processed_files": [os.path.basename(f) for f in json_files]
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nSummary saved to {summary_file}")

if __name__ == "__main__":
    main()
