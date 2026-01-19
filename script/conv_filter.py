"""Bedrock client handling for synthetic prompt generation."""
import time
import boto3
import botocore
from botocore.config import Config
import json
import logging
import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BedrockConfig:
    """Configuration class."""
    region: str = "us-west-2"
    model_id: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

class BedrockClient:
    """Client for interacting with AWS Bedrock service."""

    def __init__(self, config: BedrockConfig, output_log_dir: str = None):
        """Initialize Bedrock client with configuration."""
        self.config = config
        self.client = self._create_client()
        
        # Directory to store LLM outputs
        if output_log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_log_dir = f"llm_logs_{timestamp}"
        
        self.output_log_dir = output_log_dir
        Path(self.output_log_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize log file for LLM outputs
        self.log_file = os.path.join(self.output_log_dir, "llm_responses.jsonl")
        
        # Log header info
        with open(self.log_file, 'a') as f:
            header = {
                "timestamp": datetime.now().isoformat(),
                "model_id": self.config.model_id,
                "session_start": True
            }
            f.write(json.dumps(header) + "\n")

    def _create_client(self):
        """Create and configure Bedrock client."""
        boto_config = Config(
            read_timeout=7200,
            connect_timeout=7200,
            retries={
                "max_attempts": 10000,
                "mode": "standard"
            }
        )
        return boto3.client(
            'bedrock-runtime', 
            config=boto_config,
            region_name=self.config.region
        )

    def generate_message(
    self,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 2048,
    max_retries: int = 1000,
    retry_delay: int = 10,
    metadata: Dict = None
) -> Optional[str]:
        """Generate message using Bedrock model with retry logic."""
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": system_prompt,  # System prompt as top-level parameter
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0
        }
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = self.client.invoke_model(
                    body=json.dumps(body),
                    modelId=self.config.model_id,
                    accept="application/json",
                    contentType="application/json"
                )
                
                response_body = json.loads(response.get('body').read())
                content = response_body["content"][0]["text"]  # Updated response parsing
                
                # Log the LLM response
                self._log_response(system_prompt, user_prompt, content, time.time() - start_time, metadata)
                
                return content
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Retry {attempt + 1}/{max_retries} (Error: {str(e)})")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Max retries reached. Error: {str(e)}")
                    raise

        return None
    
    def _log_response(self, system_prompt: str, user_prompt: str, response: str, 
                     elapsed_time: float, metadata: Dict = None):
        """Log the LLM responses to a file for tracking."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response": response,
            "elapsed_time_seconds": elapsed_time,
            "metadata": metadata or {}
        }
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to log LLM response: {str(e)}")

def evaluate_issue(issue: Dict, bedrock_client: BedrockClient) -> bool:
    """
    Evaluate a GitHub issue using the Bedrock LLM.
    
    Args:
        issue: Dictionary containing issue data
        bedrock_client: Instance of BedrockClient
        
    Returns:
        bool: True if issue meets all criteria, False otherwise
    """
    # Prepare the content for LLM evaluation
    title = issue['title']
    body = issue['body']
    author = issue['author']
    comments = "\n".join([f"{comment['user']}: {comment['body']}" for comment in issue['comments']])
    
    # Extract issue URL
    issue_url = issue.get('url', '')
    
    # If URL doesn't exist or is empty, create a placeholder with file info and title
    if not issue_url:
        # Using title and author as a fallback identifier
        issue_url = f"no_url_{hash(title + author) % 10000}"
    
    user_prompt = f"""
    Please evaluate the following GitHub issue and its comments:

    Title: {title}

    Author: {author}

    Body:
    {body}

    Comments:
    {comments}

    Based on this conversation, please answer the following questions with Yes or No:
    1. Is the problem resolved by someone other than the author (not self-answered)?
    2. Does the conversation contain confirmation from the author that the problem has been resolved?
    3. Is the problem a specific technical issue (not a feature request, opinion, or open-ended question)?
    4. Is there a clear, definitive solution provided within the conversation?
    5. Can the solution be directly applied without requiring additional context or resources?
    6. Does the conversation contain any personally identifiable information (PII) such as Email addresses, Phone numbers, Physical addresses, Full names (beyond just GitHub usernames), Passwords or credentials, Personal identification numbers, IP addresses, or Any other sensitive personal information?
    7. Can this problem be reproduced and solved using the provided solution today (April 2025)?

    Please provide your answers in the format:
    1. [Yes/No]
    2. [Yes/No]
    3. [Yes/No]
    4. [Yes/No]
    5. [Yes/No]
    6. [Yes/No]
    7. [Yes/No]
    """

    system_prompt = "You are an AI assistant tasked with evaluating GitHub issues. Provide accurate and concise answers based on the given information."

    # Add metadata for logging - focus on URL as the primary identifier
    metadata = {
        "issue_url": issue_url,
        "issue_title": title,
        "issue_author": author,
        "repo": issue.get('repo', '')
    }
    
    response = bedrock_client.generate_message(
        system_prompt, 
        user_prompt, 
        metadata=metadata
    )
    
    if response is None:
        return False

    # Parse the response
    lines = response.strip().split('\n')
    not_self_answered = False
    resolved = False
    specific = False
    clear = False
    answer_in_conversation = False
    pii = False
    reproducible = False

    for line in lines:
        if line.startswith('1.'):
            not_self_answered = 'yes' in line.lower()
        elif line.startswith('2.'):
            resolved = 'yes' in line.lower()
        elif line.startswith('3.'):
            specific = 'yes' in line.lower()
        elif line.startswith('4.'):
            clear = 'yes' in line.lower()
        elif line.startswith('5.'):
            answer_in_conversation = 'yes' in line.lower()
        elif line.startswith('6.'):
            pii = 'no' in line.lower()
        elif line.startswith('7.'):
            reproducible = 'yes' in line.lower()  

    # Add evaluation result to metadata
    evaluation_result = {
        "not_self_answered": not_self_answered,
        "resolved": resolved,
        "specific": specific,
        "clear": clear,
        "answer_in_conversation": answer_in_conversation,
        "pii": pii,
        "qualified": resolved and specific and not_self_answered and answer_in_conversation and clear and pii,
        "reproducible": reproducible
    }
    
    # Save the evaluation result with issue URL as the primary identifier
    evaluation_log = {
        "issue_url": issue_url,
        "issue_title": title,
        "issue_author": author,
        "repo": issue.get('repo', ''),
        "evaluation_result": evaluation_result
    }
    
    with open(os.path.join(bedrock_client.output_log_dir, "evaluation_results.jsonl"), 'a') as f:
        f.write(json.dumps(evaluation_log) + "\n")
        
    return resolved and specific and not_self_answered and answer_in_conversation and clear and pii

def process_file(input_file: str, output_file: str, bedrock_client: BedrockClient) -> None:
    """
    Process a single JSON file containing GitHub issues.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        bedrock_client: Instance of BedrockClient
    """
    try:
        with open(input_file, 'r') as f:
            issues = json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {input_file}")
        return
    
    # Check if output file already exists and load processed issues
    already_processed_urls = set()
    qualified_issues = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                qualified_issues = json.load(f)
                # Extract URLs or identifiers of already processed and qualified issues
                already_processed_urls = {issue.get('url', f"no_url_{hash(issue['title'] + issue['author']) % 10000}") 
                                         for issue in qualified_issues}
            logger.info(f"Found existing output file with {len(qualified_issues)} qualified issues")
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error reading existing output file {output_file}: {str(e)}")
            qualified_issues = []
    
    # Also check evaluation log to find processed but unqualified issues
    eval_log_path = os.path.join(bedrock_client.output_log_dir, "evaluation_results.jsonl")
    if os.path.exists(eval_log_path):
        try:
            with open(eval_log_path, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        already_processed_urls.add(entry.get('issue_url', ''))
                    except:
                        continue
        except Exception as e:
            logger.error(f"Error reading evaluation log: {str(e)}")
    
    # Process only unprocessed issues
    new_qualified_issues = []
    for i, issue in enumerate(issues):
        try:
            # Extract URL or generate identifier
            issue_url = issue.get('url', '')
            if not issue_url:
                issue_url = f"no_url_{hash(issue['title'] + issue['author']) % 10000}"
            
            # Skip if already processed
            if issue_url in already_processed_urls:
                logger.info(f"Skipping already processed issue {i+1}/{len(issues)} in {input_file}")
                continue
                
            if evaluate_issue(issue, bedrock_client):
                new_qualified_issues.append(issue)
            logger.info(f"Processed issue {i+1}/{len(issues)} in {input_file}")
        except Exception as e:
            logger.error(f"Error processing issue {i+1} in {input_file}: {str(e)}")
    
    # Combine previously qualified issues with new ones
    all_qualified_issues = qualified_issues + new_qualified_issues
    
    try:
        with open(output_file, 'w') as f:
            json.dump(all_qualified_issues, f, indent=2)
        logger.info(f"Processed {len(issues)} issues. Added {len(new_qualified_issues)} new qualified issues (total: {len(all_qualified_issues)}) saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving qualified issues to {output_file}: {str(e)}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Filter GitHub issues using LLM quality evaluation')
    parser.add_argument('--language', '-l', type=str, default=None,
                       help='Process only a specific language (e.g., python, java, javascript)')
    parser.add_argument('--input-dir', '-i', type=str, default='../recent_v2_June2025_Jan2026',
                       help='Input directory containing raw issues')
    parser.add_argument('--output-dir', '-o', type=str, default='../recent_v2_June2025_Jan2026_conv',
                       help='Output directory for filtered issues')
    args = parser.parse_args()
    
    # Configure input and output directories
    input_dir = args.input_dir
    output_dir = args.output_dir

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Find the most recent log directory if it exists
    existing_log_dirs = sorted([d for d in os.listdir(output_dir) if d.startswith('llm_logs_')], reverse=True)
    
    if existing_log_dirs:
        log_dir = os.path.join(output_dir, existing_log_dirs[0])
        logger.info(f"Continuing with existing log directory: {log_dir}")
    else:
        # Create new log directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(output_dir, f"llm_logs_{timestamp}")
        logger.info(f"Creating new log directory: {log_dir}")
    
    # Initialize Bedrock client with logging directory
    config = BedrockConfig()
    bedrock_client = BedrockClient(config, output_log_dir=log_dir)

    # Get list of files to process
    files_to_process = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            # Filter by language if specified
            if args.language:
                lang_lower = args.language.lower()
                file_lower = filename.lower()
                # Match patterns like python_issues.json, java_issues.json, etc.
                # Only match if language is at the start followed by underscore or dot
                if not (file_lower.startswith(lang_lower + '_') or 
                       file_lower.startswith(lang_lower + '.')):
                    continue
            files_to_process.append(filename)
    
    if args.language:
        logger.info(f"Filtering for language: {args.language}")
        logger.info(f"Found {len(files_to_process)} files to process")
    
    # Process each JSON file
    for filename in files_to_process:
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)
        logger.info(f"Processing {input_file}")
        process_file(input_file, output_file, bedrock_client)
            
    logger.info(f"All LLM outputs have been logged to {log_dir}")

if __name__ == "__main__":
    main()
