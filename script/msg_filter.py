"""Bedrock client implementation for filtering irrelevant comments in GitHub issue conversations."""
import time
import boto3
import botocore
from botocore.config import Config
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add a separate logger for raw LLM responses
raw_response_logger = logging.getLogger('raw_responses')
raw_response_logger.setLevel(logging.INFO)
raw_response_formatter = logging.Formatter('%(asctime)s - %(message)s')

# Create directory for storing raw responses
raw_responses_dir = 'raw_llm_responses'
Path(raw_responses_dir).mkdir(exist_ok=True)

# Create file handler for raw responses
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
raw_response_file = os.path.join(raw_responses_dir, f'raw_responses_{timestamp}.jsonl')
raw_response_handler = logging.FileHandler(raw_response_file)
raw_response_handler.setFormatter(raw_response_formatter)
raw_response_logger.addHandler(raw_response_handler)

@dataclass
class BedrockConfig:
    """Configuration class."""
    region: str = "us-west-2"
    model_id: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

class BedrockClient:
    """Client for interacting with AWS Bedrock service."""

    def __init__(self, config: BedrockConfig):
        """Initialize Bedrock client with configuration."""
        self.config = config
        self.client = self._create_client()

    def _create_client(self):
        """Create and configure Bedrock client."""
        boto_config = Config(
            read_timeout=7200,
            connect_timeout=7200,
            retries={
                "max_attempts": 100,
                "mode": "standard"
            }
        )
        # No profile_name parameter - we'll use default AWS credentials
        return boto3.client('bedrock-runtime', region_name=self.config.region, config=boto_config)

    def _log_response(self, system_prompt: str, user_prompt: str, response: str, 
                     duration: float, metadata: Dict = None):
        """Log raw LLM responses to a file for analysis."""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response": response,
                "duration_seconds": duration,
                "model_id": self.config.model_id
            }
            
            # Add metadata if provided
            if metadata:
                log_entry["metadata"] = metadata
                
            # Write to the raw responses log
            raw_response_logger.info(json.dumps(log_entry))
        except Exception as e:
            logger.error(f"Error logging raw response: {str(e)}")

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

def merge_consecutive_comments(comments: List[Dict]) -> List[Dict]:
    """
    Merge consecutive comments from the same author into a single comment.
    Handle cases where user might be a string.
    """
    if not comments:
        return []
    
    merged_comments = []
    current_author = None
    current_comment = None
    
    for comment in comments:
        try:
            # Extract author safely - handle both string and dict formats
            author = comment.get('user', 'Unknown User')
            if isinstance(author, dict) and 'login' in author:
                author = author['login']
            
            # If this is a comment from the same author as previous one
            if author == current_author:
                # Append this comment's body to the current merged comment
                current_comment['body'] += f"\n\n---\n\n{comment.get('body', '')}"
                # Keep the latest created_at timestamp
                if 'created_at' in comment:
                    current_comment['created_at'] = comment['created_at']
            else:
                # If there was a previous comment being tracked, add it to results
                if current_comment:
                    merged_comments.append(current_comment)
                
                # Start tracking a new comment
                current_author = author
                current_comment = comment.copy()  # Create a copy to avoid modifying the original
        except Exception as e:
            logger.warning(f"Error processing comment during merge: {str(e)}")
            # If error, add the original comment
            if comment:
                merged_comments.append(comment)
    
    # Add the last comment if it exists
    if current_comment:
        merged_comments.append(current_comment)
    
    logger.info(f"Merged {len(comments)} comments into {len(merged_comments)} comments")
    return merged_comments

def filter_relevant_comments(comments: List[Dict], bedrock_client: BedrockClient, issue_title: str = "", issue_body: str = "", issue_number: str = "") -> List[Dict]:
    """
    Filter out only completely irrelevant comments by analyzing the entire conversation context.
    """
    if not comments:
        return []
    
    # First merge consecutive comments from same author
    try:
        merged_comments = merge_consecutive_comments(comments)
    except Exception as e:
        logger.error(f"Error merging comments: {str(e)}")
        merged_comments = comments
    
    if not merged_comments:
        return []

    # Construct the conversation context with richer issue details
    conversation = f"""Issue Title: {issue_title}
Issue Description:
{issue_body}

The conversation is about a technical issue in software development. The following are all comments on this issue:"""
    
    for i, comment in enumerate(merged_comments, 1):
        try:
            # Extract author safely - handle both string and dict formats
            author = comment.get('user', 'Unknown User')
            if isinstance(author, dict) and 'login' in author:
                author = author['login']
                
            body = comment.get('body', '[No content]')
            conversation += f"\n\n[Comment {i} by {author}]:\n{body}"
        except Exception as e:
            logger.warning(f"Error processing comment {i}: {str(e)}")
            conversation += f"\n\n[Comment {i}]:\n[Error processing comment]"

    user_prompt = f"""
Analyze each comment in this GitHub issue conversation:

{conversation}

Your task is to identify ONLY comments that have ABSOLUTELY NO support-related value.
A comment should ONLY be removed if it falls into ALL of these criteria:
- Contains NO technical information
- Provides NO context about the issue
- Asks NO relevant questions (technical or process-related)
- Provides NO status updates or next steps
- Offers NO feedback on proposed solutions
- Contains NO clarifications about the user's situation or environment
- Has NO administrative or process value (like assigning work, requesting more info)

Examples of comments to remove:
1. Pure social messages at conversation end: "Thanks!", "Cool", "👍"
2. Empty status updates: "+1", "Same issue", "Any updates?", "Bump" with no additional context
3. Completely off-topic discussions unrelated to the issue

IMPORTANT: Preserve comments that show the natural flow of support interaction. If a comment contains ANY support-related value, even if minimal or alongside thanks/acknowledgements, DO NOT remove it.

List ONLY the comment numbers that should be removed because they have absolutely no support-related value.
Format: 
NUMBERS: <comma-separated list of numbers>
EXPLANATION: <specific reasons why these comments add no support-related value>

If no comments should be removed, respond with:
NUMBERS: none
EXPLANATION: All comments contain some support-related value or context"""

    system_prompt = """You are a precise comment filter for GitHub issues focused on support interactions.
Your primary goal is to identify and remove ONLY comments that clearly add no value to the support conversation.
PRESERVE any comment that:
- Contains ANY technical information
- Provides ANY context about the issue
- Asks ANY relevant questions (technical or process-related)
- Provides ANY status updates or next steps
- Offers ANY feedback on proposed solutions
- Contains ANY clarifications about the user's situation or environment
- Has ANY administrative or process value (like assigning work, requesting more info)

Even brief contributions should be preserved if they add any value to understanding the support interaction.
Only remove comments that are purely social with no other content (like standalone "Thanks!" and "hi!" messages) or completely off-topic"""

    # Create metadata for logging
    metadata = {
        "issue_title": issue_title,
        "issue_number": issue_number,
        "comment_count": len(merged_comments)
    }

    try:
        response = bedrock_client.generate_message(
            system_prompt=system_prompt, 
            user_prompt=user_prompt, 
            max_tokens=200,
            metadata=metadata  # Use metadata parameter to match method signature
        )
        
        # Log the raw response
        logger.info(f"Raw AI Response: {response}")
        
        if not response:
            return merged_comments

        # Parse the formatted response
        try:
            # Split response into numbers and explanation
            parts = response.split('\n')
            numbers_part = next((p for p in parts if p.startswith('NUMBERS:')), '')
            explanation_part = next((p for p in parts if p.startswith('EXPLANATION:')), '')

            # Extract just the numbers
            numbers_str = numbers_part.replace('NUMBERS:', '').strip()
            explanation = explanation_part.replace('EXPLANATION:', '').strip()

            # Log the parsed components
            logger.info(f"Parsed numbers: {numbers_str}")
            logger.info(f"Explanation: {explanation}")

            if not numbers_str or numbers_str.lower() == 'none':
                return merged_comments

            # Parse the numbers into indices
            remove_indices = {int(num) - 1 for num in numbers_str.split(',') if num.strip().isdigit()}
            
            # Log the content of comments being removed
            if remove_indices:
                logger.info("Comments being removed:")
                for idx in remove_indices:
                    if 0 <= idx < len(merged_comments):
                        # Get author safely
                        author = merged_comments[idx].get('user', 'Unknown User')
                        if isinstance(author, dict) and 'login' in author:
                            author = author['login']
                        
                        preview = merged_comments[idx].get('body', '')[0:100].replace('\n', ' ') + ('...' if len(merged_comments[idx].get('body', '')) > 100 else '')
                        logger.info(f"Comment {idx+1} by {author}: {preview}")
            
            # Keep comments that weren't marked for removal
            filtered_comments = [
                comment for i, comment in enumerate(merged_comments)
                if i not in remove_indices
            ]
            
            # Log the filtering results
            logger.info(f"Original comment count: {len(comments)}")
            logger.info(f"Merged comment count: {len(merged_comments)}")
            logger.info(f"Filtered comment count: {len(filtered_comments)}")
            logger.info(f"Removed indices: {remove_indices}")
            
            return filtered_comments

        except ValueError as e:
            logger.error(f"Error parsing response '{response}': {str(e)}")
            return merged_comments

    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return merged_comments

def process_file(input_file: str, output_file: str, bedrock_client: BedrockClient) -> None:
    """
    Process a single JSON file containing GitHub issues.
    """
    try:
        with open(input_file, 'r') as f:
            issues = json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {input_file}")
        return

    filtered_issues = []
    for i, issue in enumerate(issues):
        try:
            issue_number = issue.get('number', f"{i}")
            # Filter comments for each issue
            filtered_comments = filter_relevant_comments(
                issue['comments'], 
                bedrock_client,
                issue_title=issue.get('title', ''),
                issue_body=issue.get('body', ''),
                issue_number=str(issue_number)
            )
            if filtered_comments:  # Only include issues that have relevant comments
                issue['comments'] = filtered_comments
                filtered_issues.append(issue)
            logger.info(f"Processed issue {i+1}/{len(issues)} in {input_file}")
        except Exception as e:
            logger.error(f"Error processing issue {i+1} in {input_file}: {str(e)}")

    try:
        with open(output_file, 'w') as f:
            json.dump(filtered_issues, f, indent=2)
        logger.info(f"Processed {len(issues)} issues. {len(filtered_issues)} issues with relevant comments saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving filtered issues to {output_file}: {str(e)}")

def main():
    # Configure input and output directories
    input_dir = '../recent_v2_June2025_Jan2026_conv'      # Input: from conv_filter.py (Step 3)
    output_dir = '../recent_v2_June2025_Jan2026_msg'      # Output: noise removed

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create logs directory within the output directory
    logs_dir = os.path.join(output_dir, 'logs')
    Path(logs_dir).mkdir(exist_ok=True)
    
    # Set up raw response logger to write to the output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    raw_response_file = os.path.join(logs_dir, f'raw_responses_{timestamp}.jsonl')
    
    # Remove existing handlers from the logger
    for handler in raw_response_logger.handlers[:]:
        raw_response_logger.removeHandler(handler)
    
    # Add new handler with the updated file path
    raw_response_handler = logging.FileHandler(raw_response_file)
    raw_response_handler.setFormatter(raw_response_formatter)
    raw_response_logger.addHandler(raw_response_handler)
    
    logger.info(f"Storing raw LLM responses to: {raw_response_file}")

    # Initialize Bedrock client
    config = BedrockConfig()
    bedrock_client = BedrockClient(config)

    # Process each JSON file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)
            logger.info(f"Processing {input_file}")
            process_file(input_file, output_file, bedrock_client)

if __name__ == "__main__":
    main()
