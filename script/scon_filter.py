"""Bedrock client handling for virtual user generation from GitHub conversations."""
import time
import boto3
import botocore
import json
import logging
import os
import uuid
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

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

    def __init__(self, config: BedrockConfig):
        """Initialize Bedrock client with configuration."""
        self.config = config
        self.client = self._create_client()
        # Add storage for raw responses
        self.raw_responses = []

    def _create_client(self):
        """Create and configure Bedrock client."""
        boto_config = botocore.config.Config(
            read_timeout=7200,
            connect_timeout=7200,
            retries={"max_attempts": 100}
        )
        # Use default credentials without specifying profile_name
        return boto3.client('bedrock-runtime', region_name=self.config.region, config=boto_config)

    def generate_message(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 5000,
        max_retries: int = 1000,
        retry_delay: int = 10,
        metadata: Dict = None
    ) -> Optional[str]:
        """Generate message using Bedrock model with retry logic."""
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": system_prompt,
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
                
                # Store the prompt and response
                self.raw_responses.append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "response": content,
                    "metadata": metadata
                })
                
                return content
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Retry {attempt + 1}/{max_retries} (Error: {str(e)})")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Max retries reached. Error: {str(e)}")
                    raise

        return None


class ConversationAnalyzer:
    """Analyzer for GitHub conversations."""

    def __init__(self, bedrock_client: BedrockClient):
        self.bedrock_client = bedrock_client

    def verify_conditions(self, conversation: Dict, conditions: List[Dict]) -> List[str]:
        """Verify which conditions are satisfied in the original conversation."""
        
        # Format comments with user information
        formatted_comments = []
        for comment in conversation['comments']:
            formatted_comment = {
                "user": comment['user'],
                "created_at": comment['created_at'],
                "body": comment['body']
            }
            formatted_comments.append(formatted_comment)
        
        verification_prompt = f"""
        Given this GitHub conversation:

        Title: {conversation['title']}
        Author: {conversation['author']}
        Question: {conversation['body']}
        
        Comments:
        {json.dumps(formatted_comments, indent=2)}

        And these potential user satisfaction conditions:
        {json.dumps(conditions)}

        For each condition, determine if it represents a general criterion that satisfied the user, NOT a specific solution.
        
        A good satisfaction condition describes WHAT the user needed to know or understand, not the exact answer provided.
        
        Return your analysis in this exact JSON format:
        {{
            "verified_conditions": [
                {{
                    "condition": "The condition text",
                    "is_satisfied": true/false,
                    "explanation": "Why this represents a general satisfaction criterion (not a specific answer) based on the user's response"
                }}
            ]
        }}
        
        Reject conditions that contain specific answers rather than general satisfaction criteria.
        """

        try:
            metadata = {
                "operation": "verify_conditions",
                "conversation_id": conversation.get('number', 'unknown')
            }
            
            llm_response = self.bedrock_client.generate_message(
                "You are an expert GitHub conversation analyst.", 
                verification_prompt,
                metadata=metadata
            )
            
            if not llm_response:
                return []

            # Parse response and extract satisfied conditions
            response_data = json.loads(llm_response[llm_response.find('{'):llm_response.rfind('}')+1])
            satisfied_conditions = [
                item['condition'] 
                for item in response_data['verified_conditions'] 
                if item['is_satisfied']
            ]
            
            # Log verification results
            logger.info("Verification results:")
            for item in response_data['verified_conditions']:
                status = "SATISFIED" if item['is_satisfied'] else "NOT SATISFIED"
                logger.info(f"{status}: {item['condition']}")
                logger.info(f"Explanation: {item['explanation']}")
                logger.info("---")
                
            return satisfied_conditions
            
        except Exception as e:
            logger.error(f"Error verifying conditions: {str(e)}")
            return []

    def analyze_conversation(self, conversation: Dict) -> Optional[List[str]]:
        """Analyze a conversation and extract satisfaction conditions."""
        
        # Updated system prompt to avoid including answers in satisfaction conditions
        system_prompt = """
        You are an expert at analyzing GitHub issues and extracting user satisfaction conditions. Your task is to identify the general criteria that determine whether a response would satisfy the user's needs.

A satisfaction condition describes WHAT the user needs, not HOW the solution was implemented.

## Levels of Abstraction for Satisfaction Conditions:

TOO SPECIFIC (AVOID THESE):
- "Use numpy.where(condition, x, y) with condition being data > 0"
- "Add the line 'export PATH=$PATH:/usr/local/bin' to .bashrc"
- "Set max_depth=5 in the RandomForest constructor"

GOOD ABSTRACTION LEVEL:
- "A vectorized approach to conditional element selection in arrays"
- "A permanent solution to the PATH environment variable configuration"
- "Guidance on appropriate hyperparameter settings for tree depth"

TOO GENERIC (AVOID THESE):
- "A working solution"
- "Information about the library"
- "Code that does what they want"

## Key characteristics of good satisfaction conditions:

1. TRANSFERABLE: Could apply to multiple potential solutions
2. VERIFIABLE: Clear criteria to determine if a solution meets this condition
3. EVIDENCED: Based on the user's explicit or implicit feedback
4. NEEDS-FOCUSED: Describes what problem needs solving, not implementation specifics
5. TECHNICAL: Must be about technical requirements, not social interactions

## EXCLUDE these types of conditions (NOT ALLOWED):
- "Acknowledgment that the issue was legitimate" (social, not technical)
- "Validation of the user's concern" (conversational dynamic)
- "Appreciation or thanks from maintainer" (social courtesy)
- "Confirmation that the bug was real" (meta-discussion)
- "Recognition of the problem" (acknowledgment, not solution)
- Any condition about how the responder COMMUNICATED rather than WHAT they provided
        """

        # Format comments with user information
        formatted_comments = []
        for comment in conversation['comments']:
            formatted_comment = {
                "user": comment['user'],
                "created_at": comment['created_at'],
                "body": comment['body']
            }
            formatted_comments.append(formatted_comment)
        
        user_prompt = f"""
Given this GitHub conversation:

Title: {conversation['title']}
Author: {conversation['author']}
Question: {conversation['body']}

Comments:
{json.dumps(formatted_comments, indent=2)}

Extract user satisfaction conditions - these are criteria that any acceptable answer must meet to satisfy the user.

IMPORTANT: Satisfaction conditions are NOT the answers themselves, but the criteria by which any answer would be judged.

Return your response in this exact JSON format:
{{
    "satisfaction_conditions": [
        {{
            "condition": "A criteria that any satisfactory answer must meet",
            "explanation": "Why this meets the user's needs as shown in the conversation"
        }}
    ]
}}

You can have as many conditions as you like, but they should be:
- Generic enough that multiple different solutions could satisfy it
- Focused on what the user needs, not how it was implemented
- Based on clear evidence from the user's response showing satisfaction
- Free from specific implementation details
"""

        try:
            logger.info(f"Generating LLM response for conversation: {conversation['number']}")
            
            metadata = {
                "operation": "analyze_conversation",
                "conversation_id": conversation.get('number', 'unknown'),
                "conversation_title": conversation.get('title', 'unknown')
            }
            
            llm_response = self.bedrock_client.generate_message(
                system_prompt, 
                user_prompt,
                metadata=metadata
            )
            
            if not llm_response:
                logger.error("LLM returned empty response")
                return None

            # Log the raw response
            logger.info(f"Raw LLM response: {llm_response}")

            # Clean up the response
            cleaned_response = llm_response.strip()
            start = cleaned_response.find('{')
            end = cleaned_response.rfind('}') + 1
            
            if start == -1 or end == 0:
                logger.error(f"Could not find JSON object in response: {cleaned_response}")
                return None
                
            json_str = cleaned_response[start:end]
            logger.info(f"Cleaned JSON string: {json_str}")
            
            try:
                response_data = json.loads(json_str)
                if not isinstance(response_data, dict) or 'satisfaction_conditions' not in response_data:
                    logger.error("Invalid JSON structure")
                    return None

                # Verify which conditions are actually satisfied
                satisfied_conditions = self.verify_conditions(conversation, response_data['satisfaction_conditions'])
                
                if not satisfied_conditions:
                    logger.error("No satisfied conditions found")
                    return None

                return satisfied_conditions

            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                return None

        except Exception as e:
            logger.error(f"Error analyzing conversation {conversation['number']}: {str(e)}")
            return None

def get_output_filename(input_file: str, output_dir: str) -> str:
    """Generate output filename based on input filename."""
    base_filename = os.path.basename(input_file)
    return os.path.join(output_dir, base_filename)  # Keep the same filename in output dir
    
def process_directory(input_dir: str, output_dir: str, bedrock_client: BedrockClient):
    """Process all JSON files in the input directory."""
    logger.info(f"Processing directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    logger.info(f"Found {len(json_files)} JSON files")

    for idx, json_file in enumerate(json_files, 1):
        input_file = os.path.join(input_dir, json_file)
        output_file = get_output_filename(json_file, output_dir)
        logger.info(f"\nProcessing file {idx}/{len(json_files)}: {json_file}")
        
        try:
            process_conversations(input_file, output_file, bedrock_client)
        except Exception as e:
            logger.error(f"Error processing file {json_file}: {str(e)}")
            continue

def process_conversations(input_file: str, output_file: str, bedrock_client: BedrockClient):
    """Process conversations and extract virtual users with satisfaction conditions."""
    try:
        logger.info(f"Processing {input_file}")
        logger.info(f"Output will be saved to {output_file}")

        with open(input_file, 'r') as f:
            conversations = json.load(f)

        logger.info(f"Loaded {len(conversations)} conversations from input file")

        analyzer = ConversationAnalyzer(bedrock_client)
        processed_conversations = []

        for idx, conversation in enumerate(conversations, 1):
            logger.info(f"Processing conversation {idx}/{len(conversations)} (ID: {conversation['number']})")
            
            # Extract satisfaction conditions
            satisfaction_conditions = analyzer.analyze_conversation(conversation)
            
            if satisfaction_conditions:
                # Add satisfaction conditions directly to the original conversation object
                conversation_copy = conversation.copy()
                conversation_copy["satisfaction_conditions"] = satisfaction_conditions
                processed_conversations.append(conversation_copy)
                logger.info(f"Added satisfaction conditions for conversation {conversation['number']}")
            else:
                logger.warning(f"Failed to extract satisfaction conditions for conversation {conversation['number']}")

        if processed_conversations:
            with open(output_file, 'w') as f:
                json.dump(processed_conversations, f, indent=2)
            logger.info(f"Saved {len(processed_conversations)} processed conversations to {output_file}")
            
            # Save prompts and responses
            prompts_responses_file = output_file.replace('.json', '_prompts_responses.json')
            with open(prompts_responses_file, 'w') as f:
                json.dump(bedrock_client.raw_responses, f, indent=2)
            logger.info(f"Saved {len(bedrock_client.raw_responses)} prompts and responses to {prompts_responses_file}")
        else:
            logger.warning("No conversations were processed")

        # Count conversations with satisfaction conditions
        conversations_with_conditions = sum(1 for conv in processed_conversations if "satisfaction_conditions" in conv)
        logger.info(f"Processed {len(conversations)} conversations. "
                  f"{conversations_with_conditions} have satisfaction conditions.")

    except Exception as e:
        logger.error(f"Error processing file {input_file}: {str(e)}")
        raise

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate satisfaction conditions for GitHub issues using LLM.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scon_filter.py --input-dir my_data/issues --output-dir my_data/with_scon
  python scon_filter.py -i ../issues -o ../scon --model us.anthropic.claude-sonnet-4-5-20250929-v1:0
  python scon_filter.py -i data/raw -o data/processed --region us-east-1
        """
    )
    parser.add_argument('--input-dir', '-i', type=str, required=True,
                        help='Directory containing JSON files with issues')
    parser.add_argument('--output-dir', '-o', type=str, required=True,
                        help='Output directory for issues with satisfaction conditions')
    parser.add_argument('--region', '-r', type=str, default='us-west-2',
                        help='AWS region for Bedrock (default: us-west-2)')
    parser.add_argument('--model', '-m', type=str, 
                        default='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                        help='Bedrock model ID (default: claude-sonnet-4.5)')
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    # Check if input directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory {input_dir} does not exist")
        exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Bedrock client with CLI arguments
    config = BedrockConfig(region=args.region, model_id=args.model)
    bedrock_client = BedrockClient(config)
    
    print(f"Using model: {args.model}")
    print(f"Using region: {args.region}")

    # Process all files in directory
    process_directory(input_dir, output_dir, bedrock_client)

if __name__ == "__main__":
    main()
