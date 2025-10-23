#!/usr/bin/env python3
"""
Generic Dockerfile Generator for GitHub Issues Dataset

This script generates Dockerfiles for GitHub issues that are classified as dockerizable
but are missing dockerfile information. Designed for use across different environments,
programming languages, and folder structures.

Usage:
    python generate_dockerfile.py --input-dir dataset/issues --output-dir results
    
    python generate_dockerfile.py \\
        --input-dir /path/to/dataset \\
        --output-dir /path/to/results \\
        --log-dir /path/to/logs \\
        --failure-logs-dir /path/to/failure_logs
"""

import os
import json
import boto3
import time
import datetime
import subprocess
import tempfile
import shutil
import concurrent.futures
import yaml  # For parsing YAML workflow files
import logging
import argparse
import sys
from botocore.config import Config

# Global configuration variables - will be set by argument parser
input_dir = None
output_dir = None
failure_logs_dir = None
log_dir = None

# Counter for LLM API calls
LLM_CALL_COUNT = 0

def setup_argument_parser():
    """
    Set up command line argument parser for configurable script execution.
    
    This allows users to customize paths and behavior based on their environment
    and requirements, making the script suitable for various use cases.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='Generate Dockerfiles for dockerizable GitHub issues missing dockerfile information',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with default settings
    python generate_dockerfile.py --input-dir dataset/issues
    
    # Full customization
    python generate_dockerfile.py \\
        --input-dir /path/to/dataset \\
        --output-dir /path/to/results \\
        --log-dir /path/to/logs \\
        --failure-logs-dir /path/to/failure_logs \\
        --languages python javascript \\
        --dockerizable-category "Can be dockerized without any issue" \\
        --candidates 5 \\
        --disable-docker-testing
        """
    )
    
    # Required arguments
    parser.add_argument('--input-dir', '-i', 
                       required=True,
                       help='Input directory containing the GitHub issues dataset (e.g., dataset/updated)')
    
    # Optional directory arguments with defaults
    parser.add_argument('--output-dir', '-o',
                       default='logs/dockerfile_generation',
                       help='Output directory for generated results (default: logs/dockerfile_generation)')
    
    parser.add_argument('--log-dir', 
                       default='logs',
                       help='Directory for log files (default: logs)')
    
    parser.add_argument('--failure-logs-dir',
                       default='logs/dockerfile_failures', 
                       help='Directory for build failure logs (default: logs/dockerfile_failures)')
    
    # Processing configuration
    parser.add_argument('--languages', nargs='+',
                       help='Specific programming languages to process (default: all languages)')
    
    parser.add_argument('--dockerizable-category',
                       default='Can be dockerized without any issue',
                       help='Classification category for dockerizable issues')
    
    # Generation parameters
    parser.add_argument('--candidates', type=int, default=3,
                       help='Number of Dockerfile candidates to generate per issue (default: 3)')
    
    parser.add_argument('--max-attempts', type=int, default=4,
                       help='Maximum improvement attempts for failed Dockerfiles (default: 4)')
    
    # Docker testing configuration
    parser.add_argument('--disable-docker-testing', action='store_true',
                       help='Disable Docker build testing (generate Dockerfiles only)')
    
    parser.add_argument('--docker-timeout', type=int, default=600,
                       help='Docker build timeout in seconds (default: 600)')
    
    # AWS/LLM configuration
    parser.add_argument('--aws-region', default='us-east-2',
                       help='AWS region for Bedrock API calls (default: us-east-2)')
    
    parser.add_argument('--model-id', 
                       default='us.anthropic.claude-3-7-sonnet-20250219-v1:0',
                       help='AWS Bedrock model ID to use for generation')
    
    # Advanced parameters
    parser.add_argument('--max-workers', type=int, default=5,
                       help='Maximum number of parallel workers (default: 5)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging output')
    
    return parser

def setup_logging(log_directory):
    """
    Set up comprehensive logging for dockerfile generation process.
    
    Args:
        log_directory (str): Directory where log files should be created
        
    Returns:
        str: Path to the created log file
    """
    # Ensure log directory exists
    os.makedirs(log_directory, exist_ok=True)
    
    # Create timestamped log file for this session
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"dockerfile_generation_{timestamp}.log"
    log_path = os.path.join(log_directory, log_filename)
    
    # Configure comprehensive logging
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Log session start with configuration info
    logging.info("="*60)
    logging.info("DOCKERFILE GENERATION SESSION STARTED")
    logging.info("="*60)
    logging.info(f"Log file: {log_path}")
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Failure logs directory: {failure_logs_dir}")
    
    print(f"📝 Logging detailed information to: {log_path}")
    
    return log_path

def log_llm_interaction(model_id, prompt, response_body, call_context=""):
    """
    Log the LLM interaction details including prompt and raw response
    
    Args:
        model_id: The ID of the model used
        prompt: The prompt sent to the model
        response_body: The raw response from the model
        call_context: Additional context about where this call was made from
    """
    logging.info(f"===== LLM INTERACTION {LLM_CALL_COUNT} =====")
    logging.info(f"Context: {call_context}")
    logging.info(f"Model: {model_id}")
    logging.info("--- PROMPT ---")
    logging.info(prompt)
    logging.info("--- RAW RESPONSE ---")
    logging.info(json.dumps(response_body, indent=2))
    logging.info("===========================================\n")

def call_llm_with_adaptive_context(bedrock_client, model_id, system_prompt, user_prompt, 
                                  readme_content, repo_structure_text, workflows_text, reference_dockerfile_text,
                                  max_retries=5):
    """
    Call LLM with adaptive context reduction when hitting context window errors
    """
    # Start with your original content sizes
    readme_length = 10000  # Start with your default size
    structure_length = 5000
    workflow_files_max = 2
    
    for attempt in range(max_retries):
        try:
            # Prepare the content with current sizes
            truncated_readme = truncate_text(readme_content, readme_length) if readme_content else ""
            readme_text = f"\nRepository README:\n{truncated_readme}\n\n" if truncated_readme else ""
            
            # Truncate structure text
            truncated_structure = truncate_text(repo_structure_text, structure_length) if repo_structure_text else ""
            structure_text = f"\nRepository Structure:\n```\n{truncated_structure}\n```\n" if truncated_structure else ""
            
            # Limit workflow files based on current setting
            limited_workflows_text = ""
            if workflows_text:
                # Extract workflow files (assuming workflows_text already contains the workflow files)
                # This is a simplified approach - you may need to adjust based on your actual data structure
                workflow_parts = workflows_text.split("Workflow file: ")
                if len(workflow_parts) > 1:
                    limited_workflows_text = "\nGitHub Workflow files found in repository:\n"
                    # Take first part plus up to workflow_files_max additional parts
                    for part in workflow_parts[:workflow_files_max+1]:
                        if "```yaml" in part:
                            # Truncate the yaml content
                            before_yaml = part.split("```yaml")[0]
                            yaml_content = part.split("```yaml")[1].split("```")[0]
                            after_yaml = part.split("```", 2)[2] if len(part.split("```")) > 2 else ""
                            
                            truncated_yaml = truncate_text(yaml_content, 2000)
                            limited_workflows_text += f"Workflow file: {before_yaml}```yaml{truncated_yaml}```{after_yaml}"
                        else:
                            limited_workflows_text += part
            
            # Combine all for the final prompt
            full_prompt = user_prompt + readme_text + structure_text + limited_workflows_text + reference_dockerfile_text
            
            # Log attempt details
            print(f"LLM call attempt {attempt+1}/{max_retries} - README size: {readme_length}, Structure size: {structure_length}, Workflows: {workflow_files_max}")
            
            # Call Claude 3.7 Sonnet
            response_text = call_sonnet_37(
                prompt=full_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=15000
            )
            
            # Create a response body similar to what we'd get from DeepSeek for logging compatibility
            response_body = {
                "content": [{"text": response_text}]
            }
            
            # Log the interaction
            log_llm_interaction(
                model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                prompt=f"SYSTEM:\n{system_prompt}\n\nUSER:\n{full_prompt}",
                response_body=response_body,
                call_context=f"adaptive_context_call (attempt #{attempt+1})"
            )
            
            # Return the successful response in a format compatible with the rest of the code
            return {
                "choices": [{"message": {"content": response_text}}]
            }
            
        except Exception as e:
            error_str = str(e)
            print(f"Error in LLM call (attempt {attempt+1}): {error_str}")
            
            # Check if it's a context window error
            if "context" in error_str.lower() and "window" in error_str.lower() or "token" in error_str.lower():
                # Reduce context sizes for next attempt
                readme_length = int(readme_length * 0.7)  # Reduce by 30%
                structure_length = int(structure_length * 0.7)
                if workflow_files_max > 0:
                    workflow_files_max -= 1
                    
                print(f"Reducing context for next attempt - README: {readme_length}, Structure: {structure_length}, Workflows: {workflow_files_max}")
                
                # Ensure we don't go too low
                if readme_length < 1000:
                    readme_length = 1000
                if structure_length < 1000:
                    structure_length = 1000
            else:
                # If it's not a context window error, just raise it
                raise
    
    # If we get here, we've exhausted all retries
    raise Exception(f"Failed to call LLM after {max_retries} attempts with progressively reduced context")

def generate_failure_explanation(error_message):
    """Generate one sentence explanation for Docker build failure"""
    # Truncate error message if too long
    truncated_error = truncate_text(error_message, 3000)
    
    # System prompt
    system_prompt = 'Provide clear, concise explanations about Docker build failures.'
    
    # User prompt
    user_prompt = f"""
    The following is an error message from a failed Docker image build:
    
    ```
    {truncated_error}
    ```
    
    Please provide a clear, concise one-sentence explanation of why the Docker build failed.
    Your response should be ONLY the single sentence explanation with no additional text.
    """
    
    try:
        # Call Claude 3.7 Sonnet
        explanation = call_sonnet_37(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=1000
        )
        
        # Create a response body similar to what we'd get from DeepSeek for logging
        response_body = {
            "content": [{"text": explanation}]
        }
        
        # Log the interaction
        log_llm_interaction(
            model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            prompt=f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}",
            response_body=response_body,
            call_context="generate_failure_explanation"
        )
        
        # Ensure it's just one sentence (take the first sentence if multiple)
        if "." in explanation:
            explanation = explanation.split(".", 1)[0] + "."
            
        return explanation
    except Exception as e:
        print(f"Error generating explanation: {str(e)}")
        logging.error(f"Error in generate_failure_explanation: {str(e)}")
        return "Docker build failed due to an unspecified error."

def truncate_text(text, max_length=3000):
    """Truncate text to a maximum length, adding indicator if truncated"""
    if not text or len(text) <= max_length:
        return text
    return text[:max_length//2] + "\n...[truncated]...\n" + text[-max_length//2:]


def parse_github_workflow(workflow_content):
    """Extract useful information from GitHub workflow file"""
    try:
        # Parse YAML content
        workflow_data = yaml.safe_load(workflow_content)
        
        info = {
            "name": workflow_data.get("name", "Unnamed workflow"),
            "jobs": [],
            "build_steps": [],
            "dependencies": []
        }
        
        # Extract job information
        for job_name, job_config in workflow_data.get("jobs", {}).items():
            job_info = {
                "name": job_name,
                "runs-on": job_config.get("runs-on", ""),
                "steps": []
            }
            
            # Extract steps
            for step in job_config.get("steps", []):
                job_info["steps"].append({
                    "name": step.get("name", ""),
                    "uses": step.get("uses", ""),
                    "run": step.get("run", "")
                })
                
                # Collect build steps specifically (useful for Dockerfile generation)
                if step.get("run") and any(keyword in step.get("run", "").lower() 
                                         for keyword in ["build", "install", "compile", "dotnet", "msbuild", "npm", "mvn"]):
                    info["build_steps"].append(step.get("run"))
                    
                # Extract dependency information
                if step.get("uses") and any(keyword in step.get("uses", "").lower() 
                                         for keyword in ["setup-dotnet", "setup-node", "setup-java", "setup-python"]):
                    info["dependencies"].append(step.get("uses"))
                    
            info["jobs"].append(job_info)
            
        return info
    except Exception as e:
        print(f"Error parsing workflow: {str(e)}")
        return None

def generate_dockerfile_candidates(issue_data, num_candidates=5, error_message=None, reference_dockerfile=None):
    """Generate multiple Dockerfile candidates in parallel"""
    global LLM_CALL_COUNT
    
    # Get repository information
    repo_url = issue_data.get("url", "").split("/issues/")[0] if "/issues/" in issue_data.get("url", "") else ""
    issue_title = issue_data.get("title", "")
    issue_body = truncate_text(issue_data.get("body", ""), 3000)
    issue_number = issue_data.get("number", "unknown")
    commit_sha = issue_data.get("git_commit_info", {}).get("sha", "")
    
    # Add README if available (truncate if too long)
    readme_content = issue_data.get("repository_info", {}).get("readme", "")
    
    # Add repository structure if available
    repo_structure_text = ""
    if issue_data.get("repository_info", {}).get("structure_summary"):
        repo_structure_text = issue_data['repository_info']['structure_summary']
    
    # Add reference Dockerfile if available
    reference_dockerfile_text = ""
    if reference_dockerfile:
        reference_dockerfile_text = f"\nReference Dockerfile from another successful issue in this repository:\n{reference_dockerfile}\n\n"
    # If no reference but there's a repo Dockerfile, use that
    elif issue_data.get("repository_info", {}).get("dockerfile", ""):
        original_dockerfile = issue_data.get("repository_info", {}).get("dockerfile", "")
        reference_dockerfile_text = f"\nOriginal Dockerfile from repository:\n{original_dockerfile}\n\n"
    
    # Add GitHub workflow files if available (provide raw files)
    workflows_text = ""
    if issue_data.get("repository_info", {}).get("github_workflows"):
        workflows = issue_data.get("repository_info", {}).get("github_workflows", {})
        if workflows:
            # Include up to 2 workflow files to avoid overly large prompts
            workflow_files = list(workflows.items())[:2]
            
            workflows_text = "\nGitHub Workflow files found in repository:\n"
            for name, content in workflow_files:
                # Truncate each workflow file if too long
                truncated_content = truncate_text(content, 4000)
                workflows_text += f"\nWorkflow file: {name}\n```yaml\n{truncated_content}\n```\n"
    
    # Create a system prompt
    system_prompt = "You are an expert Docker engineer who creates Dockerfiles to build and validate GitHub projects."
    
    # Create a prompt based on whether we're generating a new Dockerfile or fixing an existing one
    if error_message:
        # Get existing Dockerfile
        existing_dockerfile = issue_data.get("dockerfile", "")
        
        prompt = f"""
        I need to fix a Dockerfile that failed to build for this GitHub issue. Please help me correct the errors.
        
        Repository URL: {repo_url}
        Issue Number: {issue_number}
        Issue Title: {issue_title}
        Reference Commit SHA: {commit_sha}
        
        Here's the current Dockerfile:
        ```
        {existing_dockerfile}
        ```
        
        The Docker build failed with the following error:
        ```
        {truncate_text(error_message, 3000)}
        ```
        
        Please provide an improved version of the Dockerfile that fixes these errors.
        Make sure it:
        1. Sets up the appropriate environment to build and validate the solution for this issue
        2. Installs all necessary dependencies
        3. Clones the repository and checks out commit {commit_sha} if the author did not specifically mention the version or commit hash
        4. Builds the project
        5. Do not need to run anything but have all dependencies and build the project the user need
        6. Do not have option flags when you build the project unless the user asked to do so or the project mentioned to have them
        
        IMPORTANT: Your response should ONLY contain the Dockerfile content itself, with no additional text, markdown formatting, or code blocks. Just the plain Dockerfile content that I can use directly.
        """
    else:
        prompt = f"""
        I need to create a Dockerfile to validate the solution to this GitHub issue.
        
        Repository URL: {repo_url}
        Issue Number: {issue_number}
        Issue Title: {issue_title}
        Commit SHA: {commit_sha}
        
        Issue Description:
        {issue_body}
        
        Please create a detailed Dockerfile that:
        1. Sets up the appropriate environment to build and validate the solution for this issue
        2. Installs all necessary dependencies
        3. Clones the repository and checks out the specific commit {commit_sha} if the author did not specifically mention the version or commit hash.
        4. Builds the project
        5. Do not need to run anything but have all dependencies and build the project the the user need.
        6. Do not have option flags when you build the project unless the user asked to do so or the project mentioned to have them
        
        The Dockerfile should be complete and ready to use. Please include comments to explain each step.
        
        IMPORTANT: Your response should ONLY contain the Dockerfile content itself, with no additional text, markdown formatting, or code blocks. Just the plain Dockerfile content that I can use directly.
        """
    
    # Function to generate a single candidate with adaptive context handling
    def generate_candidate(candidate_num):
        global LLM_CALL_COUNT  # Use global instead of nonlocal
        try:
            print(f"LLM API Call #{LLM_CALL_COUNT+1}: Generating candidate #{candidate_num}")
            
            # Use adaptive context handling
            response_body = call_llm_with_adaptive_context(
                None,  # No bedrock client needed for call_sonnet_37
                "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                system_prompt,
                prompt,
                readme_content,
                repo_structure_text,
                workflows_text,
                reference_dockerfile_text
            )
            
            dockerfile_content = response_body["choices"][0]["message"]["content"]
            
            # Clean up the response
            # First, try to extract content from code blocks if present
            if "```dockerfile" in dockerfile_content and "```" in dockerfile_content.split("```dockerfile", 1)[1]:
                dockerfile_content = dockerfile_content.split("```dockerfile", 1)[1].split("```", 1)[0].strip()
            elif "```Dockerfile" in dockerfile_content and "```" in dockerfile_content.split("```Dockerfile", 1)[1]:
                dockerfile_content = dockerfile_content.split("```Dockerfile", 1)[1].split("```", 1)[0].strip()
            elif "```" in dockerfile_content and "```" in dockerfile_content.split("```", 1)[1]:
                dockerfile_content = dockerfile_content.split("```", 1)[1].split("```", 1)[0].strip()
            
            # Remove any explanations before/after the actual Dockerfile content
            lines = dockerfile_content.split('\n')
            cleaned_lines = []
            in_dockerfile = False
            
            for line in lines:
                # Start capturing at the FROM instruction
                if line.strip().startswith("FROM "):
                    in_dockerfile = True
                    
                if in_dockerfile:
                    cleaned_lines.append(line)
                    
            # If we found a FROM directive, use the cleaned content
            if in_dockerfile:
                dockerfile_content = '\n'.join(cleaned_lines)
                
            return dockerfile_content
        except Exception as e:
            print(f"  Error generating Dockerfile candidate #{candidate_num}: {str(e)}")
            return None

    # Generate candidates concurrently
    candidates = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_candidates, 5)) as executor:
        future_to_candidate = {
            executor.submit(generate_candidate, i + 1): i + 1
            for i in range(num_candidates)
        }
        
        for future in concurrent.futures.as_completed(future_to_candidate):
            candidate_num = future_to_candidate[future]
            try:
                dockerfile_content = future.result()
                if dockerfile_content and dockerfile_content.strip():
                    candidates.append({
                        "dockerfile": dockerfile_content,
                        "candidate_number": candidate_num
                    })
                    print(f"  Successfully generated candidate #{candidate_num}")
                else:
                    print(f"  Failed to generate valid content for candidate #{candidate_num}")
            except Exception as e:
                print(f"  Error processing candidate #{candidate_num}: {str(e)}")
                
    return candidates

def improve_dockerfile_candidate(issue_data, candidate, error, attempt_number, improvement_attempt, reference_dockerfile=None):
    """Improve a specific Dockerfile candidate based on its error"""
    global LLM_CALL_COUNT
    
    # Get repository information
    repo_url = issue_data.get("url", "").split("/issues/")[0] if "/issues/" in issue_data.get("url", "") else ""
    issue_title = issue_data.get("title", "")
    issue_body = truncate_text(issue_data.get("body", ""), 3000)
    issue_number = issue_data.get("number", "unknown")
    commit_sha = issue_data.get("git_commit_info", {}).get("sha", "")
    
    # Add README if available (truncate if too long)
    readme_content = issue_data.get("repository_info", {}).get("readme", "")
    
    # Add repository structure if available
    repo_structure_text = ""
    if issue_data.get("repository_info", {}).get("structure_summary"):
        repo_structure_text = issue_data['repository_info']['structure_summary']
    
    # Add reference Dockerfile if available
    reference_dockerfile_text = ""
    if reference_dockerfile:
        reference_dockerfile_text = f"\nReference Dockerfile from another successful issue in this repository:\n{reference_dockerfile}\n\n"
    # If no reference but there's a repo Dockerfile, use that
    elif issue_data.get("repository_info", {}).get("dockerfile", ""):
        original_dockerfile = issue_data.get("repository_info", {}).get("dockerfile", "")
        reference_dockerfile_text = f"\nOriginal Dockerfile from repository:\n{original_dockerfile}\n\n"
    
    # Add GitHub workflow files if available (provide raw files)
    workflows_text = ""
    if issue_data.get("repository_info", {}).get("github_workflows"):
        workflows = issue_data.get("repository_info", {}).get("github_workflows", {})
        if workflows:
            # Include up to 2 workflow files to avoid overly large prompts
            workflow_files = list(workflows.items())[:2]
            
            workflows_text = "\nGitHub Workflow files found in repository:\n"
            for name, content in workflow_files:
                # Truncate each workflow file if too long
                truncated_content = truncate_text(content, 4000)
                workflows_text += f"\nWorkflow file: {name}\n```yaml\n{truncated_content}\n```\n"
    
    # Create a system prompt
    system_prompt = "You are an expert Docker engineer who specializes in fixing Dockerfiles that failed to build. Provide solutions that address build errors directly."
    
    # Create a prompt to fix this specific candidate
    fix_prompt = f"""
    I need to fix a Dockerfile that failed to build for this GitHub issue. Please help me correct the errors.
    
    Repository URL: {repo_url}
    Issue Number: {issue_number}
    Issue Title: {issue_title}
    Commit SHA: {commit_sha}
    
    Issue Description:
    {issue_body}
    
    Here's the current Dockerfile that's failing:
    ```
    {candidate["dockerfile"]}
    ```
    
    The Docker build failed with the following error:
    ```
    {truncate_text(error, 3000)}
    ```
    
    Please provide an improved version of the Dockerfile that fixes these specific errors.
    Make sure it:
    1. Sets up the appropriate environment to build and validate the solution for this issue
    2. Installs all necessary dependencies
    3. Clones the repository and checks out commit {commit_sha}
    4. Builds the project
    5. Do not need to run anything but have all dependencies and build the project the user need.
    6. Do not have option flags when you build the project unless the user asked to do so or the project mentioned to have them
    
    IMPORTANT: Your response should ONLY contain the Dockerfile content itself, with no additional text, markdown formatting, or code blocks. Just the plain Dockerfile content that I can use directly.
    """
    
    try:
        # Make LLM call to fix the error with adaptive context
        print(f"  LLM API Call #{LLM_CALL_COUNT+1}: Improving candidate #{candidate['candidate_number']} (attempt #{improvement_attempt})")
        
        # Use adaptive context handling
        fix_response_body = call_llm_with_adaptive_context(
            None,  # No bedrock client needed for call_sonnet_37
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            system_prompt,
            fix_prompt,
            readme_content,
            repo_structure_text,
            workflows_text,
            reference_dockerfile_text
        )
        
        fixed_dockerfile = fix_response_body["choices"][0]["message"]["content"]
        
        # Clean up the response
        if "```dockerfile" in fixed_dockerfile and "```" in fixed_dockerfile.split("```dockerfile", 1)[1]:
            fixed_dockerfile = fixed_dockerfile.split("```dockerfile", 1)[1].split("```", 1)[0].strip()
        elif "```Dockerfile" in fixed_dockerfile and "```" in fixed_dockerfile.split("```Dockerfile", 1)[1]:
            fixed_dockerfile = fixed_dockerfile.split("```Dockerfile", 1)[1].split("```", 1)[0].strip()
        elif "```" in fixed_dockerfile and "```" in fixed_dockerfile.split("```", 1)[1]:
            fixed_dockerfile = fixed_dockerfile.split("```", 1)[1].split("```", 1)[0].strip()
        
        # Remove any explanations before/after the actual Dockerfile content
        fixed_lines = fixed_dockerfile.split('\n')
        fixed_cleaned_lines = []
        in_fixed_dockerfile = False
        
        for line in fixed_lines:
            # Start capturing at the FROM instruction
            if line.strip().startswith("FROM "):
                in_fixed_dockerfile = True
                
            if in_fixed_dockerfile:
                fixed_cleaned_lines.append(line)
                
        # If we found a FROM directive, use the cleaned content
        if in_fixed_dockerfile:
            fixed_dockerfile = '\n'.join(fixed_cleaned_lines)
        
        return {
            "dockerfile": fixed_dockerfile,
            "candidate_number": candidate["candidate_number"],
            "improvement_attempt": improvement_attempt
        }
    except Exception as e:
        print(f"  Error improving Dockerfile candidate #{candidate['candidate_number']}: {str(e)}")
        return None

def test_build_dockerfile(dockerfile_content, repo_name, issue_number):
    """
    Test building the Docker image from the Dockerfile.
    Returns (success, error_message)
    """
    # Create a temporary directory to work in
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a unique tag for the image
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        image_tag = f"{repo_name.replace('/', '_')}_{issue_number}_{timestamp}".lower()
        
        # Write Dockerfile to temp directory
        dockerfile_path = os.path.join(temp_dir, "Dockerfile")
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)
            
        # Build the Docker image
        print(f"Building Docker image {image_tag}...")
        process = subprocess.run(
            ["docker", "build", "-t", image_tag, "."],
            cwd=temp_dir,
            text=True,
            capture_output=True,
            timeout=600  # 10 minute timeout
        )
        
        # Check if build was successful
        if process.returncode == 0:
            print(f"Successfully built Docker image {image_tag}")
            return True, ""
        else:
            error_message = process.stderr or process.stdout
            print(f"Error building Docker image: {error_message[:500]}...")
            return False, error_message
    
    except subprocess.TimeoutExpired:
        error_message = "Docker build timed out after 10 minutes"
        return False, error_message
    except Exception as e:
        error_message = f"Error during Docker build: {str(e)}"
        return False, error_message
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)

def test_candidates_in_parallel(candidates, repo_name, issue_number, max_workers=5):
    """Test multiple Dockerfile candidates in parallel"""
    results = []
    
    def test_single_candidate(candidate):
        try:
            dockerfile_content = candidate["dockerfile"]
            candidate_num = candidate["candidate_number"]
            improvement_attempt = candidate.get("improvement_attempt", 0)
            print(f"Testing candidate #{candidate_num}...")
            success, error = test_build_dockerfile(dockerfile_content, repo_name, issue_number)
            
            # Generate and save failure explanation if this is a failure
            if not success:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                repo_name_clean = repo_name.replace('/', '_')
                attempt_num = improvement_attempt + 1  # Default to 1 if not specified
                failure_file = f"{repo_name_clean}_{issue_number}_{timestamp}_attempt{attempt_num}_candidate{candidate_num}.json"
                failure_path = os.path.join(failure_logs_dir, failure_file)
                
                explanation = generate_failure_explanation(error)
                result = {
                    "build_status": "failed",
                    "failure_explanation": explanation,
                    "error_message": error,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                with open(failure_path, 'w') as f:
                    json.dump(result, f, indent=2)
                    
                print(f"Saved failure explanation to {failure_path}")
            
            # Return result dict
            return {
                "dockerfile": dockerfile_content,
                "success": success,
                "error": error if not success else None,
                "improvement_attempt": improvement_attempt,
                "candidate_number": candidate_num
            }
        except Exception as e:
            print(f"Error testing candidate #{candidate.get('candidate_number', 'unknown')}: {str(e)}")
            return {
                "dockerfile": candidate.get("dockerfile", ""),
                "success": False,
                "error": f"Testing error: {str(e)}",
                "improvement_attempt": candidate.get("improvement_attempt", 0),
                "candidate_number": candidate.get("candidate_number", 0)
            }
    
    # Process candidates in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_candidate = {
            executor.submit(test_single_candidate, candidate): candidate
            for candidate in candidates
        }
        
        for future in concurrent.futures.as_completed(future_to_candidate):
            candidate = future_to_candidate[future]
            try:
                result = future.result()
                results.append(result)
                
                # If we found a successful candidate, we can return early
                if result["success"]:
                    # Cancel any remaining futures
                    for f in future_to_candidate:
                        f.cancel()
                    break
                    
            except Exception as e:
                print(f"Error processing result for candidate #{candidate.get('candidate_number', 'unknown')}: {str(e)}")
    
    return results

def generate_and_test_dockerfile_candidates(issue_data, repo_name, issue_number, error_message=None, num_candidates=5, reference_dockerfile=None):
    """
    Generate and test Dockerfile candidates in parallel for improved efficiency:
    1. Generate all candidates in parallel
    2. Test all candidates in parallel, stopping if one succeeds
    3. If all candidates fail, improve them in parallel and test again
    """
    # STEP 1: Generate candidates in parallel
    print(f"Generating {num_candidates} Dockerfile candidates in parallel...")
    candidates = generate_dockerfile_candidates(issue_data, num_candidates, error_message, reference_dockerfile)
    
    # Save all generated candidates
    all_candidates = candidates.copy()
    
    # STEP 2: Test candidates in parallel
    print(f"Testing {len(candidates)} candidates in parallel...")
    results = test_candidates_in_parallel(candidates, repo_name, issue_number)
    
    # Save all results
    all_results = results.copy()
    
    # Check if any candidate was successful
    successful_result = next((r for r in results if r["success"]), None)
    if successful_result:
        print(f"Found successful candidate #{successful_result['candidate_number']}!")
        return successful_result["dockerfile"], all_candidates, all_results, True
    
    # STEP 3: If all candidates failed, try to improve each one
    print(f"\nAll {len(candidates)} candidates failed. Starting improvement attempts...")
    
    # Maximum number of improvement attempts per candidate
    max_improvement_attempts = 4
    
    # For each round of improvements
    for improvement_attempt in range(1, max_improvement_attempts + 1):
        print(f"\nImprovement round #{improvement_attempt}")
        
        # Collect candidates to improve
        candidates_to_improve = []
        for result in results:
            candidates_to_improve.append({
                "dockerfile": result["dockerfile"],
                "candidate_number": result["candidate_number"],
                "error": result["error"]
            })
        
        # Skip if no candidates to improve
        if not candidates_to_improve:
            break
            
        # Improve candidates in parallel
        improved_candidates = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(candidates_to_improve), 5)) as executor:
            future_to_candidate = {
                executor.submit(
                    improve_dockerfile_candidate, 
                    issue_data,
                    candidate,
                    candidate["error"],
                    1,  # attempt_number (always 1 for this refactored version)
                    improvement_attempt,
                    reference_dockerfile  # Add reference dockerfile if available
                ): candidate
                for candidate in candidates_to_improve
            }
            
            for future in concurrent.futures.as_completed(future_to_candidate):
                candidate = future_to_candidate[future]
                try:
                    improved_candidate = future.result()
                    if improved_candidate and improved_candidate.get("dockerfile"):
                        improved_candidates.append(improved_candidate)
                        all_candidates.append(improved_candidate)  # Add to all candidates
                except Exception as e:
                    print(f"Error improving candidate #{candidate['candidate_number']}: {str(e)}")
        
        # Test the improved candidates in parallel
        if improved_candidates:
            print(f"Testing {len(improved_candidates)} improved candidates in parallel...")
            improved_results = test_candidates_in_parallel(improved_candidates, repo_name, issue_number)
            
            # Add to all results
            all_results.extend(improved_results)
            
            # Update current results for next round of improvements
            results = improved_results
            
            # Check if any improved candidate was successful
            successful_improved = next((r for r in improved_results if r["success"]), None)
            if successful_improved:
                print(f"Found successful improved candidate #{successful_improved['candidate_number']} (improvement attempt #{improvement_attempt})!")
                return successful_improved["dockerfile"], all_candidates, all_results, True
        else:
            print("No viable improved candidates to test")
            break
    
    # If we get here, no candidates were successful
    return None, all_candidates, all_results, False


def extract_repo_name_from_url(url):
    """Extract repository name from GitHub URL"""
    if not url or "github.com" not in url:
        return None
    
    try:
        # URL format: https://github.com/owner/repo/issues/number
        parts = url.split("github.com/")[1].split("/")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
    except:
        pass
    
    return None


def find_dockerizable_missing_dockerfile(dataset_directory, dockerizable_category, target_languages=None):
    """
    Find issues that are dockerizable but missing dockerfile information.
    
    Args:
        dataset_directory (str): Path to the dataset directory
        dockerizable_category (str): Classification category to look for
        target_languages (list, optional): Specific languages to process
        
    Returns:
        list: List of dictionaries containing issue information and file paths
    """
    from pathlib import Path
    
    dataset_dir = Path(dataset_directory)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    dockerizable_missing = []
    
    logging.info(f"Scanning dataset directory: {dataset_dir}")
    logging.info(f"Looking for category: '{dockerizable_category}'")
    logging.info(f"Target languages: {target_languages or 'all languages'}")
    
    # Walk through all language subdirectories
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if not file.endswith('.json'):
                continue
                
            file_path = Path(root) / file
            language = Path(root).name
            
            # Filter by target languages if specified
            if target_languages and language not in target_languages:
                logging.debug(f"Skipping {language} (not in target languages)")
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for issue_index, issue in enumerate(data):
                        if isinstance(issue, dict):
                            # Check if issue matches the dockerizable category
                            classification = issue.get('_classification', {})
                            category = classification.get('category', '')
                            
                            if category == dockerizable_category:
                                # Check if dockerfile is missing
                                has_dockerfile = 'dockerfile' in issue and issue['dockerfile']
                                
                                if not has_dockerfile:
                                    dockerizable_missing.append({
                                        'file_path': file_path,
                                        'filename': file,
                                        'language': language,
                                        'issue_index': issue_index,
                                        'issue': issue
                                    })
            
            except Exception as e:
                logging.error(f"Error reading {file_path}: {e}")
                print(f"Error reading {file_path}: {e}")
    
    logging.info(f"Found {len(dockerizable_missing)} dockerizable issues missing Dockerfile")
    return dockerizable_missing


def call_sonnet_37(prompt, system_prompt=None, temperature=0.7, max_tokens=15000, max_retries=1000):
    """
    Make an API call to Claude 3.7 Sonnet using Bedrock with robust retry logic
    
    Args:
        prompt: The prompt to send to the model
        system_prompt: System prompt (optional)
        temperature: Sampling temperature (default 0.7)
        max_tokens: Maximum number of tokens to generate (default 15000)
        max_retries: Maximum number of retry attempts (default 1000)
        
    Returns:
        The generated text response
    """
    global LLM_CALL_COUNT
    LLM_CALL_COUNT += 1
    
    # Initialize the Bedrock client
    config = Config(retries={"max_attempts": 10000, "mode": "standard"})
    bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-2', config=config)
    
    # Model ID for Claude 3.7 Sonnet
    model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    
    # Format request for Claude 3.7 Sonnet
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    # Add system prompt if provided
    if system_prompt:
        request_body["system"] = system_prompt
    
    # Implement retry logic
    retry_count = 0
    backoff_time = 1  # Start with 1 second backoff
    max_backoff = 60  # Maximum backoff of 60 seconds
    
    while retry_count < max_retries:
        try:
            # Call the API
            response = bedrock_client.invoke_model(
                body=json.dumps(request_body),
                modelId=model_id,
                accept='application/json',
                contentType='application/json'
            )
            
            # Process the response
            response_body = json.loads(response.get('body').read())
            
            # Log the interaction
            log_llm_interaction(
                model_id=model_id,
                prompt=f"SYSTEM: {system_prompt}\nUSER: {prompt}" if system_prompt else prompt,
                response_body=response_body,
                call_context=f"call_sonnet_37 (LLM Call #{LLM_CALL_COUNT}, Attempt #{retry_count+1})"
            )
            
            # Extract the response text
            response_text = response_body["content"][0]["text"]
            
            # If we got here, the call was successful
            if retry_count > 0:
                print(f"  Succeeded after {retry_count+1} attempts")
            
            return response_text
            
        except Exception as e:
            retry_count += 1
            
            # Log the error
            error_message = str(e)
            print(f"  API call error (attempt {retry_count}/{max_retries}): {error_message}")
            logging.warning(f"API call error (attempt {retry_count}/{max_retries}): {error_message}")
            
            if retry_count >= max_retries:
                print(f"  Failed after {max_retries} attempts, raising exception")
                logging.error(f"Failed after {max_retries} attempts: {error_message}")
                raise
            
            sleep_time = 10
            
            print(f"  Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
            
            # Increase backoff time for next attempt (exponential backoff)
            backoff_time = min(backoff_time * 2, max_backoff)

def initialize_configuration(args):
    """
    Initialize global configuration variables from parsed arguments.
    
    Args:
        args: Parsed command line arguments
    """
    global input_dir, output_dir, failure_logs_dir, log_dir
    
    # Set global directory paths
    input_dir = args.input_dir
    output_dir = args.output_dir
    failure_logs_dir = args.failure_logs_dir
    log_dir = args.log_dir
    
    # Create necessary directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(failure_logs_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Log configuration
    print(f"🔧 Configuration:")
    print(f"  Input directory: {input_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Failure logs directory: {failure_logs_dir}")
    print(f"  Log directory: {log_dir}")
    if args.languages:
        print(f"  Target languages: {', '.join(args.languages)}")
    else:
        print(f"  Target languages: all")
    print(f"  Docker testing: {'enabled' if not args.disable_docker_testing else 'disabled'}")
    print(f"  Candidates per issue: {args.candidates}")
    print(f"  Max improvement attempts: {args.max_attempts}")

def main():
    """
    Main function that handles argument parsing and orchestrates the dockerfile generation process.
    """
    # Set up argument parser
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Initialize configuration from arguments
    initialize_configuration(args)
    
    # Validate input directory exists
    if not os.path.exists(args.input_dir):
        print(f"❌ Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Set up logging with configurable directory
    current_log_file = setup_logging(log_dir)
    logging.info("Dockerfile generation process started")
    logging.info(f"Arguments: {vars(args)}")
    
    try:
        # Process dockerizable issues missing dockerfile
        print("🚀 Starting Dockerfile generation process...")
        dockerizable_missing = find_dockerizable_missing_dockerfile(
            dataset_directory=input_dir,
            dockerizable_category=args.dockerizable_category,
            target_languages=args.languages
        )
        
        if not dockerizable_missing:
            print("✅ No dockerizable issues missing dockerfile found!")
            logging.info("No issues to process - task completed")
            return
        
        print(f"📋 Found {len(dockerizable_missing)} dockerizable issues missing dockerfile")
        
        # Show language breakdown
        language_counts = {}
        for item in dockerizable_missing:
            lang = item['language']
            language_counts[lang] = language_counts.get(lang, 0) + 1
        
        print("📊 Issues by language:")
        for lang, count in sorted(language_counts.items()):
            print(f"  {lang}: {count} issues")
        
        # Group by file for efficient batch processing
        files_to_update = {}
        for item in dockerizable_missing:
            file_key = str(item['file_path'])
            if file_key not in files_to_update:
                files_to_update[file_key] = {
                    'file_path': item['file_path'],
                    'language': item['language'],
                    'issues': []
                }
            files_to_update[file_key]['issues'].append({
                'index': item['issue_index'],
                'issue': item['issue']
            })
        
        # Processing statistics
        total_processed = 0
        total_generated = 0
        successful_builds = 0
        total_files_updated = 0
        
        # Keep track of successfully generated Dockerfiles by repository for reuse
        successful_dockerfiles = {}  # repo_name -> dockerfile
        
        # Process each file containing dockerizable issues
        for file_key, file_info in files_to_update.items():
            file_path = file_info['file_path']
            language = file_info['language']
            issues_to_process = file_info['issues']
            
            print(f"\n📁 Processing {file_path.name} ({language}): {len(issues_to_process)} issues")
            
            try:
                # Load current dataset file
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                file_updated = False
                
                # Process each issue in this file that needs a dockerfile
                for issue_item in issues_to_process:
                    issue_index = issue_item['index']
                    issue_data = issue_item['issue']
                    issue_number = issue_data.get('number', 'unknown')
                    repo_name = extract_repo_name_from_url(issue_data.get('url', ''))
                    
                    if not repo_name:
                        print(f"  ⚠️  Cannot extract repo name for issue #{issue_number}, skipping")
                        continue
                    
                    print(f"  🔧 Processing issue #{issue_number} ({repo_name})...")
                    total_processed += 1
                    
                    # Validate commit_id exists and is valid
                    commit_sha = issue_data.get("commit_id", "")
                    if not commit_sha or commit_sha == "None":
                        print(f"    ⚠️ No valid commit_id for issue #{issue_number}, skipping")
                        continue
                    
                    # Create minimal compatibility data structures
                    issue_data["git_commit_info"] = {"sha": commit_sha}
                    issue_data["repository_info"] = {
                        "readme": "",
                        "structure_summary": "",
                        "github_workflows": {}
                    }
                    
                    # Check if we have a successful Dockerfile from this repo to use as reference
                    reference_dockerfile = successful_dockerfiles.get(repo_name)
                    
                    # Generate and test dockerfile with iterative improvements
                    successful_dockerfile, all_candidates, all_results, success = generate_and_test_dockerfile_candidates(
                        issue_data, 
                        repo_name, 
                        issue_number,
                        error_message=None,  # First attempt has no error
                        num_candidates=args.candidates,
                        reference_dockerfile=reference_dockerfile
                    )
                    
                    if successful_dockerfile and successful_dockerfile.strip():
                        # Update the dataset with the generated dockerfile
                        data[issue_index]['dockerfile'] = successful_dockerfile
                        
                        total_generated += 1
                        file_updated = True
                        
                        if success:
                            successful_builds += 1
                            successful_dockerfiles[repo_name] = successful_dockerfile
                            print(f"    ✅ Successfully generated and tested dockerfile for issue #{issue_number}")
                        else:
                            print(f"    ⚠️  Generated dockerfile for issue #{issue_number} but build tests failed")
                    else:
                        print(f"    ❌ Failed to generate dockerfile for issue #{issue_number}")
                
                # Save updated dataset file if any changes were made
                if file_updated:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    total_files_updated += 1
                    
                    # Count successful builds in this file
                    successful_in_file = sum(1 for i in issues_to_process 
                                           if 'dockerfile' in data[i['index']])
                    
                    print(f"  💾 Updated {file_path.name} with {successful_in_file} new dockerfiles")
                
            except Exception as e:
                print(f"  ❌ Error processing file {file_path}: {e}")
                logging.error(f"Error processing file {file_path}: {e}")
        
        # Print final summary statistics
        print(f"\n{'='*20} 📊 FINAL SUMMARY {'='*20}")
        print(f"Files processed: {len(files_to_update)}")
        print(f"Files updated: {total_files_updated}")
        print(f"Issues processed: {total_processed}")
        print(f"Dockerfiles generated: {total_generated}")
        
        if args.disable_docker_testing:
            print(f"Docker testing: disabled")
        else:
            print(f"Successful builds: {successful_builds}")
            print(f"Build success rate: {(successful_builds/total_generated)*100 if total_generated > 0 else 0:.1f}%")
        
        print(f"Total LLM API calls: {LLM_CALL_COUNT}")
        print(f"Log file: {current_log_file}")
        print("="*60)
        
        # Log final summary
        logging.info("Dockerfile generation process completed successfully")
        logging.info(f"Final statistics: {total_files_updated} files updated, "
                    f"{total_generated} dockerfiles generated, {successful_builds} successful builds")
        
    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user")
        logging.info("Process interrupted by user (KeyboardInterrupt)")
    except Exception as e:
        error_msg = f"Error in main process: {str(e)}"
        print(f"❌ {error_msg}")
        logging.error(error_msg)
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
