#!/usr/bin/env python3
"""
Strands-Powered Dockerfile Generator for GitHub Issues Dataset

This script uses Strands AI agents to generate Dockerfiles for GitHub issues by:
1. Cloning the actual repository at the specific commit
2. Exploring the codebase to understand build requirements
3. Generating informed Dockerfiles based on actual repository context
4. Iteratively improving failed builds up to 10 attempts

The Strands agent can execute commands, read files, and explore the repository
to gather the information needed to create accurate Dockerfiles.

Usage:
    python generate_dockerfile_with_strands.py \\
        --input-dir dataset/issues \\
        --output-dir results \\
        --languages python java
"""

import os
import json
import time
import datetime
import subprocess
import tempfile
import shutil
import logging
import argparse
import sys
import signal
import multiprocessing
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError

# Add CAB tools to path
import sys
cab_tools_path = str(Path(__file__).parent.parent / 'tools' / 'src')
if cab_tools_path not in sys.path:
    sys.path.insert(0, cab_tools_path)

# Global configuration
input_dir = None
output_dir = None
failure_logs_dir = None
log_dir = None
LLM_CALL_COUNT = 0


def setup_argument_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Generate Dockerfiles using Strands AI agents',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input-dir', '-i', required=True,
                       help='Input directory containing GitHub issues')
    parser.add_argument('--output-dir', '-o', default='logs/dockerfile_generation_strands',
                       help='Output directory for results')
    parser.add_argument('--log-dir', default='logs',
                       help='Directory for log files')
    parser.add_argument('--failure-logs-dir', default='logs/dockerfile_failures_strands',
                       help='Directory for build failure logs')
    parser.add_argument('--languages', nargs='+',
                       help='Specific programming languages to process')
    parser.add_argument('--dockerizable-category', default='Can be dockerized without any issue',
                       help='Classification category for dockerizable issues')
    parser.add_argument('--max-attempts', type=int, default=10,
                       help='Maximum improvement attempts per issue (default: 10)')
    parser.add_argument('--docker-timeout', type=int, default=600,
                       help='Docker build timeout in seconds (default: 600)')
    parser.add_argument('--aws-region', default='us-west-2',
                       help='AWS region for Bedrock (default: us-west-2)')
    parser.add_argument('--model-id', default='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                       help='AWS Bedrock model ID')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--parallel', '-p', type=int, default=1,
                       help='Number of issues to process in parallel (default: 1)')
    parser.add_argument('--agent-timeout', type=int, default=300,
                       help='Timeout for each agent attempt in seconds (default: 300)')
    parser.add_argument('--issue-timeout', type=int, default=1800,
                       help='Timeout for processing entire issue in seconds (default: 1800 = 30min)')
    
    return parser


def setup_logging(log_directory):
    """Set up logging."""
    os.makedirs(log_directory, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"dockerfile_generation_strands_{timestamp}.log"
    log_path = os.path.join(log_directory, log_filename)
    
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    print(f"📝 Logging to: {log_path}")
    return log_path


def initialize_configuration(args):
    """Initialize global configuration from arguments."""
    global input_dir, output_dir, failure_logs_dir, log_dir
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    failure_logs_dir = args.failure_logs_dir
    log_dir = args.log_dir
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(failure_logs_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"🔧 Configuration:")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Logs: {log_dir}")
    if args.languages:
        print(f"  Languages: {', '.join(args.languages)}")


def test_build_dockerfile(dockerfile_content: str, repo_name: str, issue_number: str, timeout: int = 600) -> Tuple[bool, str]:
    """Test building the Docker image."""
    temp_dir = tempfile.mkdtemp()
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        image_tag = f"{repo_name.replace('/', '_')}_{issue_number}_{timestamp}".lower()
        
        dockerfile_path = os.path.join(temp_dir, "Dockerfile")
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)
        
        print(f"  🐳 Building Docker image: {image_tag}")
        process = subprocess.run(
            ["docker", "build", "-t", image_tag, "."],
            cwd=temp_dir,
            text=True,
            capture_output=True,
            timeout=timeout
        )
        
        if process.returncode == 0:
            print(f"  ✅ Docker build successful!")
            return True, ""
        else:
            error_message = process.stderr or process.stdout
            print(f"  ❌ Docker build failed")
            return False, error_message
    
    except subprocess.TimeoutExpired:
        return False, f"Docker build timed out after {timeout} seconds"
    except Exception as e:
        return False, f"Error during Docker build: {str(e)}"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _run_agent_in_process(args_tuple) -> Tuple[Optional[str], bool, str]:
    """
    Run Strands agent in isolated process. This function runs in a subprocess
    and can be safely killed without corrupting the main event loop.
    
    Returns: (dockerfile_content, build_success, error_message)
    """
    (
        issue_data, repo_name, issue_number, model_id, 
        max_attempts, docker_timeout, agent_timeout, failure_logs_dir_path
    ) = args_tuple
    
    # Import inside process to avoid event loop issues
    try:
        from strands import Agent
        from strands.models.bedrock import BedrockModel
        from strands_tools import shell, file_read
    except ImportError as e:
        return None, False, f"Import error: {e}"
    
    # Set environment for shell timeout (must be less than agent_timeout)
    shell_timeout = max(60, agent_timeout - 60)
    os.environ['SHELL_DEFAULT_TIMEOUT'] = str(shell_timeout)
    
    repo_url = issue_data.get("url", "").split("/issues/")[0] if "/issues/" in issue_data.get("url", "") else ""
    commit_sha = issue_data.get("commit_id", "")
    issue_title = issue_data.get("title", "")
    issue_body = issue_data.get("body", "")[:1000]
    
    if not commit_sha or commit_sha == "None":
        return None, False, "No valid commit_id"
    
    workspace_dir = tempfile.mkdtemp(prefix=f"strands_dockerfile_{issue_number}_")
    previous_error = ""
    
    try:
        for attempt in range(1, max_attempts + 1):
            print(f"  🤖 [{repo_name}] Attempt {attempt}/{max_attempts}")
            
            try:
                # Create fresh agent for each attempt
                model = BedrockModel(
                    model_id=model_id,
                    region_name='us-west-2'
                )
                
                agent = Agent(
                    model=model,
                    tools=[shell, file_read],
                    system_prompt="""You are an expert DevOps engineer specializing in creating Dockerfiles.
                    
Your task is to explore a GitHub repository and create a working Dockerfile that:
1. Sets up the correct base image and environment
2. Installs all necessary dependencies
3. Clones the repository and checks out the specified commit
4. Builds the project successfully

You have access to:
- shell: Execute bash commands to explore the repo
- file_read: Read file contents

Use these tools to understand the project structure and build requirements."""
                )
                
                # Prepare the task prompt
                if attempt == 1:
                    task = f"""Please create a Dockerfile for this GitHub project:

Repository: {repo_url}
Commit SHA: {commit_sha}
Issue: {issue_title}

Description: {issue_body}

Steps to follow:
1. Clone the repository: git clone {repo_url} repo
2. Change directory: cd repo
3. Checkout the specific commit: git checkout {commit_sha}
4. Explore the repository structure to understand the project
5. Identify build requirements (check package.json, pom.xml, CMakeLists.txt, requirements.txt, etc.)
6. Create a Dockerfile that:
   - Uses an appropriate base image
   - Installs all dependencies
   - Clones and checks out the repository
   - Builds the project
   
Return ONLY the Dockerfile content, no explanations or markdown formatting."""
                else:
                    task = f"""The previous Dockerfile failed to build. Here's the error:

{previous_error[:2000]}

Please fix the Dockerfile. The repository is already cloned at ./repo

Steps:
1. Analyze the build error
2. Explore the repository if needed to understand the issue
3. Create an improved Dockerfile that fixes the error

Return ONLY the Dockerfile content, no explanations or markdown formatting."""
                
                # Use synchronous invoke (no asyncio) - this is the key fix
                print(f"  🔄 Agent working (timeout: {agent_timeout}s)...")
                
                # Set alarm for hard timeout (Unix only)
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Agent timed out after {agent_timeout}s")
                
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(agent_timeout)
                
                try:
                    # Use synchronous call instead of async
                    result = agent(task)
                    signal.alarm(0)  # Cancel alarm
                finally:
                    signal.signal(signal.SIGALRM, old_handler)
                
                # Extract text from result
                if hasattr(result, 'message') and 'content' in result.message:
                    response_text = result.message['content'][0]['text']
                else:
                    response_text = str(result)
                
                # Extract Dockerfile from response
                dockerfile_content = response_text
                
                # Clean up response (remove markdown code blocks if present)
                if "```dockerfile" in dockerfile_content:
                    dockerfile_content = dockerfile_content.split("```dockerfile")[1].split("```")[0].strip()
                elif "```Dockerfile" in dockerfile_content:
                    dockerfile_content = dockerfile_content.split("```Dockerfile")[1].split("```")[0].strip()
                elif "```" in dockerfile_content:
                    parts = dockerfile_content.split("```")
                    if len(parts) >= 3:
                        dockerfile_content = parts[1].strip()
                
                # Extract only FROM and subsequent lines
                lines = dockerfile_content.split('\n')
                dockerfile_lines = []
                found_from = False
                for line in lines:
                    if line.strip().startswith("FROM "):
                        found_from = True
                    if found_from:
                        dockerfile_lines.append(line)
                
                if dockerfile_lines:
                    dockerfile_content = '\n'.join(dockerfile_lines)
                
                print(f"  ✅ Dockerfile generated ({len(dockerfile_content)} chars)")
                
                # Test the Dockerfile
                print(f"  🧪 Testing Dockerfile...")
                success, error = test_build_dockerfile(
                    dockerfile_content, repo_name, issue_number, docker_timeout
                )
                
                if success:
                    print(f"  🎉 Success on attempt {attempt}!")
                    return dockerfile_content, True, ""
                else:
                    print(f"  ❌ Build failed on attempt {attempt}")
                    previous_error = error
                    
                    # Save failure log
                    if failure_logs_dir_path:
                        try:
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            failure_file = f"{repo_name.replace('/', '_')}_{issue_number}_{timestamp}_attempt{attempt}.json"
                            failure_path = os.path.join(failure_logs_dir_path, failure_file)
                            
                            with open(failure_path, 'w') as f:
                                json.dump({
                                    "attempt": attempt,
                                    "dockerfile": dockerfile_content,
                                    "error": error[:5000],
                                    "timestamp": datetime.datetime.now().isoformat()
                                }, f, indent=2)
                        except Exception:
                            pass
                    
                    if attempt < max_attempts:
                        print(f"  🔄 Will retry with error feedback...")
                        
            except TimeoutError as e:
                print(f"  ⏰ Agent timed out: {e}")
                if attempt >= max_attempts:
                    return None, False, str(e)
            except Exception as e:
                print(f"  ❌ Agent error: {str(e)}")
                if attempt >= max_attempts:
                    return None, False, str(e)
        
        # All attempts failed
        print(f"  ❌ Failed after {max_attempts} attempts")
        return None, False, f"Failed after {max_attempts} attempts"
        
    finally:
        # Cleanup workspace
        shutil.rmtree(workspace_dir, ignore_errors=True)


def extract_repo_name_from_url(url: str) -> Optional[str]:
    """Extract repository name from GitHub URL."""
    if not url or "github.com" not in url:
        return None
    
    try:
        parts = url.split("github.com/")[1].split("/")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
    except:
        pass
    
    return None


def find_dockerizable_missing_dockerfile(dataset_directory: str, dockerizable_category: str, target_languages=None):
    """Find issues that need Dockerfiles."""
    dataset_dir = Path(dataset_directory)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    dockerizable_missing = []
    
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if not file.endswith('.json'):
                continue
            
            file_path = Path(root) / file
            language = Path(root).name
            
            if target_languages and language not in target_languages:
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for issue_index, issue in enumerate(data):
                        if isinstance(issue, dict):
                            classification = issue.get('_classification', {})
                            category = classification.get('category', '')
                            
                            if category == dockerizable_category:
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
    
    return dockerizable_missing


def process_single_issue(
    file_path: Path,
    language: str,
    issue_index: int,
    issue_data: Dict,
    model_id: str,
    max_attempts: int,
    docker_timeout: int,
    agent_timeout: int,
    issue_timeout: int,
    failure_logs_dir_path: str
) -> Tuple[bool, bool, Optional[str]]:
    """
    Process a single issue using multiprocessing with hard timeout.
    Returns: (processed, build_success, dockerfile)
    """
    issue_number = issue_data.get('number', 'unknown')
    repo_name = extract_repo_name_from_url(issue_data.get('url', ''))
    
    if not repo_name:
        print(f"  [{language}] ⚠️  Cannot extract repo name for issue #{issue_number}")
        return False, False, None
    
    print(f"\n📦 [{language}] Processing issue #{issue_number}")
    print(f"  Repository: {repo_name}")
    
    # Prepare arguments for subprocess
    args = (
        issue_data, repo_name, str(issue_number), model_id,
        max_attempts, docker_timeout, agent_timeout, failure_logs_dir_path
    )
    
    # Use ProcessPoolExecutor with hard timeout
    # spawn context ensures clean process without inherited event loop
    ctx = multiprocessing.get_context('spawn')
    
    try:
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
            future = executor.submit(_run_agent_in_process, args)
            try:
                dockerfile, build_success, error = future.result(timeout=issue_timeout)
                
                if dockerfile:
                    return True, build_success, dockerfile
                else:
                    print(f"  [{language}] ❌ Failed: {error}")
                    return True, False, None
                    
            except FuturesTimeoutError:
                print(f"  [{language}] ⏰ Issue #{issue_number} timed out after {issue_timeout}s - killing process")
                logging.warning(f"Issue #{issue_number} timed out after {issue_timeout}s - process killed")
                # ProcessPoolExecutor will kill the worker on shutdown
                return True, False, None
            except Exception as e:
                print(f"  [{language}] ❌ Error: {e}")
                logging.error(f"Error processing issue #{issue_number}: {e}")
                return True, False, None
                
    except Exception as e:
        print(f"  [{language}] ❌ Process pool error: {e}")
        logging.error(f"Process pool error for issue #{issue_number}: {e}")
        return False, False, None


def main():
    """Main function."""
    global LLM_CALL_COUNT
    
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    initialize_configuration(args)
    
    if not os.path.exists(args.input_dir):
        print(f"❌ Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    log_file = setup_logging(log_dir)
    logging.info("Strands-powered Dockerfile generation started (multiprocessing mode)")
    
    # Check for required imports
    try:
        from strands import Agent
        from strands.models.bedrock import BedrockModel
        from strands_tools import shell, file_read
        print(f"✅ Using CAB tools from: {cab_tools_path}")
    except ImportError as e:
        print("❌ Error: Strands SDK or CAB tools not found.")
        print("   Please ensure:")
        print("   1. Strands SDK is installed: pip install strands")
        print("   2. CAB tools are available at: CodeAssistBench/tools/src/strands_tools/")
        print(f"   Error: {e}")
        sys.exit(1)
    
    try:
        print("🚀 Starting Strands-powered Dockerfile generation (multiprocessing mode)...")
        print(f"⚡ Parallel processing: {args.parallel} issue(s) at a time")
        
        # Find issues to process
        dockerizable_missing = find_dockerizable_missing_dockerfile(
            dataset_directory=args.input_dir,
            dockerizable_category=args.dockerizable_category,
            target_languages=args.languages
        )
        
        if not dockerizable_missing:
            print("✅ No dockerizable issues missing Dockerfile found!")
            return
        
        print(f"📋 Found {len(dockerizable_missing)} issues to process")
        
        # Group by file for efficient I/O
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
        
        total_processed = 0
        successful_dockerfiles = 0
        successful_builds = 0
        
        # Process files sequentially (issues within files can be parallel)
        for file_key, file_info in files_to_update.items():
            file_path = file_info['file_path']
            language = file_info['language']
            issues = file_info['issues']
            
            print(f"\n📁 [{language}] Processing {file_path.name}: {len(issues)} issues")
            
            # Read the file once
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_modified = False
            
            # Process issues with parallelism using ProcessPoolExecutor
            if args.parallel > 1:
                # Use spawn context for clean processes
                ctx = multiprocessing.get_context('spawn')
                with ProcessPoolExecutor(max_workers=args.parallel, mp_context=ctx) as executor:
                    futures = {}
                    
                    for issue_item in issues:
                        issue_index = issue_item['index']
                        issue_data = issue_item['issue']
                        
                        args_tuple = (
                            issue_data, 
                            extract_repo_name_from_url(issue_data.get('url', '')),
                            str(issue_data.get('number', 'unknown')),
                            args.model_id,
                            args.max_attempts,
                            args.docker_timeout,
                            args.agent_timeout,
                            args.failure_logs_dir
                        )
                        
                        future = executor.submit(_run_agent_in_process, args_tuple)
                        futures[future] = (issue_index, issue_data)
                    
                    for future in futures:
                        issue_index, issue_data = futures[future]
                        issue_number = issue_data.get('number', 'unknown')
                        
                        try:
                            dockerfile, build_success, error = future.result(timeout=args.issue_timeout)
                            total_processed += 1
                            
                            if dockerfile:
                                successful_dockerfiles += 1
                                data[issue_index]['dockerfile'] = dockerfile
                                file_modified = True
                                
                                if build_success:
                                    successful_builds += 1
                                    print(f"  [{language}] ✅ Issue #{issue_number}: Success!")
                                else:
                                    print(f"  [{language}] ⚠️  Issue #{issue_number}: Dockerfile created but build failed")
                            else:
                                print(f"  [{language}] ❌ Issue #{issue_number}: Failed - {error}")
                                
                        except FuturesTimeoutError:
                            total_processed += 1
                            print(f"  [{language}] ⏰ Issue #{issue_number}: Timed out")
                            logging.warning(f"Issue #{issue_number} timed out")
                        except Exception as e:
                            total_processed += 1
                            print(f"  [{language}] ❌ Issue #{issue_number}: Error - {e}")
                            logging.error(f"Error processing issue #{issue_number}: {e}")
            else:
                # Sequential processing (parallel=1)
                for issue_item in issues:
                    issue_index = issue_item['index']
                    issue_data = issue_item['issue']
                    
                    processed, build_success, dockerfile = process_single_issue(
                        file_path=file_path,
                        language=language,
                        issue_index=issue_index,
                        issue_data=issue_data,
                        model_id=args.model_id,
                        max_attempts=args.max_attempts,
                        docker_timeout=args.docker_timeout,
                        agent_timeout=args.agent_timeout,
                        issue_timeout=args.issue_timeout,
                        failure_logs_dir_path=args.failure_logs_dir
                    )
                    
                    if processed:
                        total_processed += 1
                    
                    if dockerfile:
                        successful_dockerfiles += 1
                        data[issue_index]['dockerfile'] = dockerfile
                        file_modified = True
                        
                        if build_success:
                            successful_builds += 1
            
            # Save file if modified
            if file_modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"  [{language}] 💾 Saved updates to {file_path.name}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"📊 FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Issues processed: {total_processed}")
        print(f"Dockerfiles generated: {successful_dockerfiles}")
        print(f"Successful builds: {successful_builds}")
        print(f"Build success rate: {(successful_builds/successful_dockerfiles)*100 if successful_dockerfiles > 0 else 0:.1f}%")
        print(f"Log file: {log_file}")
        print("="*60)
        
        logging.info(f"Process completed. {successful_dockerfiles} Dockerfiles generated, {successful_builds} successful builds")
        
    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user")
        logging.info("Process interrupted by user")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logging.error(f"Error in main: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    # Use spawn method for multiprocessing to avoid fork issues with threads
    multiprocessing.set_start_method('spawn', force=True)
    main()
