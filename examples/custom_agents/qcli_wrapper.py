#!/usr/bin/env python3
"""
Q-CLI wrapper for CodeAssistBench custom agent interface.

This wrapper enables Q-CLI to be used as a custom maintainer agent in CodeAssistBench.
It translates between CAB's JSON interface and Q-CLI's command-line interface.
"""

import sys
import json
import subprocess
import time
import logging
from pathlib import Path

# Configure logging to stderr (stdout reserved for JSON response)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] Q-CLI Wrapper: %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


def run_qcli(prompt: str, repo_dir: str, timeout: int = 300) -> dict:
    """
    Execute Q-CLI with the given prompt.
    
    Args:
        prompt: Combined system + user prompt
        repo_dir: Repository directory path
        timeout: Command timeout in seconds
        
    Returns:
        Dictionary with response and metrics
    """
    start_time = time.time()
    
    try:
        logger.info(f"Running Q-CLI in directory: {repo_dir}")
        logger.info(f"Prompt length: {len(prompt)} characters")
        
        # Construct Kiro CLI command (Q-CLI is now Kiro CLI)
        # Kiro CLI format: kiro-cli chat [INPUT] --no-interactive --trust-all-tools
        qcli_cmd = ["kiro-cli", "chat", prompt, "--no-interactive", "--trust-all-tools"]
        
        logger.info(f"Executing: kiro-cli chat <prompt> --no-interactive --trust-all-tools")
        
        # Execute Q-CLI
        result = subprocess.run(
            qcli_cmd,
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"✅ Q-CLI succeeded in {elapsed_time:.2f}s")
            
            # Q-CLI output is the response
            response_text = result.stdout.strip()
            
            # Parse any metrics from stderr if available
            metrics = {
                "execution_time": elapsed_time,
                "tool": "q-cli",
                "success": True
            }
            
            # Try to extract token info from stderr if Q-CLI provides it
            if result.stderr:
                logger.info(f"Q-CLI stderr output: {result.stderr[:500]}")
                # Q-CLI might log metrics to stderr - attempt to parse
                try:
                    for line in result.stderr.split('\n'):
                        if 'token' in line.lower():
                            logger.info(f"Found token info: {line}")
                except:
                    pass
            
            return {
                "response": response_text,
                "exploration": f"Q-CLI executed in {elapsed_time:.2f}s",
                "metrics": metrics
            }
        else:
            error_msg = result.stderr or "No error output"
            logger.error(f"❌ Q-CLI failed with return code {result.returncode}")
            logger.error(f"Error: {error_msg[:500]}")
            
            raise Exception(f"Q-CLI failed: {error_msg}")
            
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        logger.error(f"⏰ Q-CLI timeout after {elapsed_time:.2f}s")
        raise Exception(f"Q-CLI timeout after {timeout}s")
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"⚠️ Q-CLI error: {e}")
        raise


def main():
    """Main entry point for Q-CLI wrapper."""
    try:
        logger.info("=== Q-CLI Wrapper Started ===")
        
        # Read JSON input from stdin
        input_data = json.load(sys.stdin)
        logger.info("📥 Input received from CodeAssistBench")
        
        # Extract required fields
        system_prompt = input_data.get("system_prompt", "")
        user_prompt = input_data.get("user_prompt", "")
        repo_dir = input_data.get("repo_dir", ".")
        
        logger.info(f"System prompt length: {len(system_prompt)}")
        logger.info(f"User prompt length: {len(user_prompt)}")
        
        # Combine prompts for Q-CLI
        # Q-CLI works best with a single combined prompt
        combined_prompt = f"""{system_prompt}

---

{user_prompt}"""
        
        # Execute Q-CLI
        result = run_qcli(
            prompt=combined_prompt,
            repo_dir=repo_dir,
            timeout=300
        )
        
        # Output JSON response to stdout
        json.dump(result, sys.stdout, indent=2)
        logger.info("✅ Response sent to CodeAssistBench")
        
        return 0
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse input JSON: {e}")
        # Still output valid JSON even on error
        error_response = {
            "response": "Error: Failed to parse input JSON",
            "exploration": "",
            "metrics": {"success": False, "error": str(e)}
        }
        json.dump(error_response, sys.stdout, indent=2)
        return 1
        
    except Exception as e:
        logger.error(f"Wrapper error: {e}")
        # Still output valid JSON even on error
        error_response = {
            "response": f"Error executing Q-CLI: {str(e)}",
            "exploration": "",
            "metrics": {"success": False, "error": str(e)}
        }
        json.dump(error_response, sys.stdout, indent=2)
        return 1


if __name__ == "__main__":
    exit(main())
