#!/usr/bin/env python3
"""
Docker-based custom agent wrapper for CodeAssistBench.

This script runs inside a Docker container and interfaces with CodeAssistBench.
The container receives input via CAB_INPUT environment variable or stdin.
"""

import os
import sys
import json
import time


def generate_response(user_prompt: str, system_prompt: str, repo_dir: str) -> str:
    """Generate response using your custom logic.
    
    Args:
        user_prompt: User's prompt/question
        system_prompt: System prompt with context
        repo_dir: Repository directory (mounted at /workspace)
        
    Returns:
        Generated response text
    """
    # This is where you would integrate your actual agent/model
    # Examples:
    
    # OPTION 1: Call local model
    # import ollama
    # response = ollama.chat(model='llama2', messages=[
    #     {'role': 'system', 'content': system_prompt},
    #     {'role': 'user', 'content': user_prompt}
    # ])
    # return response['message']['content']
    
    # OPTION 2: Call API
    # import anthropic
    # client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    # message = client.messages.create(
    #     model="claude-3-5-sonnet-20241022",
    #     max_tokens=8192,
    #     system=system_prompt,
    #     messages=[{"role": "user", "content": user_prompt}]
    # )
    # return message.content[0].text
    
    # OPTION 3: Use local files in repository
    # You have read access to /workspace (mounted repo)
    # import os
    # files = os.listdir(repo_dir)
    # ... analyze files ...
    
    # Mock response for demonstration
    return f"""I've analyzed the code issue in the repository at {repo_dir}.

**Analysis**:
The issue requires careful examination of the codebase. Based on the repository structure 
and the question, here's my recommendation:

1. **Root Cause**: [Describe the issue]
2. **Solution**: [Provide the fix]
3. **Implementation**: [Code changes needed]

**Recommended Changes**:
```python
# Example code fix
def improved_function():
    # Your improved implementation
    pass
```

This solution should resolve the reported issue.
"""


def main():
    """Main entry point for Docker agent."""
    try:
        # Try to read from environment variable first
        input_json = os.getenv("CAB_INPUT")
        
        # Fallback to stdin if env var not set
        if not input_json:
            print("[Docker Agent] Reading from stdin...", file=sys.stderr)
            input_json = sys.stdin.read()
        else:
            print("[Docker Agent] Reading from CAB_INPUT env var", file=sys.stderr)
        
        # Parse input
        input_data = json.loads(input_json)
        
        system_prompt = input_data.get("system_prompt", "")
        user_prompt = input_data.get("user_prompt", "")
        repo_dir = input_data.get("repo_dir", "/workspace")
        
        print(f"[Docker Agent] Processing prompt (length: {len(user_prompt)} chars)", file=sys.stderr)
        print(f"[Docker Agent] Repository: {repo_dir}", file=sys.stderr)
        
        # Generate response
        start_time = time.time()
        agent_response = generate_response(user_prompt, system_prompt, repo_dir)
        execution_time = time.time() - start_time
        
        # Prepare output with metrics
        output = {
            "response": agent_response,
            "exploration": f"Explored repository at {repo_dir}",
            "metrics": {
                "execution_time": execution_time,
                "input_tokens": len(user_prompt) // 4,  # Rough estimate
                "output_tokens": len(agent_response) // 4,  # Rough estimate
                "cost": 0.020,  # Mock cost
                "docker_container": True
            }
        }
        
        # Output JSON to stdout
        print(json.dumps(output), file=sys.stdout)
        
        print(f"[Docker Agent] Success in {execution_time:.2f}s", file=sys.stderr)
        return 0
        
    except Exception as e:
        print(f"[Docker Agent] Error: {e}", file=sys.stderr)
        
        # Output valid JSON even on error
        error_output = {
            "response": f"Error in Docker agent: {str(e)}",
            "exploration": "",
            "metrics": {}
        }
        print(json.dumps(error_output), file=sys.stdout)
        return 1


if __name__ == "__main__":
    sys.exit(main())
