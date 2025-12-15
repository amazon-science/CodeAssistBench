#!/usr/bin/env python3
"""
Example CLI-based custom agent for CodeAssistBench.

This demonstrates the expected interface for CLI-based custom agents.
Your custom CLI agent should:
1. Read JSON input from stdin
2. Process the prompt and generate a response
3. Output JSON response to stdout
4. Exit with code 0 for success

Usage:
    chmod +x examples/custom_agents/example_cli_agent.py
    
    python -m cab_evaluation.cli generation-dataset examples/sample_dataset.jsonl \
      --agent-framework '{"maintainer": "custom"}' \
      --custom-agent-config '{
        "type": "cli",
        "command": "examples/custom_agents/example_cli_agent.py",
        "timeout": 300,
        "metrics": {
          "enabled": true,
          "expected_fields": ["input_tokens", "output_tokens", "cost"]
        }
      }'
"""

import sys
import json
import subprocess
import time


def main():
    """Main CLI agent entry point."""
    try:
        # Read input from stdin
        input_json = sys.stdin.read()
        input_data = json.loads(input_json)
        
        system_prompt = input_data.get("system_prompt", "")
        user_prompt = input_data.get("user_prompt", "")
        repo_dir = input_data.get("repo_dir", ".")
        
        # Log to stderr for debugging (stdout is for response only)
        print(f"[CLI Agent] Processing prompt (length: {len(user_prompt)} chars)", file=sys.stderr)
        print(f"[CLI Agent] Working directory: {repo_dir}", file=sys.stderr)
        
        # Example: You would call your actual agent here
        # This is a mock implementation - replace with your actual agent logic
        
        start_time = time.time()
        
        # OPTION 1: Call another CLI tool (e.g., Claude Code, cursor, etc.)
        # Uncomment and modify for your tool:
        # result = subprocess.run(
        #     ["your-agent-command", "--prompt", user_prompt],
        #     capture_output=True,
        #     text=True,
        #     cwd=repo_dir,
        #     timeout=300
        # )
        # agent_response = result.stdout
        
        # OPTION 2: Use an API (e.g., OpenAI, Anthropic, local model)
        # import anthropic
        # client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        # message = client.messages.create(
        #     model="claude-3-5-sonnet-20241022",
        #     max_tokens=8192,
        #     messages=[{"role": "user", "content": user_prompt}]
        # )
        # agent_response = message.content[0].text
        
        # Mock response for demonstration
        agent_response = f"""Based on the code issue, here's my analysis:

**Issue Summary**: {user_prompt[:200]}...

**Solution**:
I've analyzed the repository and identified the root cause. Here's what needs to be fixed:

1. Review the relevant files in the repository
2. Apply the necessary changes
3. Test the solution

**Code Example**:
```python
# Example fix
def fixed_function():
    # Your fix here
    pass
```

This solution addresses the issue by implementing the necessary changes.
"""
        
        execution_time = time.time() - start_time
        
        # Prepare response with metrics
        response = {
            "response": agent_response,
            "exploration": "Repository explored via custom agent tools",
            "metrics": {
                "execution_time": execution_time,
                "input_tokens": len(user_prompt) // 4,  # Rough estimate
                "output_tokens": len(agent_response) // 4,  # Rough estimate
                "cost": 0.015  # Mock cost
            }
        }
        
        # Output JSON to stdout
        print(json.dumps(response), file=sys.stdout)
        
        print(f"[CLI Agent] Response generated successfully", file=sys.stderr)
        return 0
        
    except Exception as e:
        # Log error to stderr
        print(f"[CLI Agent] Error: {e}", file=sys.stderr)
        
        # Still output a valid JSON response for graceful failure
        error_response = {
            "response": f"Error processing request: {str(e)}",
            "exploration": "",
            "metrics": {}
        }
        print(json.dumps(error_response), file=sys.stdout)
        return 1


if __name__ == "__main__":
    sys.exit(main())
