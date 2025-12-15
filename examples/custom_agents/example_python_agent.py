"""
Example Python class-based custom agent for CodeAssistBench.

This demonstrates how to create a custom Python agent that integrates with CodeAssistBench.
Your custom agent must:
1. Implement the required interface methods (generate_response, get_system_prompt)
2. Optionally provide metrics via get_metrics()
3. Optionally provide exploration results via get_exploration_results()

Usage:
    python -m cab_evaluation.cli generation-dataset examples/sample_dataset.jsonl \
      --agent-framework '{"maintainer": "custom"}' \
      --custom-agent-config '{
        "type": "python_class",
        "module_path": "examples/custom_agents/example_python_agent.py",
        "class_name": "MyCustomAgent",
        "init_args": {
          "api_key": "your-api-key-here",
          "model": "your-model-name"
        },
        "metrics": {
          "enabled": true,
          "expected_fields": ["input_tokens", "output_tokens", "total_cost"]
        }
      }'
"""

import os
import time
from typing import Optional, Dict, Any


class MyCustomAgent:
    """Example custom agent implementation.
    
    This is a minimal example showing the required interface.
    You can extend this with your own model, API calls, or custom logic.
    """
    
    def __init__(self, api_key: str = None, model: str = "custom-model", **kwargs):
        """Initialize your custom agent.
        
        Args:
            api_key: API key for your service (if needed)
            model: Model identifier
            **kwargs: Additional configuration arguments
        """
        self.api_key = api_key
        self.model = model
        self.call_metrics = []
        
        print(f"[Custom Agent] Initialized with model: {model}")
    
    def get_system_prompt(self, **kwargs) -> str:
        """Return system prompt for the agent.
        
        This method is REQUIRED by CodeAssistBench.
        
        Args:
            **kwargs: Context information (repo_url, commit_hash, etc.)
            
        Returns:
            System prompt string
        """
        base_prompt = "You are a helpful coding assistant that analyzes code issues and provides solutions."
        
        # Add context if provided
        if kwargs.get('repo_url'):
            base_prompt += f"\nRepository: {kwargs['repo_url']}"
        if kwargs.get('commit_hash'):
            base_prompt += f"\nCommit: {kwargs['commit_hash']}"
        
        return base_prompt
    
    async def generate_response(
        self,
        user_prompt: str,
        system_prompt: str,
        issue_id: str = "unknown",
        **kwargs
    ) -> str:
        """Generate response to user prompt.
        
        This method is REQUIRED by CodeAssistBench.
        
        Args:
            user_prompt: User's question/prompt
            system_prompt: System prompt with context
            issue_id: Issue identifier for tracking
            **kwargs: Additional context (repo_dir, etc.)
            
        Returns:
            Generated response text
        """
        repo_dir = kwargs.get('repo_dir', '.')
        
        print(f"[Custom Agent] Generating response for issue: {issue_id}")
        print(f"[Custom Agent] Prompt length: {len(user_prompt)} chars")
        print(f"[Custom Agent] Repository: {repo_dir}")
        
        start_time = time.time()
        
        # YOUR CUSTOM LOGIC HERE
        # This is where you would call your LLM, model, or API
        
        # Example integrations:
        
        # OPTION 1: Call OpenAI API
        # import openai
        # client = openai.OpenAI(api_key=self.api_key)
        # response = client.chat.completions.create(
        #     model=self.model,
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": user_prompt}
        #     ]
        # )
        # agent_response = response.choices[0].message.content
        # input_tokens = response.usage.prompt_tokens
        # output_tokens = response.usage.completion_tokens
        
        # OPTION 2: Call Anthropic API
        # import anthropic
        # client = anthropic.Anthropic(api_key=self.api_key)
        # message = client.messages.create(
        #     model=self.model,
        #     max_tokens=8192,
        #     system=system_prompt,
        #     messages=[{"role": "user", "content": user_prompt}]
        # )
        # agent_response = message.content[0].text
        # input_tokens = message.usage.input_tokens
        # output_tokens = message.usage.output_tokens
        
        # OPTION 3: Use local model (e.g., Ollama)
        # import ollama
        # response = ollama.chat(
        #     model=self.model,
        #     messages=[
        #         {'role': 'system', 'content': system_prompt},
        #         {'role': 'user', 'content': user_prompt}
        #     ]
        # )
        # agent_response = response['message']['content']
        
        # OPTION 4: Explore repository files
        # import os
        # from pathlib import Path
        # 
        # print(f"[Custom Agent] Exploring {repo_dir}...")
        # python_files = list(Path(repo_dir).rglob("*.py"))
        # print(f"[Custom Agent] Found {len(python_files)} Python files")
        # 
        # # Read and analyze relevant files
        # for file_path in python_files[:10]:  # Limit to first 10
        #     try:
        #         content = file_path.read_text()
        #         # ... analyze content ...
        #     except:
        #         pass
        
        # Mock response for demonstration
        agent_response = f"""I've analyzed the code issue using my custom agent logic.

**Analysis**:
Based on the repository exploration and the user's question, here's my assessment:

**Question**: {user_prompt[:200]}...

**Root Cause**:
The issue appears to be related to [specific technical problem].

**Recommended Solution**:
1. Identify the problematic code section
2. Apply the following fix:

```python
# Proposed solution
def fixed_implementation():
    # Your corrected code here
    pass
```

3. Test the changes thoroughly

**Additional Notes**:
- Ensure all dependencies are properly installed
- Run tests after applying the fix
- Consider edge cases

This solution has been generated by {self.model} custom agent.
"""
        
        execution_time = time.time() - start_time
        
        # Track metrics for this call
        self.call_metrics.append({
            "issue_id": issue_id,
            "execution_time": execution_time,
            "input_length": len(user_prompt),
            "output_length": len(agent_response),
            "input_tokens": len(user_prompt) // 4,  # Rough estimate
            "output_tokens": len(agent_response) // 4,  # Rough estimate
            "total_cost": 0.018  # Mock cost
        })
        
        print(f"[Custom Agent] Response generated in {execution_time:.2f}s")
        
        return agent_response
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return metrics for CAB tracking.
        
        This method is OPTIONAL but recommended for better evaluation insights.
        
        Returns:
            Dictionary with aggregated metrics
        """
        if not self.call_metrics:
            return {
                "total_calls": 0,
                "total_execution_time": 0,
                "total_cost": 0
            }
        
        return {
            "total_calls": len(self.call_metrics),
            "total_execution_time": sum(m["execution_time"] for m in self.call_metrics),
            "total_input_tokens": sum(m["input_tokens"] for m in self.call_metrics),
            "total_output_tokens": sum(m["output_tokens"] for m in self.call_metrics),
            "total_cost": sum(m["total_cost"] for m in self.call_metrics),
            "calls": self.call_metrics
        }
    
    def get_exploration_results(self) -> str:
        """Return exploration results if your agent performs repository exploration.
        
        This method is OPTIONAL.
        
        Returns:
            String with exploration details
        """
        return "Custom agent explored repository using internal tools"


# Alternative: More advanced custom agent with actual LLM integration
class AdvancedCustomAgent:
    """Advanced example with real API integration.
    
    Demonstrates how to integrate with actual LLM APIs like OpenAI or Anthropic.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4", provider: str = "openai", **kwargs):
        """Initialize advanced custom agent.
        
        Args:
            api_key: API key for LLM provider
            model: Model identifier
            provider: Provider name ("openai", "anthropic", "local")
            **kwargs: Additional configuration
        """
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.call_metrics = []
        
        # Initialize provider client
        if provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("OpenAI package required. Install: pip install openai")
        
        elif provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("Anthropic package required. Install: pip install anthropic")
        
        print(f"[Advanced Agent] Initialized {provider} with model: {model}")
    
    def get_system_prompt(self, **kwargs) -> str:
        """Get system prompt."""
        return "You are an expert software engineer helping debug code issues."
    
    async def generate_response(
        self,
        user_prompt: str,
        system_prompt: str,
        issue_id: str = "unknown",
        **kwargs
    ) -> str:
        """Generate response using actual LLM API."""
        
        start_time = time.time()
        
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                agent_response = response.choices[0].message.content
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                
            elif self.provider == "anthropic":
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=8192,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                agent_response = message.content[0].text
                input_tokens = message.usage.input_tokens
                output_tokens = message.usage.output_tokens
            
            else:
                agent_response = "Provider not supported"
                input_tokens = 0
                output_tokens = 0
            
            execution_time = time.time() - start_time
            
            # Track metrics
            self.call_metrics.append({
                "issue_id": issue_id,
                "execution_time": execution_time,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_cost": self._calculate_cost(input_tokens, output_tokens)
            })
            
            return agent_response
            
        except Exception as e:
            print(f"[Advanced Agent] Error: {e}", file=sys.stderr)
            return f"Error generating response: {str(e)}"
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate approximate cost based on tokens."""
        # Adjust these rates for your provider/model
        input_cost_per_1m = 3.0  # $3 per 1M input tokens
        output_cost_per_1m = 15.0  # $15 per 1M output tokens
        
        input_cost = (input_tokens / 1_000_000) * input_cost_per_1m
        output_cost = (output_tokens / 1_000_000) * output_cost_per_1m
        
        return input_cost + output_cost
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return aggregated metrics."""
        if not self.call_metrics:
            return {"total_calls": 0}
        
        return {
            "total_calls": len(self.call_metrics),
            "total_execution_time": sum(m["execution_time"] for m in self.call_metrics),
            "total_input_tokens": sum(m["input_tokens"] for m in self.call_metrics),
            "total_output_tokens": sum(m["output_tokens"] for m in self.call_metrics),
            "total_cost": sum(m["total_cost"] for m in self.call_metrics),
            "calls": self.call_metrics
        }
