"""Strands agent implementation for CAB evaluation."""

import os
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path

from .base_agent import BaseAgent
from ..core.config import CABConfig, ModelConfig
from ..core.exceptions import AgentError, LLMError
from ..prompts.prompt_manager import PromptManager

import logging

logger = logging.getLogger(__name__)

# Custom JSON encoder to handle non-serializable objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

# Pricing for Claude Sonnet 4 (per 1M tokens) - Updated for Nov 2024
PRICING = {
    "input_base": 3.00,           # Base input price per 1M tokens
    "output": 15.00,              # Output price per 1M tokens
    "cache_write": 3.75,          # Cache write price per 1M tokens (1.25x base)
    "cache_read": 0.30,           # Cache read price per 1M tokens (0.1x base)
}


class StrandsAgent(BaseAgent):
    """Agent that uses the Strands framework for enhanced tool capabilities."""
    
    def __init__(
        self,
        model_name: str = "sonnet37",
        config: Optional[CABConfig] = None,
        prompt_manager: Optional[PromptManager] = None,
        read_only: bool = False,
        **kwargs
    ):
        """Initialize Strands agent.
        
        Args:
            model_name: Model to use (defaults to sonnet37)
            config: CAB configuration
            prompt_manager: Prompt manager
            read_only: If True, only read-only tools are enabled
            **kwargs: Additional arguments for base agent
        """
        # Set read_only first so it's available during initialization
        self.read_only = read_only
        
        super().__init__(
            agent_type="strands",
            model_name=model_name,
            config=config,
            prompt_manager=prompt_manager
        )
        
        self._strands_agent = None
        self._strands_tools = None
        self._setup_strands_environment()
        
    def _setup_strands_environment(self):
        """Setup Strands environment and tools."""
        # Set environment variable for execute_bash based on read-only mode
        if not self.read_only:
            # Enable unrestricted bash execution in write-allowed mode
            os.environ["EXECUTE_BASH_UNRESTRICTED"] = "true"
        else:
            # Ensure restricted mode in read-only mode
            os.environ["EXECUTE_BASH_UNRESTRICTED"] = "false"
    
    def _get_strands_model_id(self, model_name: str) -> str:
        """Map CAB model names to Strands BedrockModel IDs.
        
        Args:
            model_name: CAB model name
            
        Returns:
            Strands model ID
        """
        # Model mapping from CAB to Strands/Bedrock
        model_mapping = {
            "haiku": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            "sonnet": "us.anthropic.claude-3-7-sonnet-20250219-v1:0", 
            "sonnet37": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "thinking": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "deepseek": "us.deepseek.r1-v1:0",
            "llama": "us.meta.llama3-3-70b-instruct-v1:0",
            # Default fallback to sonnet37 equivalent
            "default": "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        }
        
        return model_mapping.get(model_name, model_mapping["default"])
    
    def _build_strands_agent(self, system_prompt: str):
        """Build and configure a Strands Agent instance.
        
        Args:
            system_prompt: System prompt for the agent
            
        Returns:
            Configured Strands Agent
        """
        try:
            # Import Strands components
            from strands import Agent
            from strands.models import BedrockModel
            from strands.hooks import HookProvider, MessageAddedEvent
            
            # Import Strands tools
            from tools.src.strands_tools import (
                execute_bash, fs_read, fs_write, report_issue, use_aws, thinking
            )
        except ImportError as e:
            raise AgentError(f"Failed to import Strands components: {e}. Make sure Strands is installed and available.")
        
        # Get model ID for this model
        model_id = self._get_strands_model_id(self.model_name)
        
        # Create Bedrock model with caching enabled
        model = BedrockModel(
            model_id=model_id,
            region_name="us-west-2",
            max_retries=1000,
            cache_prompt="default",  # Cache system prompt
            cache_tools="default"    # Cache tools
        )
        
        # Select tools based on read-only mode
        if self.read_only:
            # Read-only mode: only allow safe read operations
            tools = [execute_bash, fs_read, thinking]
            self.logger.info("🔒 Strands agent in READ-ONLY mode (execute_bash [safe commands only], fs_read, thinking)")
        else:
            # Default mode: all tools enabled including write operations
            tools = [execute_bash, fs_read, fs_write, report_issue, use_aws, thinking]
            self.logger.info("✅ Strands agent in WRITE-ALLOWED mode (execute_bash, fs_read, fs_write, report_issue, use_aws, thinking)")
        
        # Create cache point hook to enable prompt caching
        cache_hook = CachePointHook()
        
        # Create Strands agent
        strands_agent = Agent(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            hooks=[cache_hook]
        )
        
        self.logger.info(f"Strands agent created with model {model_id}")
        return strands_agent
    
    def get_system_prompt(self, **kwargs) -> str:
        """Get system prompt for this agent type.
        
        Args:
            **kwargs: Additional context for prompt generation
            
        Returns:
            System prompt string
        """
        # Use a generic system prompt since Strands agent can be flexible
        base_prompt = """You are an AI assistant with comprehensive tool capabilities including:
- File system operations (read/write files and directories)
- Bash command execution with safety controls
- Issue reporting and AWS service interaction
- Advanced reasoning with the thinking tool

You have access to the filesystem and can execute commands to help solve programming problems.
Answer questions accurately and provide practical solutions with code when appropriate.
Be concise but thorough in your explanations."""
        
        # Add specific context if provided
        if kwargs.get('context'):
            base_prompt += f"\n\nAdditional context: {kwargs['context']}"
            
        return base_prompt
    
    async def generate_response(
        self,
        user_prompt: str,
        system_prompt: str,
        issue_id: str = "unknown",
        **kwargs
    ) -> str:
        """Generate response using Strands Agent.
        
        Args:
            user_prompt: User input prompt
            system_prompt: System prompt to use
            issue_id: Issue ID for tracking
            **kwargs: Additional arguments
            
        Returns:
            Generated response text
            
        Raises:
            AgentError: If Strands agent execution fails
        """
        # Increment counter
        self.increment_call_counter(issue_id)
        
        # Build Strands agent if not already created or system prompt changed
        if self._strands_agent is None:
            self._strands_agent = self._build_strands_agent(system_prompt)
        
        # Log prompts
        self.logger.info(f"===== SYSTEM PROMPT =====\n{system_prompt}\n")
        self.logger.info(f"===== USER PROMPT =====\n{user_prompt}\n")
        
        start_time = time.time()
        self.logger.info(
            f"Calling Strands agent with {self.model_name} model "
            f"(prompt length: {len(user_prompt)}, issue: {issue_id})"
        )
        
        # Add minimal instruction to encourage concise responses
        enhanced_prompt = user_prompt + "\n\n<implicitInstruction>\n- Write only the ABSOLUTE MINIMAL amount of code needed to address the requirement correctly. Avoid verbose implementations and any code that doesn't directly contribute to the solution\n</implicitInstruction>"
        
        try:
            # Execute with retry logic for throttling
            response = None
            for attempt in range(300):
                try:
                    response = self._strands_agent(enhanced_prompt)
                    break
                except Exception as e:
                    if "throttl" in str(e).lower() and attempt < 299:
                        self.logger.warning(f"Throttled. Retry {attempt + 1}/300 in 19s...")
                        time.sleep(19)
                    else:
                        raise AgentError(f"Strands agent execution failed after {attempt + 1} attempts: {e}")
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Strands agent responded in {elapsed_time:.2f} seconds")
            
            # Log metrics summary
            self._log_metrics_summary()
            
            self.logger.info(f"===== STRANDS RESPONSE =====\n{response}\n")
            
            return str(response)
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"Strands agent call failed after {elapsed_time:.2f}s: {e}")
            raise AgentError(f"Strands agent execution failed: {e}")
    
    def _log_metrics_summary(self):
        """Log comprehensive metrics from Strands agent."""
        if self._strands_agent is None:
            return
        
        try:
            metrics_summary = self._strands_agent.event_loop_metrics.get_summary()
            usage = metrics_summary["accumulated_usage"]
            
            # Calculate pricing and cache efficiency
            pricing_info = self._calculate_cost(usage)
            cache_efficiency = self._calculate_cache_efficiency(usage)
            
            self.logger.info("\n" + "="*60)
            self.logger.info("STRANDS AGENT METRICS SUMMARY")
            self.logger.info("="*60)
            
            # Token usage
            self.logger.info("\n📊 Token Usage:")
            self.logger.info(f"  Input Tokens:        {usage.get('inputTokens', 0):>10,}")
            self.logger.info(f"  Output Tokens:       {usage.get('outputTokens', 0):>10,}")
            self.logger.info(f"  Total Tokens:        {usage.get('totalTokens', 0):>10,}")
            
            # Cache metrics
            if usage.get('cacheReadInputTokens', 0) > 0 or usage.get('cacheWriteInputTokens', 0) > 0:
                self.logger.info(f"\n  ✓ CACHE IS WORKING:")
                self.logger.info(f"  Cache Read Tokens:   {usage.get('cacheReadInputTokens', 0):>10,}")
                self.logger.info(f"  Cache Write Tokens:  {usage.get('cacheWriteInputTokens', 0):>10,}")
                self.logger.info(f"  Cache Hit Rate:      {cache_efficiency['cache_hit_rate_percent']:>9.2f}%")
            
            # Pricing
            self.logger.info(f"\n💰 Cost Breakdown:")
            self.logger.info(f"  Total Cost:          ${pricing_info['total_cost']:>10.6f}")
            if pricing_info['cache_read_cost'] > 0:
                self.logger.info(f"  Cache Savings:       ${cache_efficiency['cache_savings_usd']:>10.6f}")
            
            # Performance
            self.logger.info(f"\n⏱️  Performance:")
            self.logger.info(f"  Cycles:              {metrics_summary['total_cycles']:>10}")
            self.logger.info(f"  Bedrock Latency:     {metrics_summary['accumulated_metrics']['latencyMs']:>10}ms")
            
        except Exception as e:
            self.logger.warning(f"Error logging Strands metrics: {e}")
    
    def _calculate_cost(self, usage: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate the cost based on token usage.
        
        Args:
            usage: Dictionary containing token counts
            
        Returns:
            Dictionary with detailed cost breakdown
        """
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)
        cache_read_tokens = usage.get("cacheReadInputTokens", 0)
        cache_write_tokens = usage.get("cacheWriteInputTokens", 0)
        
        # Calculate costs (convert from per 1M to actual)
        input_cost = (input_tokens / 1_000_000) * PRICING["input_base"]
        output_cost = (output_tokens / 1_000_000) * PRICING["output"]
        cache_write_cost = (cache_write_tokens / 1_000_000) * PRICING["cache_write"]
        cache_read_cost = (cache_read_tokens / 1_000_000) * PRICING["cache_read"]
        
        total_cost = input_cost + output_cost + cache_write_cost + cache_read_cost
        
        return {
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "cache_write_cost": round(cache_write_cost, 6),
            "cache_read_cost": round(cache_read_cost, 6),
            "total_cost": round(total_cost, 6),
        }
    
    def _calculate_cache_efficiency(self, usage: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate cache hit rate and efficiency metrics.
        
        Args:
            usage: Dictionary containing token counts
            
        Returns:
            Dictionary with cache efficiency metrics
        """
        cache_read_tokens = usage.get("cacheReadInputTokens", 0)
        cache_write_tokens = usage.get("cacheWriteInputTokens", 0)
        input_tokens = usage.get("inputTokens", 0)
        
        total_input_with_cache = input_tokens + cache_read_tokens + cache_write_tokens
        
        cache_hit_rate = 0.0
        if total_input_with_cache > 0:
            cache_hit_rate = (cache_read_tokens / total_input_with_cache) * 100
        
        # Calculate savings from cache
        # Cache read is 10% cost of normal input, so 90% savings
        normal_cost_for_cached = (cache_read_tokens / 1_000_000) * PRICING["input_base"]
        actual_cache_cost = (cache_read_tokens / 1_000_000) * PRICING["cache_read"]
        cache_savings = normal_cost_for_cached - actual_cache_cost
        
        return {
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
            "cache_savings_usd": round(cache_savings, 6),
            "total_input_tokens_with_cache": total_input_with_cache,
        }
    
    def save_interaction_log(
        self,
        log_dir: str,
        issue_id: str,
        system_prompt: str,
        query: str,
        response: str,
        start_time: float,
        end_time: float,
    ) -> str:
        """Save comprehensive interaction log with metrics.
        
        Args:
            log_dir: Directory to save logs
            issue_id: Issue ID
            system_prompt: System prompt content
            query: User query
            response: Agent response
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            Path to saved log file
        """
        if self._strands_agent is None:
            self.logger.warning("No Strands agent available for logging")
            return ""
        
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        logfile = os.path.join(log_dir, f"strands_interaction_{ts}_{issue_id}.json")
        
        # Get metrics from agent
        metrics_summary = self._strands_agent.event_loop_metrics.get_summary()
        usage = metrics_summary["accumulated_usage"]
        
        # Calculate pricing and cache efficiency
        pricing_info = self._calculate_cost(usage)
        cache_efficiency = self._calculate_cache_efficiency(usage)
        
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "issue_id": issue_id,
            "model_name": self.model_name,
            "read_only_mode": self.read_only,
            "system_prompt": system_prompt,
            "query": query,
            "response": str(response),
            "conversation_history": getattr(self._strands_agent, "messages", []),
            
            # Token usage metrics
            "token_metrics": {
                "accumulated_usage": usage,
                "inputTokens": usage.get("inputTokens", 0),
                "outputTokens": usage.get("outputTokens", 0),
                "totalTokens": usage.get("totalTokens", 0),
                "cacheReadInputTokens": usage.get("cacheReadInputTokens", 0),
                "cacheWriteInputTokens": usage.get("cacheWriteInputTokens", 0),
            },
            
            # Pricing information
            "pricing": pricing_info,
            
            # Cache performance
            "cache_performance": cache_efficiency,
            
            # Performance metrics
            "performance_metrics": {
                "total_execution_time_secs": round(end_time - start_time, 3),
                "total_cycles": metrics_summary["total_cycles"],
                "total_cycle_duration_secs": round(metrics_summary["total_duration"], 3),
                "average_cycle_time_secs": round(metrics_summary["average_cycle_time"], 3),
                "bedrock_latency_ms": metrics_summary["accumulated_metrics"]["latencyMs"],
            },
            
            # Tool usage metrics
            "tool_usage": metrics_summary["tool_usage"],
            
            # Execution traces
            "traces": metrics_summary["traces"],
        }
        
        # Write full log with safe encoding
        with open(logfile, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, cls=CustomJSONEncoder)
        
        self.logger.info(f"Strands interaction log saved to: {logfile}")
        return logfile


# Cache Point Hook implementation (from QCLI.py)
try:
    from strands.hooks import HookProvider, MessageAddedEvent
    
    class CachePointHook(HookProvider):
        """Hook that adds cache points to enable prompt caching.
        
        This hook runs after each message is added and ensures only the last user
        message with tool results has a cache point (following QDevScience pattern).
        """
        
        def message_added(self, event: MessageAddedEvent) -> None:
            """Called after a message is added to the agent's message history.
            
            Args:
                event: Event containing the agent instance and newly added message
            """
            agent = event.agent
            message = event.message
            
            logger.debug(f"[HOOK] message_added called: role={message.get('role')}, content_blocks={len(message.get('content', []))}")
            
            # Only process user messages with tool results (like QDevScience)
            if message.get("role") != "user":
                logger.debug(f"[HOOK] Skipping non-user message")
                return
            
            # Check if this message contains tool results
            has_tool_results = any(
                "toolResult" in block 
                for block in message.get("content", [])
            )
            
            logger.debug(f"[HOOK] User message has_tool_results={has_tool_results}")
            
            if not has_tool_results:
                logger.debug(f"[HOOK] Skipping user message without tool results")
                return  # Skip regular user messages
            
            # Remove cache points from ALL messages (including this one)
            removed_count = 0
            for msg in agent.messages:
                if "content" in msg:
                    before = len(msg["content"])
                    msg["content"] = [
                        block for block in msg["content"]
                        if "cachePoint" not in block
                    ]
                    removed_count += before - len(msg["content"])
            
            logger.debug(f"[HOOK] Removed {removed_count} cache points from all messages")
            
            # Find the last user message with tool results and add cache point
            last_tool_result_msg = None
            for msg in reversed(agent.messages):
                if msg.get("role") == "user":
                    if any("toolResult" in block for block in msg.get("content", [])):
                        last_tool_result_msg = msg
                        break
            
            if last_tool_result_msg and "content" in last_tool_result_msg:
                last_tool_result_msg["content"].append({"cachePoint": {"type": "default"}})
                logger.debug(f"✓ Cache point added to last tool result message (total messages: {len(agent.messages)})")
            else:
                logger.debug(f"[HOOK] ERROR: Could not find last tool result message")

except ImportError:
    # Fallback if Strands hooks are not available
    class CachePointHook:
        def message_added(self, event):
            pass
