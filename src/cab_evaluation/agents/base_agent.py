"""Base agent implementation for CAB evaluation."""

import os
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Optional, Any
from datetime import datetime

from ..core.config import CABConfig, ModelConfig
from ..core.exceptions import AgentError, LLMError, InputTooLongError
from ..core.models import ConversationMessage
from ..prompts.prompt_manager import PromptManager

logger = logging.getLogger(__name__)

# Global LLM call counter for tracking calls per agent per issue
llm_call_counter = defaultdict(lambda: defaultdict(int))
call_counter_lock = threading.Lock()


class BaseAgent(ABC):
    """Base class for all CAB evaluation agents."""
    
    def __init__(
        self,
        agent_type: str,
        model_name: str = "sonnet",
        config: Optional[CABConfig] = None,
        prompt_manager: Optional[PromptManager] = None,
        **kwargs
    ):
        """Initialize base agent.
        
        Args:
            agent_type: Type of agent (maintainer, user, judge)
            model_name: Name of model to use
            config: Configuration object
            prompt_manager: Prompt manager instance
            **kwargs: Additional arguments including use_strands and read_only
        """
        self.agent_type = agent_type
        self.model_name = model_name
        self.config = config or CABConfig()
        self.prompt_manager = prompt_manager or PromptManager(self.config.prompts_dir)
        
        # Get model configuration
        self.model_config = self.config.get_model_config(model_name)
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{agent_type}")
        
        # Initialize LLM service (import here to avoid circular imports)
        self.llm_service = None
        
        # Strands framework integration
        self._strands_agent = None
        self._use_strands = kwargs.get('use_strands', True)  # Enable Strands by default
        self._read_only = kwargs.get('read_only', False)
        self._setup_strands_environment()
    
    @abstractmethod
    def get_system_prompt(self, **kwargs) -> str:
        """Get system prompt for this agent type.
        
        Args:
            **kwargs: Additional context for prompt generation
            
        Returns:
            System prompt string
        """
        pass
    
    @abstractmethod
    async def generate_response(
        self,
        user_prompt: str,
        system_prompt: str,
        issue_id: str = "unknown",
        **kwargs
    ) -> str:
        """Generate response using the agent's model.
        
        Args:
            user_prompt: User input prompt
            system_prompt: System prompt to use
            issue_id: Issue ID for tracking
            **kwargs: Additional arguments
            
        Returns:
            Generated response text
        """
        pass
    
    def increment_call_counter(self, issue_id: str):
        """Increment LLM call counter for this agent and issue."""
        with call_counter_lock:
            llm_call_counter[issue_id][self.agent_type] += 1
            total_calls = sum(llm_call_counter[issue_id].values())
            self.logger.info(
                f"LLM Call #{total_calls} for issue {issue_id}: Agent {self.agent_type} "
                f"(agent total: {llm_call_counter[issue_id][self.agent_type]})"
            )
    
    def get_call_statistics(self, issue_id: str) -> Dict[str, int]:
        """Get LLM call statistics for an issue."""
        with call_counter_lock:
            return dict(llm_call_counter[issue_id])
    
    def reset_call_counter(self, issue_id: str):
        """Reset call counter for an issue."""
        with call_counter_lock:
            llm_call_counter[issue_id] = defaultdict(int)
    
    async def call_llm(
        self,
        user_prompt: str,
        system_prompt: str,
        issue_id: str = "unknown",
        **kwargs
    ) -> str:
        """Call LLM with retry logic and error handling.
        
        Args:
            user_prompt: User input prompt
            system_prompt: System prompt
            issue_id: Issue ID for tracking
            **kwargs: Additional arguments
            
        Returns:
            LLM response text
            
        Raises:
            InputTooLongError: If input exceeds model context window
            LLMError: If LLM call fails after retries
        """
        # Initialize LLM service if needed (lazy initialization to avoid circular imports)
        if self.llm_service is None:
            from .llm_service import LLMService
            self.llm_service = LLMService(self.config)
        
        # Increment counter
        self.increment_call_counter(issue_id)
        
        # Log prompts
        self.logger.info(f"===== SYSTEM PROMPT =====\n{system_prompt}\n")
        self.logger.info(f"===== USER PROMPT =====\n{user_prompt}\n")
        
        start_time = time.time()
        self.logger.info(
            f"Calling {self.model_config.name} model "
            f"(prompt length: {len(user_prompt)}, agent: {self.agent_type}, issue: {issue_id})"
        )
        
        try:
            response = await self.llm_service.call_model(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                model_config=self.model_config,
                agent_type=self.agent_type,
                issue_id=issue_id
            )
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"{self.model_config.name} model responded in {elapsed_time:.2f} seconds")
            self.logger.info(f"===== LLM RESPONSE =====\n{response}\n")
            
            return response
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"LLM call failed after {elapsed_time:.2f}s: {e}")
            raise
    
    def create_conversation_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ConversationMessage:
        """Create a conversation message for this agent.
        
        Args:
            content: Message content
            metadata: Optional metadata
            
        Returns:
            ConversationMessage instance
        """
        return ConversationMessage(
            role=self.agent_type,
            content=content,
            metadata=metadata or {}
        )
    
    def _setup_strands_environment(self):
        """Setup Strands environment and tools."""
        if not self._use_strands:
            return
            
        # Set environment variable for execute_bash based on read-only mode
        if not self._read_only:
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
        if not self._use_strands:
            return None
            
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
            self.logger.warning(f"Failed to import Strands components: {e}. Falling back to standard LLM service.")
            self._use_strands = False
            return None
        
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
        if self._read_only:
            # Read-only mode: only allow safe read operations
            tools = [execute_bash, fs_read, thinking]
            self.logger.info(f"🔒 {self.agent_type} agent in READ-ONLY Strands mode (execute_bash [safe commands only], fs_read, thinking)")
        else:
            # Default mode: all tools enabled including write operations
            tools = [execute_bash, fs_read, fs_write, report_issue, use_aws, thinking]
            self.logger.info(f"✅ {self.agent_type} agent in WRITE-ALLOWED Strands mode (execute_bash, fs_read, fs_write, report_issue, use_aws, thinking)")
        
        # Create cache point hook to enable prompt caching
        try:
            cache_hook = CachePointHook()
        except:
            # Fallback if hooks not available
            cache_hook = None
        
        # Create Strands agent
        strands_agent = Agent(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            hooks=[cache_hook] if cache_hook else []
        )
        
        self.logger.info(f"Strands agent created for {self.agent_type} with model {model_id}")
        return strands_agent
    
    async def call_llm_with_strands(
        self,
        user_prompt: str,
        system_prompt: str,
        issue_id: str = "unknown",
        **kwargs
    ) -> str:
        """Call LLM using Strands framework with enhanced capabilities.
        
        Args:
            user_prompt: User input prompt
            system_prompt: System prompt
            issue_id: Issue ID for tracking
            **kwargs: Additional arguments
            
        Returns:
            LLM response text
            
        Raises:
            AgentError: If Strands agent execution fails
        """
        if not self._use_strands:
            # Fallback to standard LLM service
            return await self.call_llm(user_prompt, system_prompt, issue_id, **kwargs)
        
        # Increment counter
        self.increment_call_counter(issue_id)
        
        # Build Strands agent if not already created
        if self._strands_agent is None:
            self._strands_agent = self._build_strands_agent(system_prompt)
            if self._strands_agent is None:
                # Fallback to standard LLM service if Strands failed to initialize
                return await self.call_llm(user_prompt, system_prompt, issue_id, **kwargs)
        
        # Log prompts
        self.logger.info(f"===== SYSTEM PROMPT =====\n{system_prompt}\n")
        self.logger.info(f"===== USER PROMPT =====\n{user_prompt}\n")
        
        start_time = time.time()
        self.logger.info(
            f"Calling Strands agent with {self.model_name} model "
            f"(prompt length: {len(user_prompt)}, agent: {self.agent_type}, issue: {issue_id})"
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
            self._log_strands_metrics_summary()
            
            self.logger.info(f"===== STRANDS RESPONSE =====\n{response}\n")
            
            return str(response)
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"Strands agent call failed after {elapsed_time:.2f}s: {e}")
            # Fallback to standard LLM service
            self.logger.info("Falling back to standard LLM service")
            return await self.call_llm(user_prompt, system_prompt, issue_id, **kwargs)
    
    def _log_strands_metrics_summary(self):
        """Log comprehensive metrics from Strands agent."""
        if self._strands_agent is None:
            return
        
        try:
            metrics_summary = self._strands_agent.event_loop_metrics.get_summary()
            usage = metrics_summary["accumulated_usage"]
            
            self.logger.info(f"\n📊 Strands Token Usage for {self.agent_type}:")
            self.logger.info(f"  Input Tokens:        {usage.get('inputTokens', 0):>10,}")
            self.logger.info(f"  Output Tokens:       {usage.get('outputTokens', 0):>10,}")
            self.logger.info(f"  Total Tokens:        {usage.get('totalTokens', 0):>10,}")
            
            # Cache metrics
            if usage.get('cacheReadInputTokens', 0) > 0 or usage.get('cacheWriteInputTokens', 0) > 0:
                self.logger.info(f"  Cache Read Tokens:   {usage.get('cacheReadInputTokens', 0):>10,}")
                self.logger.info(f"  Cache Write Tokens:  {usage.get('cacheWriteInputTokens', 0):>10,}")
            
            # Performance
            self.logger.info(f"⏱️  Performance:")
            self.logger.info(f"  Cycles:              {metrics_summary['total_cycles']:>10}")
            self.logger.info(f"  Bedrock Latency:     {metrics_summary['accumulated_metrics']['latencyMs']:>10}ms")
            
        except Exception as e:
            self.logger.warning(f"Error logging Strands metrics: {e}")
