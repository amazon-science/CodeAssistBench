"""Custom maintainer agent implementation supporting CLI, Docker, and Python class interfaces."""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from dataclasses import asdict

from .base_agent import BaseAgent
from ..core.models import IssueData, ConversationMessage
from ..core.exceptions import AgentError
from ..core.config import CABConfig, CustomAgentConfig
from ..prompts.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class CustomMaintainerAgent(BaseAgent):
    """Universal adapter for custom maintainer agents.
    
    Supports three types of custom agents:
    1. CLI - External executable/script
    2. Docker - Containerized agent
    3. Python Class - Local Python module
    
    Each type uses a dedicated handler for communication and execution.
    """
    
    def __init__(
        self,
        custom_config: Union[Dict[str, Any], CustomAgentConfig],
        model_name: str = "custom-model",
        config: Optional[CABConfig] = None,
        prompt_manager: Optional[PromptManager] = None,
        **kwargs
    ):
        """Initialize custom maintainer agent.
        
        Args:
            custom_config: Custom agent configuration (dict or CustomAgentConfig dataclass)
            model_name: Model name identifier for tracking
            config: CAB configuration
            prompt_manager: Prompt manager instance
            **kwargs: Additional arguments
        """
        super().__init__(
            agent_type="maintainer",
            model_name=model_name,
            config=config or CABConfig(),
            prompt_manager=prompt_manager or PromptManager((config or CABConfig()).prompts_dir),
            **kwargs
        )
        
        # Convert CustomAgentConfig dataclass to dict if needed
        if isinstance(custom_config, CustomAgentConfig):
            custom_config.validate()
            custom_config = asdict(custom_config)
        
        # Validate custom config
        if not custom_config or "type" not in custom_config:
            raise ValueError("custom_config must include 'type' field")
        
        self.custom_config = custom_config
        self.agent_type_name = custom_config["type"].lower()
        
        # Initialize appropriate handler
        self.handler = self._create_handler(custom_config)
        
        # Storage for metrics from all calls
        self.metrics_data = []
        
        # Override model_config for custom agent
        from ..core.config import ModelConfig
        self.model_config = ModelConfig(
            name=model_name,
            model_id=model_name,
            max_tokens=8192,
            provider=f"custom_{self.agent_type_name}"
        )
        
        logger.info(f"🎨 Custom maintainer agent initialized (type: {self.agent_type_name})")
    
    def _create_handler(self, config: Dict[str, Any]):
        """Create appropriate handler based on agent type.
        
        Args:
            config: Custom agent configuration
            
        Returns:
            Handler instance (CLIAgentHandler, DockerAgentHandler, or PythonClassAgentHandler)
            
        Raises:
            ValueError: If agent type is not supported
        """
        agent_type = config["type"].lower()
        
        try:
            if agent_type == "cli":
                from .handlers.cli_handler import CLIAgentHandler
                return CLIAgentHandler(config)
                
            elif agent_type == "docker":
                from .handlers.docker_handler import DockerAgentHandler
                return DockerAgentHandler(config)
                
            elif agent_type == "python_class":
                from .handlers.python_handler import PythonClassAgentHandler
                return PythonClassAgentHandler(config)
                
            else:
                raise ValueError(
                    f"Unsupported custom agent type: {agent_type}. "
                    f"Supported types: cli, docker, python_class"
                )
        except ImportError as e:
            raise AgentError(
                f"Failed to import handler for {agent_type}: {e}",
                agent_type="custom"
            )
    
    def get_system_prompt(self, **kwargs) -> str:
        """Get maintainer system prompt for custom agent.
        
        Args:
            **kwargs: Additional context (repo_url, commit_hash, etc.)
            
        Returns:
            System prompt string
        """
        base_prompt = self.prompt_manager.get_prompt("maintainer/system_prompt")
        
        # Add repository-specific context if provided
        repo_context = ""
        if 'repo_url' in kwargs:
            repo_context += f"\nRepository: {kwargs['repo_url']}"
        if 'commit_hash' in kwargs:
            repo_context += f"\nCommit hash: {kwargs['commit_hash']}"
        if 'repo_type' in kwargs:
            repo_context += f"\nRepository type: {kwargs['repo_type']}"
        
        # Add custom agent guidance
        custom_guidance = f"""

CUSTOM AGENT CONTEXT:
- You are a custom {self.agent_type_name} agent integrated with CodeAssistBench
- You have access to the repository and can use your tools/capabilities
- Focus on providing accurate, executable solutions
- Provide clear explanations with code examples when appropriate
"""
        
        # Add Docker-specific guidance if this is a Docker issue
        docker_guidance = ""
        if kwargs.get('is_docker_issue'):
            docker_guidance = self.prompt_manager.get_prompt("maintainer/docker_prompt")
        
        return f"{base_prompt}{repo_context}{custom_guidance}{docker_guidance}"
    
    async def generate_response(
        self,
        user_prompt: str,
        system_prompt: str,
        issue_id: str = "unknown",
        **kwargs
    ) -> str:
        """Generate maintainer response using custom agent handler.
        
        Args:
            user_prompt: User input
            system_prompt: System prompt (maintainer-specific with repository context)
            issue_id: Issue ID for tracking
            **kwargs: Additional arguments including repo_dir
            
        Returns:
            Generated response text
        """
        self.increment_call_counter(issue_id)
        
        repo_dir = kwargs.get('repo_dir', '.')
        
        logger.info(f"Calling custom {self.agent_type_name} agent for issue {issue_id}")
        
        try:
            # Execute through handler
            result = await self.handler.execute(
                prompt=user_prompt,
                repo_dir=repo_dir,
                system_prompt=system_prompt
            )
            
            # Store metrics if provided
            if result.get("metrics"):
                self.metrics_data.append({
                    "issue_id": issue_id,
                    "timestamp": datetime.now().isoformat(),
                    "metrics": result["metrics"],
                    "execution_time": result.get("execution_time", 0),
                    "attempt": result.get("attempt", 1)
                })
            
            logger.info(f"✅ Custom agent response received ({len(result['response'])} chars)")
            
            return result["response"]
            
        except AgentError as e:
            logger.error(f"Custom agent execution failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in custom agent: {e}")
            raise AgentError(
                f"Custom agent execution error: {e}",
                agent_type=f"custom_{self.agent_type_name}"
            )
    
    async def generate_standard_response(
        self,
        repo_dir: str,
        issue_data: IssueData,
        conversation_history: List[ConversationMessage],
        **kwargs
    ) -> Tuple[str, str]:
        """Generate standard maintainer response during conversation.
        
        Args:
            repo_dir: Repository directory path
            issue_data: Issue data context
            conversation_history: Full conversation history
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (maintainer response, exploration results)
        """
        # Extract latest user message
        latest_user_message = ""
        for message in reversed(conversation_history):
            if message.role == "user":
                latest_user_message = message.content
                break
        
        # Build system prompt with context
        system_prompt = self.get_system_prompt(
            repo_url=issue_data.commit_info.repository if issue_data.commit_info else None,
            commit_hash=issue_data.commit_info.sha if issue_data.commit_info else None
        )
        
        # Build user prompt with conversation context
        user_prompt = f"""
Original question: {issue_data.first_question.title}

Conversation history:
{self._format_conversation_history(conversation_history)}

Latest user message: {latest_user_message}

Please respond to the user's message with a helpful, accurate answer.
"""
        
        # Execute through handler
        result = await self.handler.execute(
            prompt=user_prompt,
            repo_dir=repo_dir,
            system_prompt=system_prompt
        )
        
        return result["response"], result.get("exploration_results", "")
    
    async def generate_docker_response(
        self,
        repo_dir: str,
        issue_data: IssueData,
        conversation_history: List[ConversationMessage],
        **kwargs
    ) -> Tuple[str, Dict[str, str], Optional[str]]:
        """Generate Docker-aware response from custom agent.
        
        Args:
            repo_dir: Repository directory path
            issue_data: Issue data
            conversation_history: Conversation history
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (response, extra_files, modified_dockerfile)
        """
        # Extract latest user message
        latest_user_message = ""
        for message in reversed(conversation_history):
            if message.role == "user":
                latest_user_message = message.content
                break
        
        # Create Docker-specific system prompt
        docker_system_prompt = self.get_system_prompt(
            is_docker_issue=True,
            repo_url=issue_data.commit_info.repository if issue_data.commit_info else None,
            commit_hash=issue_data.commit_info.sha if issue_data.commit_info else None
        )
        
        # Create user prompt
        user_prompt = f"""
Original question: {issue_data.first_question.title}

Conversation history:
{self._format_conversation_history(conversation_history)}

Dockerfile:
{issue_data.dockerfile or 'No Dockerfile provided'}

Latest user message: {latest_user_message}

Please respond to the user's Docker-related issue with specific solutions.
"""
        
        # Execute through handler
        result = await self.handler.execute(
            prompt=user_prompt,
            repo_dir=repo_dir,
            system_prompt=docker_system_prompt
        )
        
        response = result["response"]
        
        # Note: Custom agents need to handle file creation/modification in their own way
        # For now, return empty extra_files and modified_dockerfile
        # Users can extend their custom agents to support these if needed
        extra_files = {}
        modified_dockerfile = None
        
        logger.info("Docker response generated by custom agent")
        
        return response, extra_files, modified_dockerfile
    
    async def choose_commit(self, reference_commit: str, user_question: str) -> str:
        """Allow custom maintainer to decide commit to use for exploration.
        
        For custom agents, we use the reference commit by default.
        Custom agents can implement their own commit selection logic internally.
        
        Args:
            reference_commit: Reference commit hash
            user_question: User's question text
            
        Returns:
            Selected commit hash (reference commit for custom agents)
        """
        logger.info(f"Using reference commit for custom agent: {reference_commit}")
        return reference_commit
    
    def get_custom_metrics(self) -> Dict[str, Any]:
        """Get aggregated custom agent metrics.
        
        Returns:
            Dictionary with call count and metrics data
        """
        return {
            "agent_type": self.agent_type_name,
            "call_count": len(self.metrics_data),
            "metrics_data": self.metrics_data,
            "total_execution_time": sum(
                m.get("execution_time", 0) for m in self.metrics_data
            )
        }
    
    def _format_conversation_history(self, history: List[ConversationMessage]) -> str:
        """Format conversation history for better readability.
        
        Args:
            history: List of conversation messages
            
        Returns:
            Formatted conversation string
        """
        formatted = ""
        for message in history:
            role = "User" if message.role == "user" else "Maintainer"
            formatted += f"{role}: {message.content}\n\n"
        return formatted
