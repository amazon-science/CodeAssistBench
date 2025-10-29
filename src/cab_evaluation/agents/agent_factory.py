"""Agent factory for creating CAB evaluation agents."""

from typing import Optional, Dict, Any

from ..core.config import CABConfig
from ..core.models import JudgeConfig
from ..prompts.prompt_manager import PromptManager
from .base_agent import BaseAgent
from .maintainer_agent import MaintainerAgent
from .user_agent import UserAgent
from .judge_agent import JudgeAgent

import logging

logger = logging.getLogger(__name__)


class AgentFactory:
    """Factory class for creating CAB evaluation agents."""
    
    def __init__(self, config: Optional[CABConfig] = None, prompt_manager: Optional[PromptManager] = None):
        """Initialize agent factory.
        
        Args:
            config: CAB configuration
            prompt_manager: Prompt manager instance
        """
        self.config = config or CABConfig()
        self.prompt_manager = prompt_manager or PromptManager(self.config.prompts_dir)
        
        # Validate configuration
        self.config.validate()
        self.config.setup_directories()
    
    def create_maintainer_agent(
        self,
        model_name: Optional[str] = None,
        **kwargs
    ) -> MaintainerAgent:
        """Create a maintainer agent.
        
        Args:
            model_name: Model to use (defaults to config default)
            **kwargs: Additional arguments
            
        Returns:
            MaintainerAgent instance
        """
        model_name = model_name or self.config.default_maintainer_model
        
        return MaintainerAgent(
            model_name=model_name,
            config=self.config,
            prompt_manager=self.prompt_manager,
            **kwargs
        )
    
    def create_user_agent(
        self,
        model_name: Optional[str] = None,
        **kwargs
    ) -> UserAgent:
        """Create a user agent.
        
        Args:
            model_name: Model to use (defaults to config default)
            **kwargs: Additional arguments
            
        Returns:
            UserAgent instance
        """
        model_name = model_name or self.config.default_user_model
        
        return UserAgent(
            model_name=model_name,
            config=self.config,
            prompt_manager=self.prompt_manager,
            **kwargs
        )
    
    def create_judge_agent(
        self,
        model_name: Optional[str] = None,
        judge_config: Optional[JudgeConfig] = None,
        **kwargs
    ) -> JudgeAgent:
        """Create a judge agent.
        
        Args:
            model_name: Model to use (defaults to config default)
            judge_config: Configuration for judge behavior
            **kwargs: Additional arguments
            
        Returns:
            JudgeAgent instance
        """
        model_name = model_name or self.config.default_judge_model
        
        return JudgeAgent(
            model_name=model_name,
            config=self.config,
            prompt_manager=self.prompt_manager,
            judge_config=judge_config,
            **kwargs
        )
    
    def create_agent(
        self,
        agent_type: str,
        model_name: Optional[str] = None,
        **kwargs
    ) -> BaseAgent:
        """Create an agent by type.
        
        Args:
            agent_type: Type of agent to create (maintainer, user, judge)
            model_name: Model to use
            **kwargs: Additional arguments
            
        Returns:
            Agent instance
            
        Raises:
            ValueError: If agent_type is not supported
        """
        if agent_type == "maintainer":
            return self.create_maintainer_agent(model_name, **kwargs)
        elif agent_type == "user":
            return self.create_user_agent(model_name, **kwargs)
        elif agent_type == "judge":
            return self.create_judge_agent(model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")
    
    def create_agent_set(
        self,
        maintainer_model: Optional[str] = None,
        user_model: Optional[str] = None,
        judge_model: Optional[str] = None
    ) -> Dict[str, BaseAgent]:
        """Create a complete set of agents for CAB evaluation.
        
        Args:
            maintainer_model: Model for maintainer agent
            user_model: Model for user agent
            judge_model: Model for judge agent
            
        Returns:
            Dictionary mapping agent types to agent instances
        """
        return {
            "maintainer": self.create_maintainer_agent(maintainer_model),
            "user": self.create_user_agent(user_model),
            "judge": self.create_judge_agent(judge_model)
        }
    
    def update_model_mapping(
        self,
        agent_model_mapping: Dict[str, str]
    ) -> Dict[str, BaseAgent]:
        """Create agents with custom model mapping.
        
        Args:
            agent_model_mapping: Dictionary mapping agent types to model names
                                Example: {"maintainer": "sonnet37", "user": "haiku", "judge": "sonnet"}
            
        Returns:
            Dictionary mapping agent types to agent instances
        """
        agents = {}
        
        for agent_type in ["maintainer", "user", "judge"]:
            model_name = agent_model_mapping.get(agent_type)
            agents[agent_type] = self.create_agent(agent_type, model_name)
        
        logger.info(f"Created agent set with model mapping: {agent_model_mapping}")
        return agents
