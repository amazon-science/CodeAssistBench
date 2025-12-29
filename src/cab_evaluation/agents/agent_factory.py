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
        """Initialize agent factory."""
        self.config = config or CABConfig()
        self.prompt_manager = prompt_manager or PromptManager(self.config.prompts_dir)
        
        self.config.validate()
        self.config.setup_directories()
    
    def create_maintainer_agent(
        self,
        model_name: Optional[str] = None,
        framework: str = "strands",
        openhands_config: Optional[str] = None,
        **kwargs
    ) -> BaseAgent:
        """Create a maintainer agent with framework selection."""
        model_name = model_name or self.config.default_maintainer_model
        
        if framework.lower() == "qcli":
            logger.info("🤖 Creating Q-CLI-based maintainer agent")
            try:
                from .qcli_maintainer_agent import QCLIMaintainerAgent
                
                return QCLIMaintainerAgent(
                    model_name=model_name,
                    config=self.config,
                    prompt_manager=self.prompt_manager,
                    qcli_path=self.config.agent_framework_config.qcli_path,
                    timeout=self.config.agent_framework_config.qcli_timeout,
                    **kwargs
                )
            except ImportError as e:
                logger.error(f"❌ Q-CLI agent import failed: {e}")
                logger.info("🔄 Falling back to Strands framework")
                framework = "strands"
        
        if framework.lower() == "openhands":
            logger.info("🤖 Creating OpenHands-based maintainer agent")
            try:
                from .openhands_maintainer_agent import OpenHandsMaintainerAgent
                from ..utils.openhands_utils import map_cab_model_to_openhands
                openhands_model = map_cab_model_to_openhands(model_name)
                
                return OpenHandsMaintainerAgent(
                    model_name=openhands_model,
                    config=self.config,
                    prompt_manager=self.prompt_manager,
                    config_file=openhands_config,
                    enable_ast_tools=kwargs.get('enable_ast_tools', True),
                    custom_system_prompt_path=kwargs.get('custom_system_prompt_path'),
                    **{k: v for k, v in kwargs.items() if k not in ['enable_ast_tools', 'custom_system_prompt_path']}
                )
            except ImportError as e:
                logger.error(f"❌ OpenHands not available: {e}")
                logger.info("📦 Install with: pip install openhands")
                logger.info("🔄 Falling back to Strands framework")
                framework = "strands"
        
        logger.info("🤖 Creating Strands-based maintainer agent")
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
        """Create a user agent."""
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
        """Create a judge agent."""
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
        """Create an agent by type."""
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
        """Create a complete set of agents for CAB evaluation."""
        return {
            "maintainer": self.create_maintainer_agent(maintainer_model),
            "user": self.create_user_agent(user_model),
            "judge": self.create_judge_agent(judge_model)
        }
    
    def update_model_mapping(
        self,
        agent_model_mapping: Dict[str, str]
    ) -> Dict[str, BaseAgent]:
        """Create agents with custom model mapping."""
        agents = {}
        
        for agent_type in ["maintainer", "user", "judge"]:
            model_name = agent_model_mapping.get(agent_type)
            agents[agent_type] = self.create_agent(agent_type, model_name)
        
        logger.info(f"Created agent set with model mapping: {agent_model_mapping}")
        return agents
