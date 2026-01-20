"""CAB evaluation agents."""

from .base_agent import BaseAgent
from .maintainer_agent import MaintainerAgent
from .user_agent import UserAgent
from .judge_agent import JudgeAgent
from .agent_factory import AgentFactory
from .kiro_cli_maintainer_agent import KiroCLIMaintainerAgent

# Backward compatibility alias
QCLIMaintainerAgent = KiroCLIMaintainerAgent

__all__ = [
    'BaseAgent',
    'MaintainerAgent',
    'UserAgent', 
    'JudgeAgent',
    'AgentFactory',
    'KiroCLIMaintainerAgent',
    'QCLIMaintainerAgent',  # Backward compatibility
]
