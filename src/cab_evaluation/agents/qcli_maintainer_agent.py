"""Q-CLI-based maintainer agent implementation."""

import os
import time
import logging
import subprocess
from typing import Dict, List, Optional, Any, Tuple

from ..core.models import IssueData, ConversationMessage
from ..core.exceptions import AgentError
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class QCLIMaintainerAgent(BaseAgent):
    """Maintainer agent using Amazon Q CLI (Kiro CLI)."""
    
    def __init__(
        self,
        model_name: str = "claude-sonnet-4.5",
        qcli_path: str = "q",
        timeout: int = 300,
        config=None,
        prompt_manager=None,
        **kwargs
    ):
        """Initialize Q-CLI maintainer agent.
        
        Args:
            model_name: Q-CLI model name (e.g., "claude-sonnet-4.5", "claude-haiku-4.5")
            qcli_path: Path to Q-CLI executable
            timeout: Timeout in seconds for Q-CLI commands
        """
        from ..core.config import CABConfig
        from ..prompts.prompt_manager import PromptManager
        
        super().__init__(
            agent_type="maintainer",
            model_name=model_name,
            config=config or CABConfig(),
            prompt_manager=prompt_manager or PromptManager((config or CABConfig()).prompts_dir),
            **kwargs
        )
        
        self.qcli_path = qcli_path
        self.timeout = timeout
        self._qcli_version = None
        self.qcli_metadata = []  # Store metadata for each Q-CLI call
        
        # Override model_config for Q-CLI
        from ..core.config import ModelConfig
        self.model_config = ModelConfig(
            name=model_name,
            model_id=model_name,
            max_tokens=8192,
            provider="qcli"
        )
        
        self._validate_qcli()
        
        logger.info(f"🤖 Q-CLI maintainer agent initialized with model: {model_name}")
        logger.info(f"📁 Q-CLI path: {self.qcli_path}")
    
    def _validate_qcli(self):
        """Validate Q-CLI availability."""
        try:
            result = subprocess.run(
                [self.qcli_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                self._qcli_version = result.stdout.strip()
                logger.info(f"✅ Q-CLI validated: {self._qcli_version}")
            else:
                raise AgentError(
                    f"Q-CLI not working: {result.stderr}",
                    agent_type="qcli_maintainer"
                )
        except FileNotFoundError:
            raise AgentError(
                f"Q-CLI not found at: {self.qcli_path}. Install from AWS documentation.",
                agent_type="qcli_maintainer"
            )
        except subprocess.TimeoutExpired:
            raise AgentError(
                "Q-CLI validation timeout",
                agent_type="qcli_maintainer"
            )
        except Exception as e:
            raise AgentError(
                f"Q-CLI validation failed: {e}",
                agent_type="qcli_maintainer"
            )
    
    def get_system_prompt(self, **kwargs) -> str:
        """Get maintainer system prompt for Q-CLI."""
        base_prompt = self.prompt_manager.get_prompt("maintainer/system_prompt")
        
        # Add repository-specific context if provided
        repo_context = ""
        if 'repo_url' in kwargs:
            repo_context += f"\nRepository: {kwargs['repo_url']}"
        if 'commit_hash' in kwargs:
            repo_context += f"\nCommit hash: {kwargs['commit_hash']}"
        
        # Add Q-CLI-specific guidance
        qcli_guidance = """

Q-CLI CONTEXT:
- You are operating within the Amazon Q CLI framework
- You have access to the repository context and file system
- Focus on providing accurate, executable solutions
- Provide clear explanations with code examples when applicable
"""
        
        return f"{base_prompt}{repo_context}{qcli_guidance}"
    
    async def call_llm(
        self,
        user_prompt: str,
        system_prompt: str,
        issue_id: str = "unknown",
        **kwargs
    ) -> str:
        """Call LLM using Q-CLI."""
        return await self.generate_response(user_prompt, system_prompt, issue_id, **kwargs)
    
    async def generate_response(
        self,
        user_prompt: str,
        system_prompt: str,
        issue_id: str = "unknown",
        **kwargs
    ) -> str:
        """Generate maintainer response using Q-CLI."""
        issue_logger = kwargs.get('issue_logger')
        log = issue_logger if issue_logger else logger
        
        self.increment_call_counter(issue_id)
        
        log.info(f"===== Q-CLI SYSTEM PROMPT =====\n{system_prompt}\n")
        log.info(f"===== Q-CLI USER PROMPT =====\n{user_prompt}\n")
        self._flush_logger(log)
        
        start_time = time.time()
        log.info(f"Calling Q-CLI with model {self.model_name} (issue: {issue_id})")
        self._flush_logger(log)
        
        # Get repository directory from kwargs
        repo_dir = kwargs.get('repo_dir', '.')
        
        try:
            # Combine system and user prompts
            combined_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"
            
            # Build Q-CLI command - use stdin instead of temp file
            cmd = [
                self.qcli_path,
                "chat",
                "--model", self.model_name,
                "--no-interactive",
                "--trust-all-tools"
            ]
            
            log.info(f"Executing Q-CLI command: {' '.join(cmd)}")
            self._flush_logger(log)
            
            # Execute Q-CLI with prompt via stdin
            result = subprocess.run(
                cmd,
                input=combined_prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=repo_dir
            )
            
            elapsed_time = time.time() - start_time
            
            if result.returncode == 0:
                response = result.stdout.strip()
                
                # Store metadata
                metadata = {
                    "execution_time_seconds": elapsed_time,
                    "response_length": len(response),
                    "model": self.model_name,
                    "issue_id": issue_id
                }
                self.qcli_metadata.append(metadata)
                
                log.info(f"Q-CLI responded in {elapsed_time:.2f} seconds")
                log.info(f"Q-CLI metadata: {metadata}")
                log.info(f"===== Q-CLI RESPONSE =====\n{response}\n")
                self._flush_logger(log)
                return response
            else:
                error_msg = f"Q-CLI failed with return code {result.returncode}: {result.stderr}"
                log.error(error_msg)
                self._flush_logger(log)
                raise AgentError(error_msg, agent_type="qcli_maintainer")
            
        except subprocess.TimeoutExpired:
            elapsed_time = time.time() - start_time
            error_msg = f"Q-CLI timeout after {elapsed_time:.2f}s"
            log.error(error_msg)
            self._flush_logger(log)
            raise AgentError(error_msg, agent_type="qcli_maintainer")
        except Exception as e:
            elapsed_time = time.time() - start_time
            log.error(f"Q-CLI call failed after {elapsed_time:.2f}s: {e}")
            self._flush_logger(log)
            raise AgentError(f"Q-CLI execution failed: {e}", agent_type="qcli_maintainer")
    
    def _flush_logger(self, log):
        """Flush logger handlers."""
        for handler in log.handlers:
            handler.flush()
    
    def get_qcli_metadata(self) -> Dict[str, Any]:
        """Get aggregated Q-CLI metadata.
        
        Returns:
            Dictionary with aggregated metadata including total execution time,
            total response length, and call count.
        """
        if not self.qcli_metadata:
            return {
                "total_execution_time_seconds": 0.0,
                "total_response_length": 0,
                "call_count": 0,
                "model": self.model_name
            }
        
        return {
            "total_execution_time_seconds": sum(m["execution_time_seconds"] for m in self.qcli_metadata),
            "total_response_length": sum(m["response_length"] for m in self.qcli_metadata),
            "call_count": len(self.qcli_metadata),
            "model": self.model_name,
            "calls": self.qcli_metadata
        }
    
    async def choose_commit(self, reference_commit: str, user_question: str) -> str:
        """Allow maintainer to decide commit to use for exploration.
        
        Args:
            reference_commit: Reference commit hash
            user_question: User's question text
            
        Returns:
            Selected commit hash
        """
        # For Q-CLI, simply return the reference commit
        # Q-CLI operates in the current repository state
        self.logger.info(f"Using reference commit: {reference_commit}")
        return reference_commit
    
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
            
        Returns:
            Tuple of (maintainer response, exploration results)
        """
        # Get latest user message from conversation history
        user_message = conversation_history[-1].content if conversation_history else ""
        
        # Build context
        context = f"Issue: {issue_data.first_question.title}\n\n"
        
        if len(conversation_history) > 1:
            context += "Conversation so far:\n"
            for msg in conversation_history[-4:-1]:  # Last 3 messages before current
                context += f"{msg.role}: {msg.content[:200]}...\n"
            context += "\n"
        
        user_prompt = f"{context}User's latest message: {user_message}\n\nPlease provide a helpful response."
        system_prompt = self.get_system_prompt(
            repo_url=issue_data.commit_info.repository if issue_data.commit_info else None,
            commit_hash=issue_data.commit_info.sha if issue_data.commit_info else None
        )
        
        response = await self.generate_response(
            user_prompt,
            system_prompt,
            issue_data.id,
            repo_dir=repo_dir,
            **kwargs
        )
        
        # Q-CLI doesn't separate exploration results, return empty string
        return response, ""
