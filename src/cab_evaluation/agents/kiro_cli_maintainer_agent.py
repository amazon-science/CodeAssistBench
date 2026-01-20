"""Kiro CLI-based maintainer agent implementation."""

import os
import time
import logging
import subprocess
from typing import Dict, List, Optional, Any, Tuple

from ..core.models import IssueData, ConversationMessage
from ..core.exceptions import AgentError
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


# Model name mapping from CAB config names to Kiro CLI model names
KIRO_CLI_MODEL_MAPPING = {
    # Sonnet models
    "sonnet45": "claude-sonnet-4.5",
    "sonnet-4.5": "claude-sonnet-4.5",
    "sonnet4": "claude-sonnet-4",
    "sonnet-4": "claude-sonnet-4",
    "sonnet": "claude-sonnet-4",
    "sonnet37": "claude-sonnet-4",
    # Haiku models
    "haiku": "claude-haiku-4.5",
    "haiku45": "claude-haiku-4.5",
    "haiku-4.5": "claude-haiku-4.5",
    # Opus models
    "opus": "claude-opus-4.5",
    "opus45": "claude-opus-4.5",
    "opus-4.5": "claude-opus-4.5",
    # Qwen models
    "qwen": "qwen3-coder-480b",
    "qwen3": "qwen3-coder-480b",
    "qwen3-coder": "qwen3-coder-480b",
    # Full Kiro CLI model names (passthrough)
    "claude-sonnet-4.5": "claude-sonnet-4.5",
    "claude-sonnet-4": "claude-sonnet-4",
    "claude-haiku-4.5": "claude-haiku-4.5",
    "claude-opus-4.5": "claude-opus-4.5",
    "qwen3-coder-480b": "qwen3-coder-480b",
}


class KiroCLIMaintainerAgent(BaseAgent):
    """Maintainer agent using Kiro CLI."""
    
    # Approximate characters per token for token estimation
    # Claude models average ~4 chars per token
    CHARS_PER_TOKEN = 4
    
    def __init__(
        self,
        model_name: str = "sonnet45",
        kiro_cli_path: str = "kiro-cli",
        timeout: int = 300,
        config=None,
        prompt_manager=None,
        **kwargs
    ):
        """Initialize Kiro CLI maintainer agent.
        
        Args:
            model_name: Model name (e.g., "sonnet45", "haiku")
            kiro_cli_path: Path to Kiro CLI executable
            timeout: Timeout in seconds for Kiro CLI commands
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
        
        self.kiro_cli_path = kiro_cli_path
        self.timeout = timeout
        self._kiro_cli_version = None
        self.kiro_cli_metadata = []  # Store metadata for each Kiro CLI call
        
        # Token tracking (estimated since Kiro CLI doesn't expose actual token counts)
        self.total_input_chars = 0
        self.total_output_chars = 0
        
        # Map CAB model name to Kiro CLI model name
        self.kiro_cli_model_name = KIRO_CLI_MODEL_MAPPING.get(model_name, model_name)
        logger.info(f"Model mapping: {model_name} -> {self.kiro_cli_model_name}")
        
        # Override model_config for Kiro CLI
        from ..core.config import ModelConfig
        self.model_config = ModelConfig(
            name=model_name,
            model_id=self.kiro_cli_model_name,
            max_tokens=8192,
            provider="kiro_cli"
        )
        
        self._validate_kiro_cli()
        
        logger.info(f"🤖 Kiro CLI maintainer agent initialized with model: {model_name}")
        logger.info(f"📁 Kiro CLI path: {self.kiro_cli_path}")
    
    def _validate_kiro_cli(self):
        """Validate Kiro CLI availability."""
        try:
            result = subprocess.run(
                [self.kiro_cli_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                self._kiro_cli_version = result.stdout.strip()
                logger.info(f"✅ Kiro CLI validated: {self._kiro_cli_version}")
            else:
                raise AgentError(
                    f"Kiro CLI not working: {result.stderr}",
                    agent_type="kiro_cli_maintainer"
                )
        except FileNotFoundError:
            raise AgentError(
                f"Kiro CLI not found at: {self.kiro_cli_path}. Install with: npm install -g @anthropics/kiro-cli",
                agent_type="kiro_cli_maintainer"
            )
        except subprocess.TimeoutExpired:
            raise AgentError(
                "Kiro CLI validation timeout",
                agent_type="kiro_cli_maintainer"
            )
        except Exception as e:
            raise AgentError(
                f"Kiro CLI validation failed: {e}",
                agent_type="kiro_cli_maintainer"
            )
    
    def get_system_prompt(self, **kwargs) -> str:
        """Get maintainer system prompt for Kiro CLI."""
        base_prompt = self.prompt_manager.get_prompt("maintainer/system_prompt")
        
        # Add repository-specific context if provided
        repo_context = ""
        if 'repo_url' in kwargs:
            repo_context += f"\nRepository: {kwargs['repo_url']}"
        if 'commit_hash' in kwargs:
            repo_context += f"\nCommit hash: {kwargs['commit_hash']}"
        
        # Add Kiro CLI-specific guidance
        kiro_cli_guidance = """

KIRO CLI CONTEXT:
- You are operating within the Kiro CLI framework
- You have access to the repository context and file system
- Focus on providing accurate, executable solutions
- Provide clear explanations with code examples when applicable
"""
        
        return f"{base_prompt}{repo_context}{kiro_cli_guidance}"
    
    async def call_llm(
        self,
        user_prompt: str,
        system_prompt: str,
        issue_id: str = "unknown",
        **kwargs
    ) -> str:
        """Call LLM using Kiro CLI."""
        return await self.generate_response(user_prompt, system_prompt, issue_id, **kwargs)
    
    async def generate_response(
        self,
        user_prompt: str,
        system_prompt: str,
        issue_id: str = "unknown",
        **kwargs
    ) -> str:
        """Generate maintainer response using Kiro CLI."""
        issue_logger = kwargs.get('issue_logger')
        log = issue_logger if issue_logger else logger
        
        self.increment_call_counter(issue_id)
        
        log.info(f"===== KIRO CLI SYSTEM PROMPT =====\n{system_prompt}\n")
        log.info(f"===== KIRO CLI USER PROMPT =====\n{user_prompt}\n")
        self._flush_logger(log)
        
        start_time = time.time()
        log.info(f"Calling Kiro CLI with model {self.model_name} (issue: {issue_id})")
        self._flush_logger(log)
        
        # Get repository directory from kwargs
        repo_dir = kwargs.get('repo_dir', '.')
        
        try:
            # Combine system and user prompts
            combined_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"
            
            # Build Kiro CLI command - use stdin instead of temp file
            cmd = [
                self.kiro_cli_path,
                "chat",
                "--model", self.kiro_cli_model_name,
                "--no-interactive",
                "--trust-all-tools"
            ]
            
            log.info(f"Executing Kiro CLI command: {' '.join(cmd)}")
            self._flush_logger(log)
            
            # Execute Kiro CLI with prompt via stdin
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
                
                # Track character counts for token estimation
                input_chars = len(combined_prompt)
                output_chars = len(response)
                self.total_input_chars += input_chars
                self.total_output_chars += output_chars
                
                # Estimate tokens (Kiro CLI doesn't expose actual token counts)
                estimated_input_tokens = input_chars // self.CHARS_PER_TOKEN
                estimated_output_tokens = output_chars // self.CHARS_PER_TOKEN
                
                # Store metadata
                metadata = {
                    "execution_time_seconds": elapsed_time,
                    "input_chars": input_chars,
                    "output_chars": output_chars,
                    "response_length": len(response),
                    "estimated_input_tokens": estimated_input_tokens,
                    "estimated_output_tokens": estimated_output_tokens,
                    "model": self.model_name,
                    "issue_id": issue_id
                }
                self.kiro_cli_metadata.append(metadata)
                
                log.info(f"Kiro CLI responded in {elapsed_time:.2f} seconds")
                log.info(f"Kiro CLI metadata: {metadata}")
                log.info(f"Token usage (estimated): input={estimated_input_tokens}, output={estimated_output_tokens}")
                log.info(f"===== KIRO CLI RESPONSE =====\n{response}\n")
                self._flush_logger(log)
                return response
            else:
                error_msg = f"Kiro CLI failed with return code {result.returncode}: {result.stderr}"
                log.error(error_msg)
                self._flush_logger(log)
                raise AgentError(error_msg, agent_type="kiro_cli_maintainer")
            
        except subprocess.TimeoutExpired:
            elapsed_time = time.time() - start_time
            error_msg = f"Kiro CLI timeout after {elapsed_time:.2f}s"
            log.error(error_msg)
            self._flush_logger(log)
            raise AgentError(error_msg, agent_type="kiro_cli_maintainer")
        except Exception as e:
            elapsed_time = time.time() - start_time
            log.error(f"Kiro CLI call failed after {elapsed_time:.2f}s: {e}")
            self._flush_logger(log)
            raise AgentError(f"Kiro CLI execution failed: {e}", agent_type="kiro_cli_maintainer")
    
    def _flush_logger(self, log):
        """Flush logger handlers."""
        for handler in log.handlers:
            handler.flush()
    
    def get_kiro_cli_metadata(self) -> Dict[str, Any]:
        """Get aggregated Kiro CLI metadata.
        
        Returns:
            Dictionary with aggregated metadata including total execution time,
            total response length, call count, and token estimates.
        """
        if not self.kiro_cli_metadata:
            return {
                "total_execution_time_seconds": 0.0,
                "total_response_length": 0,
                "call_count": 0,
                "model": self.model_name,
                "estimated_total_input_tokens": 0,
                "estimated_total_output_tokens": 0,
                "estimated_total_tokens": 0,
                "total_input_chars": 0,
                "total_output_chars": 0,
                "token_estimation_note": "Kiro CLI does not expose actual token counts. Values are estimated (~4 chars/token)."
            }
        
        estimated_input = sum(m.get("estimated_input_tokens", 0) for m in self.kiro_cli_metadata)
        estimated_output = sum(m.get("estimated_output_tokens", 0) for m in self.kiro_cli_metadata)
        
        return {
            "total_execution_time_seconds": sum(m["execution_time_seconds"] for m in self.kiro_cli_metadata),
            "total_response_length": sum(m["response_length"] for m in self.kiro_cli_metadata),
            "call_count": len(self.kiro_cli_metadata),
            "model": self.model_name,
            "estimated_total_input_tokens": estimated_input,
            "estimated_total_output_tokens": estimated_output,
            "estimated_total_tokens": estimated_input + estimated_output,
            "total_input_chars": self.total_input_chars,
            "total_output_chars": self.total_output_chars,
            "token_estimation_note": "Kiro CLI does not expose actual token counts. Values are estimated (~4 chars/token).",
            "calls": self.kiro_cli_metadata
        }
    
    def get_token_usage(self) -> Dict[str, Any]:
        """Get token usage summary for compatibility with other agents.
        
        Note: Kiro CLI doesn't expose actual token counts, so these are estimates.
        
        Returns:
            Dictionary with token usage estimates
        """
        metadata = self.get_kiro_cli_metadata()
        return {
            "input_tokens": metadata["estimated_total_input_tokens"],
            "output_tokens": metadata["estimated_total_output_tokens"],
            "total_tokens": metadata["estimated_total_tokens"],
            "reasoning_tokens": 0,  # Not available from Kiro CLI
            "cache_hit_percent": 0.0,  # Not available from Kiro CLI
            "total_cost_usd": 0.0,  # Cannot calculate without actual token counts
            "is_estimated": True,
            "note": metadata["token_estimation_note"]
        }
    
    async def choose_commit(self, reference_commit: str, user_question: str) -> str:
        """Allow maintainer to decide commit to use for exploration.
        
        Args:
            reference_commit: Reference commit hash
            user_question: User's question text
            
        Returns:
            Selected commit hash
        """
        # For Kiro CLI, simply return the reference commit
        # Kiro CLI operates in the current repository state
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
        
        # Kiro CLI doesn't separate exploration results, return empty string
        return response, ""


# Backward compatibility alias
QCLIMaintainerAgent = KiroCLIMaintainerAgent
