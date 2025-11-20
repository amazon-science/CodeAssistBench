"""OpenHands-based maintainer agent implementation using SDK."""

import os
import time
import shutil
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from ..core.models import IssueData, ConversationMessage
from ..core.exceptions import AgentError
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class OpenHandsMaintainerAgent(BaseAgent):
    """Maintainer agent using OpenHands SDK for code generation."""
    
    def __init__(
        self,
        model_name: str = "anthropic/claude-sonnet-4-5-20250929",
        config_file: Optional[str] = None,
        config=None,
        prompt_manager=None,
        **kwargs
    ):
        """Initialize OpenHands maintainer agent."""
        # Store OpenHands-specific attributes
        self.workspace_base = tempfile.mkdtemp(prefix="openhands_cab_")
        self._is_openhands = True
        self._oh_agent = None  # Will be initialized lazily
        self._oh_llm = None
        
        # Initialize parent
        from ..core.config import CABConfig
        from ..prompts.prompt_manager import PromptManager
        
        super().__init__(
            agent_type="maintainer",
            model_name=model_name,
            config=config or CABConfig(),
            prompt_manager=prompt_manager or PromptManager((config or CABConfig()).prompts_dir),
            **kwargs
        )
        
        # Override model_config for OpenHands
        from ..core.config import ModelConfig
        self.model_config = ModelConfig(
            name=model_name,
            model_id=model_name,
            max_tokens=8192,
            provider="openhands"
        )
        
        self._validate_openhands_sdk()
        
        logger.info(f"🤖 OpenHands SDK maintainer agent initialized with model: {model_name}")
        logger.info(f"📁 Workspace: {self.workspace_base}")
    
    def _validate_openhands_sdk(self):
        """Validate OpenHands SDK availability."""
        try:
            from openhands.sdk import Agent, Conversation, LLM, Tool
            from openhands.tools.terminal import TerminalTool
            from openhands.tools.file_editor import FileEditorTool
            logger.info("✅ OpenHands SDK validated successfully")
        except ImportError as e:
            error_msg = (
                f"OpenHands SDK not found: {e}\n"
                f"Install with: pip install openhands-sdk openhands-tools\n"
                f"Or: pip install -e .[openhands]"
            )
            logger.error(f"❌ {error_msg}")
            raise AgentError(error_msg, agent_type="openhands_maintainer")
    
    def _initialize_openhands_agent(self):
        """Initialize OpenHands agent and LLM."""
        if self._oh_agent is not None:
            return
        
        try:
            from openhands.sdk import Agent, LLM, Tool
            from openhands.tools.terminal import TerminalTool
            from openhands.tools.file_editor import FileEditorTool
            from ..utils.openhands_utils import validate_openhands_config
            
            # Validate configuration and get API key
            api_key = validate_openhands_config(self.model_name)
            
            # Ensure model name has proper format for litellm
            model_name = self._normalize_model_name(self.model_name)
            logger.info(f"🔧 Normalized model name: {self.model_name} -> {model_name}")
            
            # Create LLM
            self._oh_llm = LLM(
                model=model_name,
                api_key=api_key,
            )
            
            if not self._oh_llm:
                raise AgentError("LLM initialization failed", agent_type="openhands_maintainer")
            
            # Create Agent with tools
            self._oh_agent = Agent(
                llm=self._oh_llm,
                tools=[
                    Tool(name=TerminalTool.name),
                    Tool(name=FileEditorTool.name),
                ],
            )
            
            if not self._oh_agent:
                raise AgentError("Agent initialization failed", agent_type="openhands_maintainer")
            
            logger.info(f"✅ OpenHands Agent initialized with {self.model_name}")
            
        except ValueError as e:
            raise AgentError(str(e), agent_type="openhands_maintainer")
        except ImportError as e:
            raise AgentError(f"Failed to import OpenHands SDK: {e}", agent_type="openhands_maintainer")
        except Exception as e:
            raise AgentError(f"Failed to initialize OpenHands agent: {e}", agent_type="openhands_maintainer")
    
    def get_system_prompt(self, **kwargs) -> str:
        """Get maintainer system prompt for OpenHands."""
        base_prompt = self.prompt_manager.get_prompt("maintainer/system_prompt")
        
        # Add repository-specific context if provided
        repo_context = ""
        if 'repo_url' in kwargs:
            repo_context += f"\nRepository: {kwargs['repo_url']}"
        if 'commit_hash' in kwargs:
            repo_context += f"\nCommit hash: {kwargs['commit_hash']}"
        
        # Add OpenHands-specific guidance
        openhands_guidance = """

OPENHANDS SDK CONTEXT:
- You are operating within the OpenHands agent SDK framework
- You have access to file system operations and command execution via tools
- Focus on providing accurate, executable solutions
- Use tools effectively to explore and understand the codebase
"""
        
        return f"{base_prompt}{repo_context}{openhands_guidance}"
    
    async def call_llm(
        self,
        user_prompt: str,
        system_prompt: str,
        issue_id: str = "unknown",
        **kwargs
    ) -> str:
        """Call LLM using OpenHands SDK."""
        return await self.generate_response(user_prompt, system_prompt, issue_id, **kwargs)
    
    async def generate_response(
        self,
        user_prompt: str,
        system_prompt: str,
        issue_id: str = "unknown",
        **kwargs
    ) -> str:
        """Generate maintainer response using OpenHands SDK."""
        issue_logger = kwargs.get('issue_logger')
        log = issue_logger if issue_logger else logger
        
        self.increment_call_counter(issue_id)
        
        log.info(f"===== OPENHANDS SYSTEM PROMPT =====\n{system_prompt}\n")
        log.info(f"===== OPENHANDS USER PROMPT =====\n{user_prompt}\n")
        self._flush_logger(log)
        
        start_time = time.time()
        log.info(f"Calling OpenHands SDK with model {self.model_name} (issue: {issue_id})")
        self._flush_logger(log)
        
        # Get repository directory from kwargs
        repo_dir = kwargs.get('repo_dir')
        if not repo_dir:
            logger.warning("No repo_dir provided, using workspace base")
            repo_dir = self.workspace_base
        
        try:
            # Initialize OpenHands agent
            self._initialize_openhands_agent()
            
            # Prepare workspace
            workspace_dir = self._prepare_workspace(repo_dir, issue_id)
            
            # Execute using OpenHands SDK
            response = await self._execute_openhands_sdk(
                workspace_dir,
                system_prompt,
                user_prompt,
                issue_id,
                issue_logger=issue_logger
            )
            
            elapsed_time = time.time() - start_time
            log.info(f"OpenHands responded in {elapsed_time:.2f} seconds")
            log.info(f"===== OPENHANDS RESPONSE =====\n{response}\n")
            self._flush_logger(log)
            
            return response
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            log.error(f"OpenHands call failed after {elapsed_time:.2f}s: {e}")
            self._flush_logger(log)
            raise AgentError(f"OpenHands execution failed: {e}", agent_type="openhands_maintainer")
    
    def _prepare_workspace(self, repo_dir: str, issue_id: str) -> str:
        """Prepare OpenHands workspace with repository content."""
        workspace_dir = Path(self.workspace_base) / f"workspace_{issue_id}"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        
        if Path(repo_dir).exists():
            logger.info(f"Copying repository to OpenHands workspace...")
            for item in Path(repo_dir).iterdir():
                if item.name == ".git":
                    logger.info(f"⏭️  Skipping .git directory")
                    continue
            
                try:
                    if item.is_dir():
                        shutil.copytree(item, workspace_dir / item.name, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, workspace_dir / item.name)
                except Exception as e:
                    logger.warning(f"⚠️  Failed to copy {item.name}: {e}")
        
            logger.info(f"✅ Repository copied to {workspace_dir}")
        else:
            logger.warning(f"Repository directory not found: {repo_dir}")
        
        return str(workspace_dir)
    
    async def _execute_openhands_sdk(
        self,
        workspace_dir: str,
        system_prompt: str,
        user_prompt: str,
        issue_id: str,
        issue_logger=None
    ) -> str:
        """Execute OpenHands using SDK API."""
        log = issue_logger if issue_logger else logger
        
        try:
            from openhands.sdk import Conversation
            from io import StringIO
            import contextlib
            import sys
            
            # Combine system and user prompts
            combined_prompt = f"{system_prompt}\n\nTask:\n{user_prompt}"
            
            log.info(f"🚀 Starting OpenHands SDK conversation...")
            self._flush_logger(log)
            
            # Create conversation with workspace
            conversation = Conversation(
                agent=self._oh_agent,
                workspace=workspace_dir
            )
            
            # Send message
            conversation.send_message(combined_prompt)
            
            captured_stdout = StringIO()
            captured_stderr = StringIO()
            
            with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
                status = conversation.run()
            
            stdout_content = captured_stdout.getvalue()
            response = self._extract_response_from_stdout(stdout_content, issue_logger=log)
            
            if response == "OpenHands conversation completed but no response extracted.":
                log.warning("Stdout extraction failed, trying EventLog fallback...")
                eventlog_response = self._extract_response_from_conversation(conversation, issue_logger=log)
                if eventlog_response != "OpenHands conversation completed. Detailed actions were logged to console output.":
                    response = eventlog_response
            
            log.info(f"✅ OpenHands SDK completed with status: {status}")
            self._flush_logger(log)
            return response
            
        except Exception as e:
            log.error(f"❌ OpenHands SDK execution failed: {e}")
            self._flush_logger(log)
            raise AgentError(f"OpenHands SDK error: {e}", agent_type="openhands_maintainer")
    
    def _extract_response_from_stdout(self, stdout_content: str, issue_logger=None) -> str:
        """Extract agent's final message from captured stdout."""
        log = issue_logger if issue_logger else logger
        
        log.info("=" * 80)
        log.info("📊 EXTRACTING RESPONSE FROM STDOUT")
        log.info("=" * 80)
        log.info(f"Captured stdout length: {len(stdout_content)} chars")
        
        try:
            if "Message from Agent" not in stdout_content:
                log.warning("⚠️  No 'Message from Agent' marker found in stdout")
                return "OpenHands conversation completed but no response extracted."
            
            parts = stdout_content.split("Message from Agent")
            
            if len(parts) <= 1:
                log.warning("⚠️  Found marker but couldn't split content")
                return "OpenHands conversation completed but no response extracted."
            
            last_agent_section = parts[-1]
            lines = last_agent_section.split('\n')
            message_lines = []
            in_message = False
            
            for line in lines:
                if '─' in line and not in_message:
                    in_message = True
                    continue
                elif any(marker in line for marker in ['Tokens:', 'Agent Action', 'Observation', '═']):
                    break
                elif in_message and line.strip():
                    message_lines.append(line)
            
            agent_message = '\n'.join(message_lines).strip()
            
            if agent_message:
                log.info(f"✅ Extracted agent message from stdout ({len(agent_message)} chars)")
                log.info(f"📝 Message preview: {agent_message[:200]}...")
                return agent_message
            else:
                log.warning("⚠️  Message extraction resulted in empty string")
                return "OpenHands conversation completed but message extraction failed."
                
        except Exception as e:
            log.error(f"❌ Error extracting response from stdout: {e}", exc_info=True)
            return "OpenHands conversation completed but response extraction encountered an error."
    
    def _flush_logger(self, log: logging.Logger):
        """Flush all logger handlers."""
        try:
            for handler in log.handlers:
                if hasattr(handler, 'flush'):
                    handler.flush()
        except Exception as e:
            logger.error(f"Error flushing logger: {e}")
    
    def _normalize_model_name(self, model_name: str) -> str:
        """Normalize model name for OpenHands/litellm compatibility."""
        if "/" in model_name:
            return model_name
        
        if model_name.startswith("claude-"):
            return f"anthropic/{model_name}"
        elif model_name.startswith("gpt-"):
            return f"openai/{model_name}"
        
        return model_name
    
    def _extract_response_from_conversation(self, conversation, issue_logger=None) -> str:
        """Extract agent response from conversation using EventLog."""
        log = issue_logger if issue_logger else logger
        
        log.info("=" * 80)
        log.info("📊 OPENHANDS CONVERSATION ANALYSIS")
        log.info("=" * 80)
        log.info("ℹ️  OpenHands detailed actions are printed to console output")
        log.info("ℹ️  (Agent thoughts, tool calls, and observations appear in terminal)")
        log.info("=" * 80)
        self._flush_logger(log)
        
        try:
            if not hasattr(conversation, 'state') or not hasattr(conversation.state, 'events'):
                log.warning("⚠️  No state.events found in conversation")
                self._flush_logger(log)
                return "OpenHands completed but conversation state not accessible."
            
            event_log = conversation.state.events
            
            all_responses = []
            agent_final_messages = []
            event_count = 0
            
            i = 0
            max_iterations = 1000
            while i < max_iterations:
                try:
                    event = event_log.get_index(i)
                    event_count += 1
                    event_type = type(event).__name__
                    content = None
                    if hasattr(event, 'message') and event.message:
                        content = str(event.message)
                        if 'Agent' in event_type or 'Message' in event_type:
                            agent_final_messages.append(content)
                            log.info(f"Event {i+1}: {event_type} - Agent message ({len(content)} chars)")
                    elif hasattr(event, 'content') and event.content:
                        content = str(event.content)
                        log.info(f"Event {i+1}: {event_type} - Content ({len(content)} chars)")
                    elif hasattr(event, 'thought') and event.thought:
                        log.info(f"Event {i+1}: {event_type} - Thought")
                    elif hasattr(event, 'action') and event.action:
                        log.info(f"Event {i+1}: {event_type} - Action")
                    else:
                        log.info(f"Event {i+1}: {event_type}")
                    
                    if content:
                        all_responses.append(content)
                    
                    i += 1
                    if i % 10 == 0:
                        self._flush_logger(log)
                        
                except (IndexError, Exception) as e:
                    if i == 0:
                        log.warning(f"⚠️  Could not access any events: {e}")
                    break
            
            log.info(f"📊 EventLog contains {event_count} events")
            self._flush_logger(log)
            
            if agent_final_messages:
                final_response = agent_final_messages[-1]
                log.info(f"✅ Extracted final agent message ({len(final_response)} chars)")
                log.info(f"📝 Response preview: {final_response[:300]}...")
                self._flush_logger(log)
                return final_response
            elif all_responses:
                final_response = all_responses[-1]
                log.info(f"⚠️  Using last available content ({len(final_response)} chars)")
                self._flush_logger(log)
                return final_response
            else:
                log.warning("⚠️  No content extracted from EventLog")
                log.info("ℹ️  OpenHands actions were logged to console but not captured programmatically")
                self._flush_logger(log)
                return "OpenHands conversation completed. Detailed actions were logged to console output."
                
        except Exception as e:
            log.error(f"❌ Error extracting response from conversation: {e}", exc_info=True)
            self._flush_logger(log)
            return "OpenHands conversation completed but response extraction encountered an error."
    
    async def generate_docker_response(
        self,
        repo_dir: str,
        issue_data: IssueData,
        conversation_history: List[ConversationMessage],
        **kwargs
    ) -> Tuple[str, Dict[str, str], Optional[str]]:
        """Generate Docker-aware response using OpenHands."""
        latest_user_message = ""
        for message in reversed(conversation_history):
            if message.role == "user":
                latest_user_message = message.content
                break
        
        docker_system_prompt = self.get_system_prompt(
            is_docker_issue=True,
            repo_url=issue_data.commit_info.repository,
            commit_hash=issue_data.commit_info.sha
        )
        
        user_prompt = f"""
Original question: {issue_data.first_question.title}

Conversation history:
{self._format_conversation_history(conversation_history)}

Dockerfile:
{issue_data.dockerfile or 'No Dockerfile provided'}

Latest user message: {latest_user_message}

Please respond to the user's Docker-related issue with specific solutions.
If you need to create files or modify the Dockerfile, please do so using your available tools.
"""
        
        response = await self.generate_response(
            user_prompt,
            docker_system_prompt,
            issue_data.id,
            repo_dir=repo_dir
        )
        
        extra_files = self._extract_created_files(repo_dir, issue_data.id)
        modified_dockerfile = self._extract_modified_dockerfile(repo_dir, issue_data.id)
        
        return response, extra_files, modified_dockerfile
    
    async def generate_standard_response(
        self,
        repo_dir: str,
        issue_data: IssueData,
        conversation_history: List[ConversationMessage],
        **kwargs
    ) -> Tuple[str, str]:
        """Generate standard maintainer response using OpenHands."""
        latest_user_message = ""
        for message in reversed(conversation_history):
            if message.role == "user":
                latest_user_message = message.content
                break
        
        system_prompt = self.get_system_prompt(
            repo_url=issue_data.commit_info.repository,
            commit_hash=issue_data.commit_info.sha
        )
        
        user_prompt = f"""
Original question: {issue_data.first_question.title}

Conversation history:
{self._format_conversation_history(conversation_history)}

Latest user message: {latest_user_message}

Please respond to the user's message with a helpful, accurate answer.
Use your tools to explore the repository if needed.
"""
        
        response = await self.generate_response(
            user_prompt,
            system_prompt,
            issue_data.id,
            repo_dir=repo_dir
        )
        
        exploration_results = "OpenHands exploration logged internally"
        
        return response, exploration_results
    
    def _extract_created_files(self, repo_dir: str, issue_id: str) -> Dict[str, str]:
        """Extract files created by OpenHands in workspace."""
        workspace_dir = Path(self.workspace_base) / f"workspace_{issue_id}"
        extra_files = {}
        
        if not workspace_dir.exists():
            return extra_files
        
        repo_files = set(Path(repo_dir).rglob("*")) if Path(repo_dir).exists() else set()
        workspace_files = set(workspace_dir.rglob("*"))
        
        for file_path in workspace_files:
            if file_path.is_file():
                relative_path = file_path.relative_to(workspace_dir)
                original_path = Path(repo_dir) / relative_path
                
                if not original_path.exists():
                    try:
                        content = file_path.read_text()
                        extra_files[str(relative_path)] = content
                        logger.info(f"📄 Detected new file created: {relative_path}")
                    except Exception as e:
                        logger.warning(f"Could not read created file {relative_path}: {e}")
        
        return extra_files
    
    def _extract_modified_dockerfile(self, repo_dir: str, issue_id: str) -> Optional[str]:
        """Extract modified Dockerfile if changed by OpenHands."""
        workspace_dir = Path(self.workspace_base) / f"workspace_{issue_id}"
        dockerfile_path = workspace_dir / "Dockerfile"
        
        if dockerfile_path.exists():
            try:
                modified_content = dockerfile_path.read_text()
                
                original_dockerfile = Path(repo_dir) / "Dockerfile"
                if original_dockerfile.exists():
                    original_content = original_dockerfile.read_text()
                    if modified_content != original_content:
                        logger.info("📋 Dockerfile was modified by OpenHands")
                        return modified_content
                else:
                    logger.info("📋 New Dockerfile created by OpenHands")
                    return modified_content
            except Exception as e:
                logger.warning(f"Could not read Dockerfile: {e}")
        
        return None
    
    async def choose_commit(self, reference_commit: str, user_question: str) -> str:
        """Decide commit to use for exploration."""
        from ..prompts.constants import TaskPrompts, ValidationPatterns
        import re
        
        system_prompt = TaskPrompts.COMMIT_SELECTION
        
        user_prompt = f"""
        Reference commit: {reference_commit}
        
        User's question: {user_question}
        
        Has the user explicitly mentioned a specific commit hash they want me to examine? 
        If yes, what is that hash? If no, respond with USE_REFERENCE_COMMIT.
        """
        
        try:
            response = await self.generate_response(user_prompt, system_prompt, issue_id="commit_selection")
            response_text = response.strip()
            
            if "USE_REFERENCE_COMMIT" in response_text:
                logger.info(f"No specific commit mentioned. Using reference commit: {reference_commit}")
                return reference_commit
            else:
                hash_match = re.search(ValidationPatterns.GIT_COMMIT_PATTERN, response_text, re.IGNORECASE)
                if hash_match:
                    user_commit = hash_match.group(0)
                    logger.info(f"User specified commit detected: {user_commit}")
                    return user_commit
                else:
                    logger.warning(f"Unexpected response format. Using reference commit: {reference_commit}")
                    return reference_commit
        except Exception as e:
            logger.error(f"Error in commit selection: {e}")
            return reference_commit
    
    def _format_conversation_history(self, history: List[ConversationMessage]) -> str:
        """Format conversation history for display."""
        formatted = ""
        for message in history:
            role = "User" if message.role == "user" else "Maintainer"
            formatted += f"{role}: {message.content}\n\n"
        return formatted
    
    def cleanup(self):
        """Clean up OpenHands workspace."""
        try:
            if Path(self.workspace_base).exists():
                shutil.rmtree(self.workspace_base)
                logger.info(f"🧹 Cleaned up OpenHands workspace: {self.workspace_base}")
        except Exception as e:
            logger.warning(f"Failed to cleanup workspace: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        return False
    
    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()
