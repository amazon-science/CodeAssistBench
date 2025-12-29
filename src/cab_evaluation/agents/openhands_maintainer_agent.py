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
        enable_ast_tools: bool = True,
        custom_system_prompt_path: Optional[str] = None,
        **kwargs
    ):
        """Initialize OpenHands maintainer agent.
        
        Args:
            enable_ast_tools: If True, include custom read_code/edit_code tools
            custom_system_prompt_path: Absolute path to custom system prompt template (.j2)
        """
        self.custom_system_prompt_path = custom_system_prompt_path
        # Store OpenHands-specific attributes
        self.workspace_base = tempfile.mkdtemp(prefix="openhands_cab_")
        self._is_openhands = True
        self._oh_agent = None  # Will be initialized lazily
        self._oh_llm = None
        self.enable_ast_tools = enable_ast_tools
        if not hasattr(self, 'custom_system_prompt_path'):
            self.custom_system_prompt_path = custom_system_prompt_path
        
        # Token usage tracking
        self._token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_cost_usd": 0.0,
            "cache_hit_percent": 0.0
        }
        
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
        
        logger.info(f"🤖 OpenHands SDK maintainer agent initialized with model: {model_name} (AST tools: {enable_ast_tools})")
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
        
        # Validate custom AST tools availability
        if self.enable_ast_tools:
            try:
                from ..utils.openhands_tools import OPENHANDS_AVAILABLE
                if OPENHANDS_AVAILABLE:
                    logger.info("✅ Custom AST tools (read_code, edit_code) available")
                else:
                    logger.warning("⚠️ OpenHands SDK not available for custom AST tools")
            except ImportError as e:
                logger.warning(f"⚠️ Custom AST tools not available: {e}")
    
    def _initialize_openhands_agent(self):
        """Initialize OpenHands agent and LLM."""
        if self._oh_agent is not None:
            return
        
        try:
            from openhands.sdk import Agent, LLM, Tool
            from openhands.tools.terminal import TerminalTool
            from openhands.tools.file_editor import FileEditorTool
            from ..utils.openhands_utils import validate_openhands_config, is_bedrock_model
            
            # Validate configuration and get API key (None for Bedrock)
            api_key = validate_openhands_config(self.model_name)
            
            # Ensure model name has proper format for litellm
            model_name = self._normalize_model_name(self.model_name)
            logger.info(f"🔧 Normalized model name: {self.model_name} -> {model_name}")
            
            # Create LLM - Bedrock uses AWS env vars, others use api_key
            if api_key is None:  # Bedrock model
                self._oh_llm = LLM(model=model_name)
            else:
                self._oh_llm = LLM(model=model_name, api_key=api_key)
            
            if not self._oh_llm:
                raise AgentError("LLM initialization failed", agent_type="openhands_maintainer")
            
            # Create Agent with tools (AST tools first to encourage their use)
            tools = []
            
            # Add custom AST tools first if enabled
            if self.enable_ast_tools:
                try:
                    from ..utils.openhands_tools import register_ast_tools, OPENHANDS_AVAILABLE
                    if OPENHANDS_AVAILABLE:
                        register_ast_tools()
                        tools.append(Tool(name="read_code"))
                        tools.append(Tool(name="edit_code"))
                        logger.info("✅ Custom AST tools (read_code, edit_code) registered")
                    else:
                        logger.warning("⚠️ OpenHands SDK not available for custom tools")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to register custom AST tools: {e}")
            
            # Add standard tools after AST tools
            tools.append(Tool(name=TerminalTool.name))
            tools.append(Tool(name=FileEditorTool.name))
            
            # Build agent kwargs
            agent_kwargs = {
                "llm": self._oh_llm,
                "tools": tools,
            }
            
            # Use custom system prompt if provided
            if self.custom_system_prompt_path:
                agent_kwargs["system_prompt_filename"] = self.custom_system_prompt_path
                logger.info(f"📝 Using custom system prompt: {self.custom_system_prompt_path}")
            
            self._oh_agent = Agent(**agent_kwargs)
            
            if not self._oh_agent:
                raise AgentError("Agent initialization failed", agent_type="openhands_maintainer")
            
            logger.info(f"✅ OpenHands Agent initialized with {self.model_name} ({len(tools)} tools)")
            
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
        
        return f"{base_prompt}{repo_context}"
    
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
        log.info(f"📂 repo_dir from kwargs: {repo_dir}")
        
        if not repo_dir:
            logger.debug("No repo_dir provided, using workspace base")
            repo_dir = self.workspace_base
        
        # Check if repo_dir exists and has content
        repo_path = Path(repo_dir)
        if repo_path.exists():
            items = list(repo_path.iterdir())
            log.info(f"📂 repo_dir exists with {len(items)} items: {[i.name for i in items[:5]]}...")
        else:
            log.warning(f"⚠️ repo_dir does not exist: {repo_dir}")
        
        try:
            # Initialize OpenHands agent
            self._initialize_openhands_agent()
            
            # Prepare workspace
            workspace_dir = self._prepare_workspace(repo_dir, issue_id)
            log.info(f"📂 workspace_dir: {workspace_dir}")
            
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
        
        # Check if workspace already exists and has content
        if workspace_dir.exists() and any(workspace_dir.iterdir()):
            logger.info(f"✅ Reusing existing workspace: {workspace_dir}")
            return str(workspace_dir)
        
        workspace_dir.mkdir(parents=True, exist_ok=True)
        
        repo_path = Path(repo_dir)
        if not repo_path.exists():
            logger.warning(f"⚠️ Repository directory not found: {repo_dir}")
            return str(workspace_dir)
        
        # Check if repo_dir has content
        repo_items = list(repo_path.iterdir())
        if not repo_items:
            logger.warning(f"⚠️ Repository directory is empty: {repo_dir}")
            return str(workspace_dir)
        
        logger.info(f"📁 Copying {len(repo_items)} items from {repo_dir} to workspace...")
        
        copied_count = 0
        for item in repo_items:
            if item.name == ".git":
                continue
            
            try:
                dest = workspace_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)
                copied_count += 1
            except Exception as e:
                logger.warning(f"⚠️ Failed to copy {item.name}: {e}")
        
        # Verify copy
        workspace_items = list(workspace_dir.iterdir())
        logger.info(f"✅ Copied {copied_count} items. Workspace now has {len(workspace_items)} items")
        
        if workspace_items:
            logger.info(f"📂 Workspace contents: {[i.name for i in workspace_items[:10]]}...")
        else:
            logger.error(f"❌ Workspace is empty after copy!")
        
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
            
            # Parse and accumulate token usage from stdout
            self._parse_token_usage(stdout_content, issue_logger=log)
            
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
            
            # Check for max_tokens/context length error - return special marker to end conversation
            error_str = str(e).lower()
            if "max_tokens" in error_str or "max_completion_tokens" in error_str or "context length" in error_str:
                log.warning("⚠️ Context length exceeded - ending conversation to proceed to judge evaluation")
                self._flush_logger(log)
                return "ERROR_CONTEXT_LENGTH_EXCEEDED"
            
            raise AgentError(f"OpenHands SDK error: {e}", agent_type="openhands_maintainer")
    
    def _parse_token_usage(self, stdout_content: str, issue_logger=None):
        """Parse token usage from OpenHands stdout and accumulate."""
        import re
        log = issue_logger if issue_logger else logger
        
        # Pattern: Tokens: ↑ input 5.58K • cache hit 20.63% • reasoning 320 • ↓ output 330 • $ 0.0089
        pattern = r'Tokens: ↑ input ([\d.]+)K? • cache hit ([\d.]+|N/A)%? •\s*(?:reasoning (\d+) •)?\s*↓ output ([\d.]+)K? • \$ ([\d.]+)'
        
        for match in re.finditer(pattern, stdout_content):
            try:
                input_str = match.group(1)
                input_tokens = float(input_str) * 1000 if 'K' in match.group(0).split('input')[1].split('•')[0] else float(input_str)
                
                cache_hit = 0.0
                if match.group(2) != 'N/A':
                    cache_hit = float(match.group(2))
                
                reasoning_tokens = int(match.group(3)) if match.group(3) else 0
                
                output_str = match.group(4)
                output_tokens = float(output_str) * 1000 if 'K' in match.group(0).split('output')[1].split('•')[0] else float(output_str)
                
                cost = float(match.group(5))
                
                self._token_usage["input_tokens"] += int(input_tokens)
                self._token_usage["output_tokens"] += int(output_tokens)
                self._token_usage["reasoning_tokens"] += reasoning_tokens
                self._token_usage["total_cost_usd"] += cost
                self._token_usage["cache_hit_percent"] = cache_hit  # Use latest
                
                log.info(f"📊 Parsed token usage: input={int(input_tokens)}, output={int(output_tokens)}, reasoning={reasoning_tokens}, cost=${cost:.4f}")
            except Exception as e:
                log.warning(f"Failed to parse token line: {e}")
    
    def get_token_usage(self) -> Dict[str, Any]:
        """Get accumulated token usage statistics."""
        return self._token_usage.copy()
    
    def _extract_response_from_stdout(self, stdout_content: str, issue_logger=None) -> str:
        """Extract agent's final message from captured stdout."""
        log = issue_logger if issue_logger else logger
        
        log.info("=" * 80)
        log.info("📊 EXTRACTING RESPONSE FROM STDOUT")
        log.info("=" * 80)
        log.info(f"Captured stdout length: {len(stdout_content)} chars")
        
        # Log raw stdout for debugging (first 2000 chars)
        if stdout_content:
            log.info(f"Raw stdout preview:\n{stdout_content[:2000]}")
        else:
            log.warning("⚠️ stdout_content is empty")
            return "OpenHands conversation completed but no response extracted."
        
        try:
            # Try multiple extraction methods
            
            # Method 1: Look for "Message from Agent" marker
            if "Message from Agent" in stdout_content:
                parts = stdout_content.split("Message from Agent")
                if len(parts) > 1:
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
                        log.info(f"✅ Extracted via 'Message from Agent' ({len(agent_message)} chars)")
                        return agent_message
            
            # Method 2: Look for final text block after last tool output
            if "Observation:" in stdout_content:
                parts = stdout_content.split("Observation:")
                last_part = parts[-1]
                # Find text after the observation that looks like a response
                lines = last_part.split('\n')
                response_lines = []
                started = False
                for line in lines:
                    if not started and line.strip() and not line.startswith('─') and not line.startswith('═'):
                        started = True
                    if started:
                        if any(marker in line for marker in ['Tokens:', '═', 'Agent Action']):
                            break
                        if line.strip():
                            response_lines.append(line)
                
                if response_lines:
                    response = '\n'.join(response_lines).strip()
                    if len(response) > 50:  # Reasonable response length
                        log.info(f"✅ Extracted via last Observation ({len(response)} chars)")
                        return response
            
            # Method 3: Just get the last substantial text block
            lines = stdout_content.split('\n')
            text_blocks = []
            current_block = []
            for line in lines:
                if line.strip() and not any(marker in line for marker in ['─', '═', 'Tokens:', '>>>', '...']):
                    current_block.append(line)
                elif current_block:
                    text_blocks.append('\n'.join(current_block))
                    current_block = []
            if current_block:
                text_blocks.append('\n'.join(current_block))
            
            # Get the longest text block as likely response
            if text_blocks:
                longest_block = max(text_blocks, key=len)
                if len(longest_block) > 50:
                    log.info(f"✅ Extracted longest text block ({len(longest_block)} chars)")
                    return longest_block
            
            log.warning("⚠️ All extraction methods failed")
            return "OpenHands conversation completed but no response extracted."
                
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
        from ..utils.openhands_utils import is_bedrock_model
        
        # Already has provider prefix
        if "/" in model_name:
            return model_name
        
        # Map CAB model names to Bedrock litellm format
        bedrock_model_mapping = {
            "qwen32b": "bedrock/qwen.qwen3-32b-v1:0",
            "sonnet37": "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "sonnet45": "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "sonnet": "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "haiku": "bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0",
            "thinking": "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "deepseek": "bedrock/us.deepseek.r1-v1:0",
            "llama": "bedrock/us.meta.llama3-3-70b-instruct-v1:0",
        }
        
        # Check if it's a known CAB model that maps to Bedrock
        if model_name.lower() in bedrock_model_mapping:
            mapped = bedrock_model_mapping[model_name.lower()]
            logger.info(f"Mapped CAB model '{model_name}' -> Bedrock '{mapped}'")
            return mapped
        
        # Check if it looks like a Bedrock model ID
        if is_bedrock_model(model_name):
            return f"bedrock/{model_name}"
        
        # Anthropic models
        if model_name.startswith("claude-"):
            return f"anthropic/{model_name}"
        
        # OpenAI models
        if model_name.startswith("gpt-"):
            return f"openai/{model_name}"
        
        return model_name
    
    def _extract_response_from_conversation(self, conversation, issue_logger=None) -> str:
        """Extract agent response from conversation using EventLog."""
        log = issue_logger if issue_logger else logger
        
        log.info("=" * 80)
        log.info("📊 EXTRACTING FROM EVENTLOG")
        log.info("=" * 80)
        self._flush_logger(log)
        
        try:
            # Try to get events from conversation state
            events = None
            if hasattr(conversation, 'state'):
                if hasattr(conversation.state, 'events'):
                    events = conversation.state.events
                elif hasattr(conversation.state, 'history'):
                    events = conversation.state.history
            
            if events is None:
                log.warning("⚠️ No events found in conversation")
                return "OpenHands completed but conversation state not accessible."
            
            all_messages = []
            
            # Try iterating over events
            try:
                # Method 1: Direct iteration
                for event in events:
                    event_type = type(event).__name__
                    content = None
                    
                    # Check various attributes for content
                    for attr in ['message', 'content', 'text', 'output']:
                        if hasattr(event, attr):
                            val = getattr(event, attr)
                            if val and isinstance(val, str) and len(val) > 10:
                                content = val
                                break
                    
                    if content:
                        all_messages.append((event_type, content))
                        log.info(f"Event: {event_type} - {len(content)} chars")
            except TypeError:
                # Method 2: Index-based access
                i = 0
                while i < 1000:
                    try:
                        event = events.get_index(i) if hasattr(events, 'get_index') else events[i]
                        event_type = type(event).__name__
                        content = None
                        
                        for attr in ['message', 'content', 'text', 'output']:
                            if hasattr(event, attr):
                                val = getattr(event, attr)
                                if val and isinstance(val, str) and len(val) > 10:
                                    content = val
                                    break
                        
                        if content:
                            all_messages.append((event_type, content))
                        i += 1
                    except (IndexError, KeyError):
                        break
            
            log.info(f"Found {len(all_messages)} messages in EventLog")
            
            if all_messages:
                # Return the last substantial message
                final_type, final_content = all_messages[-1]
                log.info(f"✅ Using last message from {final_type} ({len(final_content)} chars)")
                return final_content
            
            return "OpenHands conversation completed. No extractable response found."
                
        except Exception as e:
            log.error(f"❌ Error extracting from EventLog: {e}", exc_info=True)
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
        """Decide commit to use for exploration.
        
        This is a simple text decision that doesn't need OpenHands tools.
        Just use the LLM directly without workspace.
        """
        from ..prompts.constants import TaskPrompts, ValidationPatterns
        import re
        
        # For commit selection, just return reference commit
        # This avoids the overhead of OpenHands for a simple decision
        # Check if user explicitly mentioned a commit hash
        hash_match = re.search(ValidationPatterns.GIT_COMMIT_PATTERN, user_question, re.IGNORECASE)
        if hash_match:
            user_commit = hash_match.group(0)
            logger.info(f"User specified commit detected in question: {user_commit}")
            return user_commit
        
        logger.info(f"No specific commit mentioned. Using reference commit: {reference_commit}")
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
