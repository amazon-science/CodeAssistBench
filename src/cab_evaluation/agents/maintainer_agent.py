"""Maintainer agent implementation."""

import os
import re
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from ..core.models import IssueData, ConversationMessage, ExplorationResult
from ..prompts.constants import TaskPrompts, ValidationPatterns
from .strands_agent import StrandsAgent
from ..core.exceptions import AgentError

import logging

logger = logging.getLogger(__name__)


class MaintainerAgent(StrandsAgent):
    """Agent that acts as a software project maintainer with Strands framework capabilities."""
    
    def __init__(self, model_name: str = "sonnet37", read_only: bool = False, **kwargs):
        """Initialize maintainer agent.
        
        Args:
            model_name: Model to use for maintainer responses (defaults to sonnet37)
            read_only: If True, only read-only tools are enabled (defaults to False for maintainers)
            **kwargs: Additional arguments for Strands agent
        """
        # Initialize with maintainer-specific settings, ensuring read_only is passed properly
        super().__init__(model_name=model_name, read_only=read_only, **kwargs)
        self.agent_type = "maintainer"  # Override agent_type from StrandsAgent
        self.read_only = read_only  # Ensure the attribute is accessible
    
    def get_system_prompt(self, **kwargs) -> str:
        """Get maintainer system prompt.
        
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
        
        # Add Docker-specific guidance if this is a Docker issue
        docker_guidance = ""
        if kwargs.get('is_docker_issue'):
            docker_guidance = self.prompt_manager.get_prompt("maintainer/docker_prompt")
        
        return f"{base_prompt}{repo_context}{docker_guidance}"
    
    async def generate_response(
        self,
        user_prompt: str,
        system_prompt: str,
        issue_id: str = "unknown",
        **kwargs
    ) -> str:
        """Generate maintainer response using Strands framework.
        
        Args:
            user_prompt: User input
            system_prompt: System prompt (maintainer-specific with repository context)
            issue_id: Issue ID for tracking
            **kwargs: Additional arguments
            
        Returns:
            Generated response with enhanced tool capabilities
        """
        # Enhanced system prompt for Strands agent with repository context
        strands_system_prompt = system_prompt + """

CRITICAL WORKING CONTEXT:
- You are working exclusively within a cloned repository directory
- All commands execute within this repository (cd commands are relative to repo root)
- All file paths are relative to the repository root
- Focus ONLY on the files and code within this repository
- Do NOT reference external projects or parent directories
- This repository contains the code the user is asking about

TOOL USAGE GUIDELINES:
- Use fs_read to read files within the repository
- Use execute_bash to run commands within the repository  
- When exploring, stay within the repository directory structure
- All exploration should be focused on understanding THIS codebase
"""
        
        # Use Strands framework with enhanced repository-aware system prompt
        return await super().generate_response(user_prompt, strands_system_prompt, issue_id, **kwargs)
    
    async def generate_docker_response(
        self,
        repo_dir: str,
        issue_data: IssueData,
        conversation_history: List[ConversationMessage],
        **kwargs
    ) -> Tuple[str, Dict[str, str], Optional[str]]:
        """Generate Docker-aware response from maintainer agent.
        
        Args:
            repo_dir: Repository directory path
            issue_data: Issue data
            conversation_history: Conversation history
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (response, extra_files, modified_dockerfile)
        """
        # Extract the user's latest question
        latest_user_message = ""
        for message in reversed(conversation_history):
            if message.role == "user":
                latest_user_message = message.content
                break
        
        # Create Docker-specific system prompt
        docker_system_prompt = self.get_system_prompt(
            is_docker_issue=True,
            repo_url=issue_data.commit_info.repository,
            commit_hash=issue_data.commit_info.sha
        ) + TaskPrompts.DOCKER_EXPLORATION
        
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
        
        # Get maintainer's response using Strands framework
        response = await super().generate_response(
            user_prompt, docker_system_prompt, issue_data.id
        )
        
        # Process file creation/modifications
        extra_files = {}
        modified_dockerfile = None
        
        # Extract CREATE_FILE blocks
        file_matches = re.finditer(
            ValidationPatterns.CREATE_FILE_PATTERN, response, re.DOTALL
        )
        for match in file_matches:
            filename = match.group(1).strip()
            content = match.group(2).strip()
            extra_files[filename] = content
        
        # Extract MODIFY_DOCKERFILE block
        dockerfile_match = re.search(
            ValidationPatterns.MODIFY_DOCKERFILE_PATTERN, response, re.DOTALL
        )
        if dockerfile_match:
            modified_dockerfile = dockerfile_match.group(1).strip()
        
        # Clean up the response (remove the special formatting)
        cleaned_response = re.sub(
            ValidationPatterns.CREATE_FILE_PATTERN,
            r'I have created a file named \1 with the necessary content.',
            response,
            flags=re.DOTALL
        )
        cleaned_response = re.sub(
            ValidationPatterns.MODIFY_DOCKERFILE_PATTERN,
            r'I have modified the Dockerfile with the necessary changes.',
            cleaned_response,
            flags=re.DOTALL
        )
        
        return cleaned_response, extra_files, modified_dockerfile
    
    async def generate_standard_response(
        self,
        repo_dir: str,
        issue_data: IssueData,
        conversation_history: List[ConversationMessage],
        **kwargs
    ) -> Tuple[str, str]:
        """Generate standard maintainer response with exploration.
        
        Args:
            repo_dir: Repository directory
            issue_data: Issue data
            conversation_history: Conversation history
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (response, exploration_results)
        """
        # Extract latest user message
        latest_user_message = ""
        for message in reversed(conversation_history):
            if message.role == "user":
                latest_user_message = message.content
                break
        
        # Create system prompt
        system_prompt = self.get_system_prompt(
            repo_url=issue_data.commit_info.repository,
            commit_hash=issue_data.commit_info.sha
        ) + """
        You are in a conversation with a user who is asking questions about a code issue.
        Respond to their latest message, using your repository knowledge to provide accurate information.
        
        IMPORTANT GUIDELINES:
        1. Include ALL relevant information needed to fully answer the user's question.
        2. If you've provided partial information in previous responses, INCLUDE that information again.
        3. When referencing code or solutions you've mentioned before, ALWAYS include the full code or solution again.
        4. Make sure your answer is complete even if it means repeating information.
        
        If you need to explore the repository further to answer their question, you can do so using:
        EXPLORE: <command to run>
        
        Then provide your complete answer to the user in a clear, helpful manner.
        """
        
        # Create user prompt
        user_prompt = f"""
        Original question: {issue_data.first_question.title}
        
        Conversation history:
        {self._format_conversation_history(conversation_history)}
        
        Latest user message: {latest_user_message}
        
        Please respond to the user's latest message.
        """
        
        # Get maintainer response using Strands framework
        maintainer_response = await super().generate_response(
            user_prompt, system_prompt, issue_data.id
        )
        
        # Process exploration commands
        exploration_results = ""
        if "EXPLORE:" in maintainer_response:
            self.logger.info("Processing exploration commands in maintainer response")
            lines = maintainer_response.split('\n')
            processed_response = []
            
            for line in lines:
                if line.strip().startswith("EXPLORE:"):
                    cmd = line.split("EXPLORE:", 1)[1].strip()
                    self.logger.info(f"Executing exploration command: {cmd}")
                    # Import here to avoid circular imports
                    from ..utils.repository_manager import execute_command
                    result = execute_command(repo_dir, cmd)
                    exploration_results += f"Command: {cmd}\nResult:\n{result}\n\n"
                else:
                    processed_response.append(line)
            
            # If exploration was performed, regenerate response with results
            if exploration_results:
                final_prompt = f"""
                Original question: {issue_data.first_question.title}
                
                Conversation history:
                {self._format_conversation_history(conversation_history)}
                
                Latest user message: {latest_user_message}
                
                Exploration results:
                {exploration_results}
                
                Please provide your final response to the user based on the exploration results.
                """
                
                maintainer_response = await super().generate_response(
                    final_prompt, system_prompt, issue_data.id
                )
        
        return maintainer_response, exploration_results
    
    async def choose_commit(self, reference_commit: str, user_question: str) -> str:
        """Allow maintainer to decide commit to use for exploration.
        
        Args:
            reference_commit: Reference commit hash
            user_question: User's question text
            
        Returns:
            Selected commit hash
        """
        system_prompt = TaskPrompts.COMMIT_SELECTION
        
        user_prompt = f"""
        Reference commit: {reference_commit}
        
        User's question: {user_question}
        
        Has the user explicitly mentioned a specific commit hash they want me to examine? 
        If yes, what is that hash? If no, respond with USE_REFERENCE_COMMIT.
        """
        
        try:
            response = await super().generate_response(user_prompt, system_prompt, issue_id="commit_selection")
            response_text = response.strip()
            
            if "USE_REFERENCE_COMMIT" in response_text:
                self.logger.info(f"No specific commit mentioned. Using reference commit: {reference_commit}")
                return reference_commit
            else:
                # Try to extract a hash-like string from the response
                hash_match = re.search(ValidationPatterns.GIT_COMMIT_PATTERN, response_text, re.IGNORECASE)
                
                if hash_match:
                    user_commit = hash_match.group(0)
                    self.logger.info(f"User specified commit detected: {user_commit}")
                    return user_commit
                else:
                    self.logger.warning(f"Unexpected response format. Using reference commit: {reference_commit}")
                    return reference_commit
        except Exception as e:
            self.logger.error(f"Error in commit selection: {e}")
            return reference_commit
    
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
