"""User agent implementation."""

import json
import re
from typing import Dict, List, Optional, Any

from ..core.models import IssueData, ConversationMessage, SatisfactionStatus
from ..prompts.constants import TaskPrompts
from .strands_agent import StrandsAgent

import logging

logger = logging.getLogger(__name__)


class UserAgent(StrandsAgent):
    """Agent that simulates a user seeking help with technical questions."""
    
    def __init__(self, model_name: str = "sonnet37", **kwargs):
        """Initialize user agent.
        
        Args:
            model_name: Model to use for user responses (defaults to sonnet37)
            **kwargs: Additional arguments for Strands agent
        """
        # Initialize with user-specific settings
        super().__init__(model_name=model_name, **kwargs)
        self.agent_type = "user"  # Override agent_type from StrandsAgent
    
    def get_system_prompt(self, **kwargs) -> str:
        """Get user system prompt.
        
        Args:
            **kwargs: Additional context (issue_data, docker_info, etc.)
            
        Returns:
            System prompt string
        """
        base_prompt = self.prompt_manager.get_prompt("user/system_prompt")
        
        # Add issue-specific context
        issue_context = ""
        if 'issue_data' in kwargs:
            issue_data = kwargs['issue_data']
            issue_context = f"""
            Your original question was: "{issue_data.first_question.title}"
            
            You have certain expectations about what would make a satisfactory answer to your question.
            These satisfaction conditions are:
            {json.dumps(issue_data.user_satisfaction_condition, indent=2)}
            """
        
        # Add Docker validation results if available
        docker_info = ""
        if 'docker_results' in kwargs and kwargs['docker_results']:
            docker_results = kwargs['docker_results']
            success_status = docker_results.get('success', False)
            docker_info = f"""
            You also have access to the results of running Docker commands to test the solution provided by the maintainer:
            
            Docker build and test {"succeeded" if success_status else "failed"}
            
            Test commands that were run:
            {json.dumps(docker_results.get('test_commands', []), indent=2)}
            
            Command output:
            {docker_results.get('logs', '')}
            
            IMPORTANT: Use these Docker test results to determine if your satisfaction conditions have been met.
            If your question involves a Docker setup that needs to work correctly, a successful Docker build and test
            is a strong indicator that the solution works. Conversely, if Docker tests fail, analyze the logs to
            understand what's still not working correctly.
            """
        
        # Add style guidance if provided
        style_guidance = ""
        if 'style_data' in kwargs and kwargs['style_data']:
            style_data = kwargs['style_data']
            user_style = style_data.get('user_style')
            if user_style:
                top_responses = user_style.get('sample_responses', [])[:3]
                
                style_guidance = f"""
                When responding, mimic the communication style of the user based on these style references:

                EXAMPLE 1: "{top_responses[0][:1000]}..." if top_responses else ""
                EXAMPLE 2: "{top_responses[1][:1000]}..." if len(top_responses) > 1 else ""
                EXAMPLE 3: "{top_responses[2][:1000]}..." if len(top_responses) > 2 else ""

                IMPORTANT: These examples are provided ONLY for style reference. Do NOT use any specific information 
                or content from these examples in your response. Your response should be based solely on the current 
                conversation and issue at hand, using only the communication style as a guide.

                Emulate the tone, sentence structure, and general communication approach, but generate your own 
                content relevant to the current conversation.
                """
        
        return f"{base_prompt}{issue_context}{docker_info}{style_guidance}{TaskPrompts.SATISFACTION_EVALUATION}"
    
    async def generate_response(
        self,
        user_prompt: str,
        system_prompt: str,
        issue_id: str = "unknown",
        **kwargs
    ) -> str:
        """Generate user response using Strands framework.
        
        Args:
            user_prompt: User input
            system_prompt: System prompt  
            issue_id: Issue ID for tracking
            **kwargs: Additional arguments
            
        Returns:
            Generated response with enhanced tool capabilities
        """
        # Use Strands framework for enhanced capabilities
        return await super().generate_response(user_prompt, system_prompt, issue_id, **kwargs)
    
    async def respond_to_maintainer(
        self,
        issue_data: IssueData,
        conversation_history: List[ConversationMessage],
        docker_results: Optional[Dict[str, Any]] = None,
        style_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate response to maintainer with satisfaction evaluation.
        
        Args:
            issue_data: Issue data
            conversation_history: Conversation history
            docker_results: Docker validation results
            style_data: User style analysis data
            
        Returns:
            Dictionary with response, satisfaction_status, and satisfaction_reason
        """
        self.logger.info("Activating user agent to respond to maintainer")
        
        # Create system prompt with context
        system_prompt = self.get_system_prompt(
            issue_data=issue_data,
            docker_results=docker_results,
            style_data=style_data
        )
        
        # Create formatted conversation history string
        formatted_conversation = self._format_conversation_history(conversation_history)
        
        # Create user prompt
        user_prompt = f"""
        Here is your original question:
        {issue_data.first_question.body}
        
        Here is your conversation with the maintainer so far:
        {formatted_conversation}
        
        Please provide your next response to the maintainer. Remember to focus on your satisfaction conditions 
        and ask for clarifications if needed. DO NOT express satisfaction unless all your conditions are fully met.
        
        After your response to the maintainer, add the SATISFACTION_STATUS section to evaluate whether your needs
        have been fully met.
        """
        
        # Get user agent's response using Strands framework
        self.logger.info("Requesting response from user agent")
        full_response = await super().generate_response(user_prompt, system_prompt, issue_data.id)
        self.logger.info(f"User agent response complete ({len(full_response)} chars)")
        
        # Parse satisfaction status
        satisfaction_result = self._parse_satisfaction_status(full_response)
        
        # Log satisfaction status
        self.logger.info(f"User satisfaction status: {satisfaction_result['satisfaction_status']}")
        self.logger.info(f"Satisfaction reason: {satisfaction_result['satisfaction_reason']}")
        
        return satisfaction_result
    
    def _parse_satisfaction_status(self, full_response: str) -> Dict[str, Any]:
        """Parse satisfaction status from user response.
        
        Args:
            full_response: Full response from user agent
            
        Returns:
            Dictionary with response, satisfaction_status, and satisfaction_reason
        """
        # Default values
        satisfaction_status = SatisfactionStatus.NOT_SATISFIED
        satisfaction_reason = "No explicit satisfaction status provided"
        
        # Look for satisfaction status section
        if "SATISFACTION_STATUS:" in full_response:
            # Extract the actual response part (before satisfaction status)
            parts = full_response.split("SATISFACTION_STATUS:", 1)
            response_to_maintainer = parts[0].strip()
            
            # Extract satisfaction information
            status_section = parts[1].strip()
            if "FULLY_SATISFIED" in status_section:
                satisfaction_status = SatisfactionStatus.FULLY_SATISFIED
            elif "PARTIALLY_SATISFIED" in status_section:
                satisfaction_status = SatisfactionStatus.PARTIALLY_SATISFIED
            else:
                satisfaction_status = SatisfactionStatus.NOT_SATISFIED
                
            # Try to extract reason if present
            if "REASON:" in status_section:
                reason_part = status_section.split("REASON:", 1)[1].strip()
                satisfaction_reason = reason_part.split("\n")[0].strip()
        else:
            # If no satisfaction status is found, use the whole response
            response_to_maintainer = full_response
        
        return {
            "response": response_to_maintainer,
            "satisfaction_status": satisfaction_status,
            "satisfaction_reason": satisfaction_reason
        }
    
    def _format_conversation_history(self, history: List[ConversationMessage]) -> str:
        """Format conversation history for better readability.
        
        Args:
            history: List of conversation messages
            
        Returns:
            Formatted conversation string
        """
        formatted = ""
        for i, message in enumerate(history):
            role = "User" if message.role == "user" else "Maintainer"
            formatted += f"{role}: {message.content}\n\n"
        return formatted
