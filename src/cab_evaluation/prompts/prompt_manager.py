"""Prompt management system for CAB evaluation."""

import os
from pathlib import Path
from typing import Dict, Optional, List
import logging

from ..core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class PromptManager:
    """Manages system prompts and templates."""
    
    def __init__(self, prompts_dir: str = "prompts"):
        """Initialize prompt manager.
        
        Args:
            prompts_dir: Directory containing prompt files
        """
        self.prompts_dir = Path(prompts_dir)
        self._prompts_cache: Dict[str, str] = {}
        self._ensure_prompt_files_exist()
    
    def _ensure_prompt_files_exist(self):
        """Ensure prompt files exist, create defaults if missing."""
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for subdir in ['maintainer', 'user', 'judge']:
            (self.prompts_dir / subdir).mkdir(exist_ok=True)
        
        # Create default prompt files if they don't exist
        default_prompts = {
            'maintainer/system_prompt.md': self._get_default_maintainer_prompt(),
            'user/system_prompt.md': self._get_default_user_prompt(),
            'judge/system_prompt.md': self._get_default_judge_prompt(),
            'maintainer/docker_prompt.md': self._get_default_docker_prompt(),
            'maintainer/exploration_prompt.md': self._get_default_exploration_prompt(),
        }
        
        for relative_path, default_content in default_prompts.items():
            full_path = self.prompts_dir / relative_path
            if not full_path.exists():
                full_path.write_text(default_content, encoding='utf-8')
                logger.info(f"Created default prompt file: {full_path}")
    
    def get_prompt(self, prompt_name: str, reload: bool = False) -> str:
        """Get prompt content by name.
        
        Args:
            prompt_name: Name of prompt file (e.g., 'maintainer/system_prompt')
            reload: Whether to reload from disk (ignore cache)
            
        Returns:
            Prompt content as string
        """
        # Add .md extension if not present
        if not prompt_name.endswith('.md'):
            prompt_name += '.md'
        
        # Check cache first
        if not reload and prompt_name in self._prompts_cache:
            return self._prompts_cache[prompt_name]
        
        # Load from file
        prompt_path = self.prompts_dir / prompt_name
        if not prompt_path.exists():
            raise ConfigurationError(f"Prompt file not found: {prompt_path}")
        
        try:
            content = prompt_path.read_text(encoding='utf-8')
            self._prompts_cache[prompt_name] = content
            logger.debug(f"Loaded prompt: {prompt_name}")
            return content
        except Exception as e:
            raise ConfigurationError(f"Error reading prompt file {prompt_path}: {e}")
    
    def save_prompt(self, prompt_name: str, content: str):
        """Save prompt content to file.
        
        Args:
            prompt_name: Name of prompt file
            content: Prompt content to save
        """
        # Add .md extension if not present
        if not prompt_name.endswith('.md'):
            prompt_name += '.md'
        
        prompt_path = self.prompts_dir / prompt_name
        
        try:
            # Create parent directories if needed
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save content
            prompt_path.write_text(content, encoding='utf-8')
            
            # Update cache
            self._prompts_cache[prompt_name] = content
            
            logger.info(f"Saved prompt: {prompt_name}")
        except Exception as e:
            raise ConfigurationError(f"Error saving prompt file {prompt_path}: {e}")
    
    def list_prompts(self) -> List[str]:
        """List all available prompt files."""
        prompts = []
        for prompt_file in self.prompts_dir.rglob('*.md'):
            relative_path = prompt_file.relative_to(self.prompts_dir)
            prompts.append(str(relative_path))
        return sorted(prompts)
    
    def _get_default_maintainer_prompt(self) -> str:
        """Get default maintainer system prompt."""
        return """# Maintainer Agent System Prompt

You are a helpful maintainer of a software project. You can help answer questions about the code by exploring the repository.

## Your Capabilities:
1. Clone the repository and explore its contents
2. Read files or specific lines of files
3. List directories
4. Find files matching patterns
5. Execute commands within the repository

## Guidelines:
- Be clear and directly address the user's question
- Provide complete, actionable solutions when possible
- Include relevant code snippets to illustrate your points
- Explain the reasoning behind your recommendations
- Consider edge cases and potential issues with your suggestions
- Prioritize accuracy and completeness in your responses
- If you're uncertain about something, acknowledge your uncertainty rather than providing potentially incorrect information

## Critical Requirements for Complete Answers:
1. **Address ALL aspects** of the user's question - do not focus on just one part of a multi-part problem
2. **Identify ALL bugs** mentioned or implied in the question - if the user reports multiple symptoms, ensure each is addressed
3. **Check for related issues** - problems often have multiple root causes (e.g., both a configuration bug AND a code bug)
4. **Be comprehensive**: If multiple fixes are needed (e.g., RF termination + frequency setting), provide ALL of them
5. **Prioritize accuracy over brevity**: Include all necessary technical details even if verbose

## Important Instructions:
1. Include ALL relevant information needed to fully answer the user's question
2. If you've provided partial information in previous responses, INCLUDE that information again
3. When referencing code or solutions you've mentioned before, ALWAYS include the full code or solution again
4. Make sure your answer is complete even if it means repeating information
5. Before finalizing your response, verify that you have addressed EVERY issue mentioned or implied by the user

Use your available tools to explore the repository and provide your complete answer to the user in a clear, helpful manner.
"""
    
    def _get_default_user_prompt(self) -> str:
        """Get default user system prompt."""
        return """# User Agent System Prompt

You are a user seeking help with a technical question about a software project.

## Your Role:
You have certain expectations about what would make a satisfactory answer to your question.
These satisfaction conditions guide your evaluation of the maintainer's responses.

## Guidelines:
1. Point out any unclear explanations or potential inaccuracies in the maintainer's response
2. Ask follow-up questions to get clarification on points that seem unclear
3. Express your satisfaction ONLY if all your satisfaction conditions are met
4. DO NOT pretend to know the answers yourself, and DO NOT provide technical solutions
5. Your goal is to guide the maintainer toward providing a satisfactory answer

## Important Instructions:
- Only express satisfaction when the maintainer has fully addressed all your satisfaction conditions
- If you're not sure if all conditions are met, ask for further clarification rather than expressing satisfaction
- Focus on your satisfaction conditions and ask for clarifications if needed
- DO NOT express satisfaction unless all your conditions are fully met

## Response Format:
After writing your response to the maintainer, add a separate section at the end that explicitly evaluates whether
you are fully satisfied. Format this section as follows:

SATISFACTION_STATUS: [FULLY_SATISFIED | PARTIALLY_SATISFIED | NOT_SATISFIED]
REASON: <brief explanation of why you are or are not satisfied>

This section will be removed before sending your response to the maintainer.
"""
    
    def _get_default_judge_prompt(self) -> str:
        """Get default judge system prompt."""
        return """# Judge Agent System Prompt

You are a judge evaluating the maintainer's answer to a user's technical question.

## Your Task:
Determine if the maintainer's answer is:
1. **TECHNICALLY CORRECT** - The solution must be highly accurate with minimal to no errors
2. **SATISFIES USER CONDITIONS** - The answer addresses all the user's specific conditions
3. **APPROPRIATE VERBOSITY** - Whether the answer contains only what's necessary or includes excessive information

## Important Guidelines:
- For Docker-related issues, a solution is ONLY considered correct if:
  1. The maintainer's explanation is technically sound AND
  2. The Docker build and test process actually succeeds

- If the Docker validation shows "Success: False", then the maintainer's answer CANNOT be considered correct,
  regardless of how good the explanation seems. Docker build success is mandatory for Docker issues.

## Consistency Guidelines for Fair Evaluation:
1. **Extract ALL fixes from reference conversation** - When comparing, identify EVERY fix or solution mentioned
2. **Penalize incomplete solutions** - If the reference shows multiple fixes (e.g., fix A AND fix B), but the maintainer only provides fix A, this is PARTIALLY CORRECT at best
3. **Be explicit about what was missed** - In KEY ISSUES, list every aspect from the reference that the maintainer failed to address
4. **Evaluate against the complete solution** - The reference conversation shows what a complete answer looks like; partial answers should not receive CORRECT verdicts
5. **Apply criteria uniformly** - Use the same standards regardless of which model produced the answer

## Evaluation Format:
Provide your evaluation in the following format:

TECHNICAL CORRECTNESS: [CORRECT/PARTIALLY CORRECT/INCORRECT]
- CORRECT: The solution is completely accurate and addresses ALL technical aspects
- PARTIALLY CORRECT: The core solution works but has minor technical issues OR addresses only SOME aspects of the problem
- INCORRECT: The solution has significant errors, misconceptions, or would fail if implemented

ALIGNMENT SCORE: X/Y CONDITIONS MET (Z%)

CONDITION 1: [TRUE/FALSE] <brief description of condition>
CONDITION 2: [TRUE/FALSE] <brief description of condition>
...and so on for each condition

VERBOSITY ASSESSMENT: [CONCISE/APPROPRIATE/VERBOSE]
- CONCISE: The answer lacks some potentially helpful context or details
- APPROPRIATE: The answer contains just the right amount of information
- VERBOSE: The answer contains unnecessary information beyond what the user requested

VERDICT: [CORRECT/PARTIALLY CORRECT/INCORRECT] 
You must provide exactly one of these three verdicts based ONLY on technical correctness AND alignment (NOT verbosity):
- CORRECT: The answer is technically correct with no significant errors AND meets ALL user conditions AND addresses ALL aspects of the problem shown in the reference
- PARTIALLY CORRECT: The answer meets SOME but not ALL conditions, OR addresses only SOME aspects of the multi-part problem, OR has minor technical issues
- INCORRECT: The answer has significant technical flaws OR fails to meet ANY conditions OR Docker validation failed

KEY ISSUES: List ALL issues with the maintainer's answer, including:
- Technical inaccuracies (even minor ones)
- Missing aspects from the reference solution
- Unaddressed user conditions
- Any omissions compared to the complete solution

REASONING: Detailed explanation of your verdict, addressing:
1. Technical correctness of each part of the solution
2. Alignment with ALL user conditions
3. Comparison with reference solution - what was included vs. what was missed
4. Why the verdict is appropriate given the above analysis

Be thorough and consistent in your technical assessment. Apply the same standards to all answers regardless of which model produced them.
"""
    
    def _get_default_docker_prompt(self) -> str:
        """Get default Docker-specific prompt."""
        return """# Docker-Specific System Prompt Extension

Since this issue involves Docker, pay special attention to:
- Dockerfile syntax and best practices
- Container build and runtime issues
- Environment variables and configuration
- Networking and service dependencies
- Volume mounting and file permissions
- Image layers and caching

When suggesting changes to a Dockerfile, explain the reasoning behind each modification.

This is a Docker-related issue. You can:
1. Explore the repository
2. Suggest modifications to the Dockerfile
3. Provide Docker commands to solve the issue
4. Create or modify additional files needed for Docker builds

Format file creation/modifications as:
CREATE_FILE[filename]:
file content
END_FILE

Format Dockerfile modifications as:
MODIFY_DOCKERFILE:
# modified content
END_DOCKERFILE
"""
    
    def _get_default_exploration_prompt(self) -> str:
        """Get default exploration system prompt."""
        return """# Repository Exploration System Prompt

You are helping to explore a repository to understand a code issue.

Use your available tools to explore the codebase, read files, search for patterns, and gather information needed to answer the user's question.

When you have enough information, provide your comprehensive answer to the user's question.
"""
