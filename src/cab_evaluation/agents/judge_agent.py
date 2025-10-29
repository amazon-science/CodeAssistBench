"""Judge agent implementation."""

import json
import os
import re
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from ..core.models import (
    IssueData, AlignmentScore, AlignmentCondition, VerdictType, 
    JudgeConfig, JudgeIteration, IterativeEvaluationResult, 
    RepositoryExploration, RepositoryFile, ConversationAnalysis,
    ConversationMessage
)
from ..prompts.constants import ResponseFormats
from .strands_agent import StrandsAgent

import logging

logger = logging.getLogger(__name__)


class JudgeAgent(StrandsAgent):
    """Agent that judges maintainer responses for correctness and alignment."""
    
    def __init__(self, model_name: str = "sonnet37", judge_config: Optional[JudgeConfig] = None, **kwargs):
        """Initialize judge agent.
        
        Args:
            model_name: Model to use for judge evaluations (defaults to sonnet37)
            judge_config: Configuration for judge behavior
            **kwargs: Additional arguments for Strands agent
        """
        # Initialize with judge-specific settings (enable read-only tools for repository exploration)
        super().__init__(model_name=model_name, read_only=True, **kwargs)
        self.agent_type = "judge"  # Override agent_type from StrandsAgent
        self.judge_config = judge_config or JudgeConfig()
        
        # Track iterations
        self.current_iteration = 0
        self.confidence_scores = []
        self.iteration_findings = {}
    
    def get_system_prompt(self, **kwargs) -> str:
        """Get judge system prompt.
        
        Args:
            **kwargs: Additional context
            
        Returns:
            System prompt string
        """
        base_prompt = self.prompt_manager.get_prompt("judge/system_prompt")
        
        # Add Docker-specific guidance if needed
        docker_context = ""
        if kwargs.get('has_docker_validation'):
            docker_context = """
            IMPORTANT: For Docker-related issues, a solution is ONLY considered correct if:
            1. The maintainer's explanation is technically sound AND
            2. The Docker build and test process actually succeeds (check the DOCKER VALIDATION RESULTS)
            
            If the Docker validation shows "Success: False", then the maintainer's answer CANNOT be considered correct,
            regardless of how good the explanation seems. Docker build success is mandatory for Docker issues.
            """
        
        return f"{base_prompt}{docker_context}"
    
    async def generate_response(
        self,
        user_prompt: str,
        system_prompt: str,
        issue_id: str = "unknown",
        **kwargs
    ) -> str:
        """Generate judge evaluation using Strands framework.
        
        Args:
            user_prompt: User input
            system_prompt: System prompt
            issue_id: Issue ID for tracking
            **kwargs: Additional arguments
            
        Returns:
            Generated evaluation with enhanced tool capabilities
        """
        # Use Strands framework for enhanced capabilities
        return await super().generate_response(user_prompt, system_prompt, issue_id, **kwargs)
    
    async def judge_maintainer_answer(
        self,
        issue_data: IssueData,
        maintainer_answer: str,
        docker_results: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, VerdictType, List[str], Optional[AlignmentScore]]:
        """Judge the maintainer's answer for correctness and alignment.
        
        Args:
            issue_data: Issue data with original conversation
            maintainer_answer: Maintainer's answer to evaluate
            docker_results: Docker validation results if available
            
        Returns:
            Tuple of (judgment, verdict, key_issues, alignment_score)
        """
        self.logger.info("Activating judge agent to evaluate maintainer's answer")
        
        # Format the original conversation for evaluation
        conversation = f"Title: {issue_data.first_question.title}\n\nQuestion: {issue_data.first_question.body}\n\n"
        conversation += "--- Comments from maintainers and users ---\n"
        
        for i, comment in enumerate(issue_data.comments):
            conversation += f"[{comment.user}]:\n{comment.body}\n\n"
        
        # Include Docker results if available
        docker_info = ""
        if docker_results:
            success_status = docker_results.get('success', False)
            docker_info = f"""
            MAINTAINER ANSWER VALIDATION RESULTS via DOCKER:
            Status: {success_status}
            Logs:
            {docker_results.get('logs', 'No logs available')}
            """
        
        # Create system prompt
        system_prompt = self.get_system_prompt(has_docker_validation=bool(docker_results))
        
        # Create judge user prompt
        user_prompt = f"""
        USER'S QUESTION AND REFERENCE CONVERSATION:
        {conversation}

        IMPORTANT: The reference conversation above contains the correct/accepted solution provided by the original maintainer(s). Use this as the authoritative baseline to judge the technical correctness of the maintainer's answer. The maintainer's answer should be consistent with the approaches, suggestions, and code provided in this reference conversation.

        USER SATISFACTION CONDITIONS:
        {json.dumps(issue_data.user_satisfaction_condition, indent=2)}

        MAINTAINER'S ANSWER TO EVALUATE:
        {maintainer_answer}

        {docker_info}

        Based on the above information:
        1. Compare the maintainer's answer against the reference conversation to assess technical accuracy
        2. Evaluate the TECHNICAL CORRECTNESS of the maintainer's answer
        3. Determine how well the answer satisfies each user condition
        4. Assess the verbosity of the answer (whether it contains unnecessary information)
        5. Provide your final VERDICT considering technical correctness and alignment (but NOT verbosity)

        For each condition, determine if it is TRUE (fully satisfied) or FALSE (not satisfied).

        IMPORTANT GUIDELINES:
        - Consider the original maintainer's solution in the reference conversation as a strong reference point
        - Be rigorous but fair in your technical assessment
        - For PARTIALLY CORRECT technical rating, the core solution must work but might have minor issues
        - For CORRECT technical rating, the solution must match the intent and approach of the reference solution
        - If this is a Docker-related issue and the Docker validation shows "Success: False", the solution is automatically incorrect
        - The verbosity assessment should not affect your final verdict, but should be noted separately
        """
        
        # Get judge's evaluation using Strands framework
        self.logger.info("Requesting evaluation from judge agent")
        evaluation = await super().generate_response(user_prompt, system_prompt, issue_data.id)
        self.logger.info(f"Judge evaluation complete ({len(evaluation)} chars)")
        
        # Parse the evaluation
        verdict, key_issues, alignment_score = self._parse_evaluation(evaluation, docker_results)
        
        return evaluation, verdict, key_issues, alignment_score
    
    def _parse_evaluation(
        self,
        evaluation: str,
        docker_results: Optional[Dict[str, Any]] = None
    ) -> Tuple[VerdictType, List[str], Optional[AlignmentScore]]:
        """Parse judge evaluation to extract structured results.
        
        Args:
            evaluation: Judge evaluation text
            docker_results: Docker validation results
            
        Returns:
            Tuple of (verdict, key_issues, alignment_score)
        """
        # Extract technical correctness
        technical_correctness = "UNKNOWN"
        if "TECHNICAL CORRECTNESS:" in evaluation:
            tech_section = evaluation.split("TECHNICAL CORRECTNESS:", 1)[1].strip()
            tech_line = tech_section.split("\n", 1)[0].strip()
            
            if "INCORRECT" in tech_line.upper():
                technical_correctness = "INCORRECT"
            elif "PARTIALLY" in tech_line.upper():
                technical_correctness = "PARTIALLY CORRECT"
            elif "CORRECT" in tech_line.upper() and "PARTIALLY" not in tech_line.upper():
                technical_correctness = "CORRECT"
        
        # Extract verdict - handle multiple formats
        verdict = VerdictType.UNKNOWN
        verdict_patterns = [
            "VERDICT:",
            "## 5. VERDICT", 
            "5. VERDICT",
            "FINAL VERDICT:",
            "**VERDICT**"
        ]
        
        for pattern in verdict_patterns:
            if pattern in evaluation:
                verdict_section = evaluation.split(pattern, 1)[1].strip()
                # Get the next few lines to catch markdown formatting
                verdict_lines = verdict_section.split("\n")[:3]  # Check first 3 lines
                verdict_text = " ".join(verdict_lines).upper()
                
                # Remove markdown formatting
                verdict_text = verdict_text.replace("**", "").replace("*", "")
                
                if "INCORRECT" in verdict_text and "PARTIALLY" not in verdict_text:
                    verdict = VerdictType.INCORRECT
                    break
                elif "PARTIALLY CORRECT" in verdict_text or "PARTIALLY_CORRECT" in verdict_text:
                    verdict = VerdictType.PARTIALLY_CORRECT
                    break
                elif "CORRECT" in verdict_text and "PARTIALLY" not in verdict_text and "INCORRECT" not in verdict_text:
                    verdict = VerdictType.CORRECT
                    break
        
        # Extract key issues - handle multiple formats
        key_issues = []
        key_issues_patterns = [
            "KEY ISSUES:",
            "## 3. KEY ISSUES",
            "3. KEY ISSUES",
            "**KEY ISSUES**"
        ]
        
        for pattern in key_issues_patterns:
            if pattern in evaluation:
                key_issues_section = evaluation.split(pattern, 1)[1]
                
                # Get the text up to the next major section if there is one
                end_patterns = ["REASONING:", "## 4.", "4. CONFIDENCE", "## 5.", "5. VERDICT"]
                for end_pattern in end_patterns:
                    if end_pattern in key_issues_section:
                        key_issues_section = key_issues_section.split(end_pattern, 1)[0]
                        break
                        
                # Split by newlines and clean up
                for line in key_issues_section.strip().split("\n"):
                    clean_line = line.strip()
                    if clean_line and not clean_line.startswith(pattern.replace(":", "")):
                        # Remove bullet points and markdown
                        clean_line = re.sub(r'^[-*\s]*', '', clean_line)
                        clean_line = clean_line.replace("**", "").replace("*", "")
                        if clean_line and len(clean_line) > 5:
                            key_issues.append(clean_line)
                break
        
        # Debug logging
        self.logger.debug(f"Parsed verdict: {verdict.value}")
        self.logger.debug(f"Parsed key issues count: {len(key_issues)}")
        if key_issues:
            self.logger.debug(f"First key issue: {key_issues[0][:100]}...")
        
        # Extract alignment score
        alignment_score = self._parse_alignment_score(evaluation)
        if alignment_score:
            alignment_score.technical_correctness = technical_correctness
            # Extract verbosity if present
            if "VERBOSITY ASSESSMENT:" in evaluation:
                verbosity_section = evaluation.split("VERBOSITY ASSESSMENT:", 1)[1].strip()
                verbosity_line = verbosity_section.split("\n", 1)[0].strip()
                
                if "CONCISE" in verbosity_line.upper():
                    alignment_score.verbosity = "CONCISE"
                elif "APPROPRIATE" in verbosity_line.upper():
                    alignment_score.verbosity = "APPROPRIATE"
                elif "VERBOSE" in verbosity_line.upper():
                    alignment_score.verbosity = "VERBOSE"
        
        # If Docker build failed, ensure "INCORRECT" verdict regardless of other factors
        if docker_results is not None and not docker_results.get('success', False):
            verdict = VerdictType.INCORRECT
            if alignment_score:
                alignment_score.technical_correctness = "INCORRECT"
            if "Docker validation failed" not in key_issues:
                key_issues.append("Docker validation failed - build or tests did not succeed")
        
        return verdict, key_issues, alignment_score
    
    def _parse_alignment_score(self, evaluation: str) -> Optional[AlignmentScore]:
        """Parse alignment score from evaluation.
        
        Args:
            evaluation: Judge evaluation text
            
        Returns:
            AlignmentScore object or None if parsing fails
        """
        try:
            # First try to find the summary pattern (X/Y CONDITIONS MET) which is most reliable
            summary_match = re.search(r'(\d+)/(\d+)\s+CONDITIONS\s+MET\s*\((\d+(?:\.\d+)?)%\)', evaluation, re.IGNORECASE)
            
            if summary_match:
                satisfied = int(summary_match.group(1))
                total = int(summary_match.group(2))
                percentage = float(summary_match.group(3))
                
                # Parse individual conditions for details
                conditions = []
                condition_patterns = [
                    r'CONDITION\s+(\d+):\s+\[(\w+)\]\s+(.+)',                    # CONDITION 1: [TRUE] description
                    r'(\d+)\.\s*"([^"]+)"\s*-\s*\*\*(TRUE|FALSE)\*\*',           # 1. "description" - **TRUE**
                    r'-\s*\*\*Condition\s+(\d+)\*\*:\s*([^-]+)\s*-\s*\*\*(TRUE|FALSE)\*\*'  # **Condition 1**: desc - **TRUE**
                ]
                
                lines = evaluation.split('\n')
                for line in lines:
                    for pattern in condition_patterns:
                        match = re.search(pattern, line.strip(), re.IGNORECASE)
                        if match:
                            if len(match.groups()) == 3 and match.group(2) in ['TRUE', 'FALSE']:
                                # CONDITION 1: [TRUE] format
                                condition_num, status, description = match.groups()
                            else:
                                # Other formats
                                condition_num, description, status = match.groups()
                            
                            conditions.append(AlignmentCondition(
                                number=int(condition_num),
                                satisfied=status.upper() == "TRUE",
                                description=description.strip()
                            ))
                            break
                
                # If no detailed conditions found, create generic ones based on summary
                if not conditions:
                    for i in range(total):
                        conditions.append(AlignmentCondition(
                            number=i + 1,
                            satisfied=i < satisfied,
                            description=f"Condition {i + 1}"
                        ))
                
                return AlignmentScore(
                    satisfied=satisfied,
                    total=total,
                    percentage=percentage,
                    conditions=conditions
                )
            
            self.logger.debug("No alignment score summary pattern found")
            return None
        
        except Exception as e:
            self.logger.error(f"Error parsing alignment score: {e}")
            return None

    async def judge_maintainer_answer_iterative(
        self,
        issue_data: IssueData,
        maintainer_answer: str,
        repository_path: str,
        full_conversation_history: List[ConversationMessage],
        docker_results: Optional[Dict[str, Any]] = None
    ) -> IterativeEvaluationResult:
        """Judge the maintainer's answer using iterative refinement.
        
        Args:
            issue_data: Issue data with original conversation
            maintainer_answer: Maintainer's answer to evaluate
            repository_path: Path to repository for exploration
            full_conversation_history: Complete conversation history
            docker_results: Docker validation results if available
            
        Returns:
            IterativeEvaluationResult with all iterations and final judgment
        """
        start_time = time.time()
        self.logger.info(f"🔄 Starting iterative judge evaluation with max {self.judge_config.max_iterations} iterations")
        
        result = IterativeEvaluationResult()
        
        # CRITICAL FIX: Initialize single Strands agent for reuse across all operations
        # Create a stable system prompt that will be reused to maintain cache context
        base_system_prompt = self.get_system_prompt(has_docker_validation=bool(docker_results))
        
        # Ensure Strands agent is created once and reused
        if self._strands_agent is None:
            self.logger.info("🔄 Creating persistent Strands agent for iterative evaluation...")
            self._strands_agent = self._build_strands_agent(base_system_prompt)
        
        # Phase 1: Repository exploration (if enabled) - Use same agent instance
        if self.judge_config.enable_repository_exploration:
            self.logger.info("📁 Exploring repository context...")
            result.repository_exploration = await self._explore_repository_context_cached(repository_path, issue_data)
        
        # Phase 2: Conversation analysis (if enabled) - Use same agent instance  
        if self.judge_config.enable_conversation_analysis:
            self.logger.info("💬 Analyzing conversation history...")
            result.conversation_analysis = await self._analyze_conversation_cached(full_conversation_history)
        
        # Phase 3: Iterative evaluation - Use same agent instance
        self.current_iteration = 0
        self.confidence_scores = []
        self.iteration_findings = {}
        
        for iteration in range(self.judge_config.max_iterations):
            iteration_start = time.time()
            self.current_iteration = iteration + 1
            
            self.logger.info(f"🎯 Judge iteration {self.current_iteration}/{self.judge_config.max_iterations}")
            
            # Perform iteration using persistent agent
            iteration_result = await self._perform_judge_iteration_cached(
                issue_data, maintainer_answer, docker_results, result, iteration
            )
            
            result.iterations.append(iteration_result)
            result.confidence_progression.append(iteration_result.confidence_score)
            
            # Check early stopping conditions
            if self.judge_config.early_stopping_enabled and self.should_continue_iteration(
                iteration, iteration_result.confidence_score, self.iteration_findings
            ):
                self.logger.info(f"✅ Early stopping at iteration {self.current_iteration} (confidence: {iteration_result.confidence_score:.2f})")
                result.stopped_early = True
                result.early_stopping_reason = f"High confidence ({iteration_result.confidence_score:.2f}) reached"
                break
            
            # Check minimum iterations
            if iteration + 1 >= self.judge_config.min_iterations and iteration_result.confidence_score >= self.judge_config.confidence_threshold:
                self.logger.info(f"✅ Confidence threshold met at iteration {self.current_iteration}")
                result.stopped_early = True
                result.early_stopping_reason = f"Confidence threshold ({self.judge_config.confidence_threshold}) met"
                break
        
        # Compile final results
        if result.iterations:
            final_iteration = result.iterations[-1]
            result.final_judgment = final_iteration.reasoning
            result.final_verdict = final_iteration.verdict
            result.final_alignment_score = final_iteration.alignment_score
            result.final_key_issues = final_iteration.key_issues
        
        result.total_evaluation_time_seconds = time.time() - start_time
        
        # Aggregate token usage
        for iteration in result.iterations:
            for key, value in iteration.token_usage.items():
                result.total_token_usage[key] = result.total_token_usage.get(key, 0) + value
        
        self.logger.info(f"🏁 Iterative evaluation complete in {result.total_evaluation_time_seconds:.2f}s")
        self.logger.info(f"📊 Final verdict: {result.final_verdict.value}")
        if result.final_alignment_score:
            self.logger.info(f"📊 Final alignment: {result.final_alignment_score.satisfied}/{result.final_alignment_score.total} conditions ({result.final_alignment_score.percentage:.1f}%)")
        
        return result




    def should_continue_iteration(
        self, 
        iteration: int, 
        current_confidence: float, 
        findings: Dict[str, Any]
    ) -> bool:
        """Determine if iterations should continue.
        
        Args:
            iteration: Current iteration number (0-based)
            current_confidence: Confidence score of current iteration
            findings: Accumulated findings
            
        Returns:
            False if should continue, True if should stop early
        """
        # Always do minimum iterations
        if iteration + 1 < self.judge_config.min_iterations:
            return False
            
        # Stop if high confidence reached
        if current_confidence >= self.judge_config.confidence_threshold:
            return True
            
        # Stop if confidence is increasing consistently
        if len(self.confidence_scores) >= 2:
            recent_scores = self.confidence_scores[-2:]
            if all(recent_scores[i] <= recent_scores[i+1] for i in range(len(recent_scores)-1)):
                if current_confidence >= 0.8:  # High enough confidence with improving trend
                    return True
        
        return False

    def _extract_confidence_score(self, response: str) -> float:
        """Extract confidence score from response.
        
        Args:
            response: Judge response text
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        try:
            # Look for confidence patterns
            confidence_patterns = [
                r"CONFIDENCE[:\s]+([0-9]*\.?[0-9]+)",
                r"confidence[:\s]+([0-9]*\.?[0-9]+)",
                r"CONFIDENCE SCORE[:\s]+([0-9]*\.?[0-9]+)",
            ]
            
            for pattern in confidence_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    # Normalize to 0-1 range
                    if score > 1.0:
                        score = score / 100.0  # Assume percentage
                    return min(1.0, max(0.0, score))
            
            # Default confidence based on verdict
            if "CORRECT" in response.upper() and "PARTIALLY" not in response.upper():
                return 0.9
            elif "PARTIALLY CORRECT" in response.upper():
                return 0.6
            elif "INCORRECT" in response.upper():
                return 0.8
            else:
                return 0.5
                
        except Exception as e:
            self.logger.warning(f"Could not extract confidence score: {e}")
            return 0.5

    def _extract_files_examined(self, response: str) -> List[str]:
        """Extract files examined from response.
        
        Args:
            response: Judge response text
            
        Returns:
            List of file paths mentioned
        """
        # Simple file path extraction
        file_patterns = [
            r'`([^`]+\.[a-zA-Z0-9]+)`',  # Files in backticks
            r'"([^"]+\.[a-zA-Z0-9]+)"',  # Files in quotes
            r"'([^']+\.[a-zA-Z0-9]+)'",  # Files in single quotes
        ]
        
        files = []
        for pattern in file_patterns:
            matches = re.findall(pattern, response)
            files.extend(matches)
        
        return list(set(files))  # Remove duplicates

    def _extract_new_findings(self, response: str, cumulative_results: IterativeEvaluationResult) -> List[str]:
        """Extract new findings from current iteration.
        
        Args:
            response: Current iteration response
            cumulative_results: Previous iteration results
            
        Returns:
            List of new findings
        """
        # Extract key points from current response
        current_findings = []
        
        # Look for bullet points or numbered lists
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('- ') or line.startswith('* ') or re.match(r'^\d+\.', line):
                finding = re.sub(r'^[-*\d\.]\s*', '', line).strip()
                if finding and len(finding) > 10:  # Meaningful findings
                    current_findings.append(finding)
        
        # Filter out findings from previous iterations
        previous_findings = set()
        for iteration in cumulative_results.iterations:
            previous_findings.update(iteration.key_issues)
            previous_findings.update(iteration.new_findings)
        
        new_findings = []
        for finding in current_findings:
            if not any(prev in finding or finding in prev for prev in previous_findings):
                new_findings.append(finding)
        
        return new_findings

    async def _explore_repository_context_cached(
        self, 
        repository_path: str, 
        issue_data: IssueData
    ) -> RepositoryExploration:
        """Explore repository context using cached Strands agent.
        
        Args:
            repository_path: Path to repository
            issue_data: Issue data for context
            
        Returns:
            RepositoryExploration with relevant files and structure
        """
        start_time = time.time()
        exploration = RepositoryExploration()
        
        try:
            # Use the persistent Strands agent directly for caching benefits
            exploration_prompt = f"""
            I need to explore a repository to understand the codebase context for evaluating a maintainer's answer.

            Issue Title: {issue_data.first_question.title}
            Issue Description: {issue_data.first_question.body}
            Programming Language: {issue_data.language}

            Please help me explore the repository at: {repository_path}

            Tasks:
            1. Get the overall structure of the repository
            2. Find files relevant to this issue (based on language, file names, etc.)
            3. Read key configuration files (package.json, requirements.txt, Dockerfile, etc.)
            4. Look for files mentioned in the issue or that seem related to the problem

            Focus on finding up to {self.judge_config.exploration_file_limit} most relevant files.
            Prioritize files that would help understand if the maintainer's solution is correct.
            """
            
            # Call the persistent Strands agent directly to maintain cache
            self.logger.info("🔄 Using cached Strands agent for repository exploration...")
            exploration_response = self._strands_agent(exploration_prompt)
            
            exploration.exploration_log = str(exploration_response)
            exploration.exploration_time_seconds = time.time() - start_time
            
            self.logger.info(f"📁 Cached repository exploration completed in {exploration.exploration_time_seconds:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Cached repository exploration failed: {e}")
            exploration.exploration_log = f"Exploration failed: {str(e)}"
            exploration.exploration_time_seconds = time.time() - start_time
        
        return exploration

    async def _analyze_conversation_cached(
        self, 
        conversation_history: List[ConversationMessage]
    ) -> Dict[str, Any]:
        """Perform conversation analysis using cached Strands agent.
        
        Args:
            conversation_history: Full conversation history
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Format conversation for analysis
            conversation_text = ""
            role_counts = {}
            
            for msg in conversation_history:
                role_counts[msg.role] = role_counts.get(msg.role, 0) + 1
                conversation_text += f"[{msg.role.upper()}]: {msg.content}\n\n"
            
            analysis_prompt = f"""
            Please analyze this conversation history to extract key technical information:

            {conversation_text}

            Provide analysis on:
            1. Technical solutions mentioned
            2. Code patterns and snippets
            3. Dependencies and libraries mentioned  
            4. File references
            5. Evolution of the solution approach
            6. Key technical concepts

            Focus on information that would help judge the correctness of a maintainer's final answer.
            """
            
            # Call the persistent Strands agent directly to maintain cache
            self.logger.info("🔄 Using cached Strands agent for conversation analysis...")
            analysis_response = self._strands_agent(analysis_prompt)
            
            return {
                "total_messages": len(conversation_history),
                "messages_by_role": role_counts,
                "analysis_summary": str(analysis_response),
                "analysis_timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Cached conversation analysis failed: {e}")
            return {
                "total_messages": len(conversation_history),
                "messages_by_role": {},
                "analysis_summary": f"Analysis failed: {str(e)}",
                "error": str(e)
            }

    async def _perform_judge_iteration_cached(
        self,
        issue_data: IssueData,
        maintainer_answer: str,
        docker_results: Optional[Dict[str, Any]],
        cumulative_results: IterativeEvaluationResult,
        iteration_num: int
    ) -> JudgeIteration:
        """Perform a single judge iteration using cached Strands agent.
        
        Args:
            issue_data: Issue data
            maintainer_answer: Answer to evaluate
            docker_results: Docker validation results
            cumulative_results: Results from previous iterations
            iteration_num: Current iteration number
            
        Returns:
            JudgeIteration with results
        """
        iteration_start = time.time()
        
        # Build context from previous iterations
        previous_context = ""
        if cumulative_results.iterations:
            previous_context = "\n--- PREVIOUS ITERATIONS ---\n"
            for i, prev_iter in enumerate(cumulative_results.iterations):
                previous_context += f"Iteration {i+1}: {prev_iter.verdict.value} (confidence: {prev_iter.confidence_score:.2f})\n"
                previous_context += f"Key findings: {', '.join(prev_iter.key_issues)}\n\n"
        
        # Repository context
        repo_context = ""
        if cumulative_results.repository_exploration:
            repo_context = f"\n--- REPOSITORY EXPLORATION ---\n{cumulative_results.repository_exploration.exploration_log}\n"
        
        # Conversation context  
        conv_context = ""
        if cumulative_results.conversation_analysis:
            conv_context = f"\n--- CONVERSATION ANALYSIS ---\n{cumulative_results.conversation_analysis.get('analysis_summary', '')}\n"
        
        # Docker context
        docker_context = ""
        if docker_results:
            docker_context = f"\n--- DOCKER VALIDATION ---\nSuccess: {docker_results.get('success', False)}\nLogs: {docker_results.get('logs', 'No logs')}\n"
        
        # Create iteration-specific prompt
        iteration_prompt = f"""
        This is iteration {iteration_num + 1} of {self.judge_config.max_iterations} for evaluating a maintainer's answer.

        ISSUE DETAILS:
        Title: {issue_data.first_question.title}
        Question: {issue_data.first_question.body}

        MAINTAINER'S ANSWER TO EVALUATE:
        {maintainer_answer}

        USER SATISFACTION CONDITIONS:
        {json.dumps(issue_data.user_satisfaction_condition, indent=2)}

        {previous_context}
        {repo_context}
        {conv_context}
        {docker_context}

        Please provide:
        1. TECHNICAL CORRECTNESS assessment
        2. ALIGNMENT SCORE for each condition (TRUE/FALSE)
        3. KEY ISSUES identified
        4. CONFIDENCE SCORE (0.0-1.0) in your assessment
        5. VERDICT (CORRECT/PARTIALLY_CORRECT/INCORRECT)
        6. REASONING for your judgment

        Focus on new insights in this iteration. If you have repository or conversation context, use it to verify technical claims.
        """
        
        # Use the persistent Strands agent directly to maintain cache context
        self.logger.info(f"🔄 Using cached Strands agent for iteration {iteration_num + 1}...")
        response = self._strands_agent(iteration_prompt)
        
        # Parse results
        verdict, key_issues, alignment_score = self._parse_evaluation(str(response), docker_results)
        confidence_score = self._extract_confidence_score(str(response))
        
        iteration_result = JudgeIteration(
            iteration_number=iteration_num + 1,
            reasoning=str(response),
            verdict=verdict,
            confidence_score=confidence_score,
            alignment_score=alignment_score,
            key_issues=key_issues,
            files_examined=self._extract_files_examined(str(response)),
            new_findings=self._extract_new_findings(str(response), cumulative_results),
            iteration_time_seconds=time.time() - iteration_start,
            token_usage={"judge": 1}  # Simplified token counting
        )
        
        return iteration_result
