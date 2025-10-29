"""Complete CAB workflow - combines generation and evaluation."""

import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

from ..core.models import (
    IssueData,
    CABResult,
    GenerationResult,
    EvaluationResult
)
from ..core.config import CABConfig
from ..core.exceptions import CABEvaluationError
from ..utils.data_processor import DataProcessor
from .generation_workflow import GenerationWorkflow
from .evaluation_workflow import EvaluationWorkflow

logger = logging.getLogger(__name__)


class CABWorkflow:
    """Complete CAB workflow that combines generation and evaluation."""
    
    def __init__(self, config: Optional[CABConfig] = None):
        """Initialize CAB workflow.
        
        Args:
            config: CAB configuration
        """
        self.config = config or CABConfig()
        self.generation_workflow = GenerationWorkflow(self.config)
        self.evaluation_workflow = EvaluationWorkflow(self.config)
        self.data_processor = DataProcessor()
        
    async def run_complete_evaluation(
        self,
        issue_data: IssueData,
        agent_model_mapping: Optional[Dict[str, str]] = None
    ) -> CABResult:
        """Run complete CAB evaluation workflow.
        
        Args:
            issue_data: Issue data to process
            agent_model_mapping: Optional mapping of agent types to model names
                                Example: {"maintainer": "sonnet37", "user": "haiku", "judge": "sonnet"}
            
        Returns:
            CABResult with complete evaluation results
        """
        logger.info(f"Starting complete CAB evaluation for issue: {issue_data.id}")
        
        # Validate issue data
        if not self.data_processor.validate_issue_data(issue_data):
            raise CABEvaluationError(f"Invalid issue data for issue: {issue_data.id}")
        
        # Log issue summary
        issue_summary = self.data_processor.create_issue_summary(issue_data)
        logger.info(f"Processing issue: {issue_summary}")
        
        try:
            # Step 1: Run generation workflow
            logger.info("=== STARTING GENERATION WORKFLOW ===")
            generation_result = await self.generation_workflow.run_generation(
                issue_data, agent_model_mapping
            )
            logger.info("=== GENERATION WORKFLOW COMPLETE ===")
            
            # Step 2: Run evaluation workflow
            logger.info("=== STARTING EVALUATION WORKFLOW ===")
            evaluation_result = await self.evaluation_workflow.run_evaluation(
                generation_result, agent_model_mapping
            )
            logger.info("=== EVALUATION WORKFLOW COMPLETE ===")
            
            # Create processing metadata
            processing_metadata = {
                'workflow_version': '1.0.0',
                'config_used': self.config.to_dict(),
                'agent_model_mapping': agent_model_mapping or {},
                'issue_summary': issue_summary,
                'processing_time': datetime.now().isoformat(),
                'docker_validation_performed': issue_data.dockerfile is not None,
                'repository_type': self.data_processor.extract_repository_type(issue_data)
            }
            
            # Create final CAB result
            result = CABResult(
                issue_id=issue_data.id,
                question_title=issue_data.first_question.title,
                question_body=issue_data.first_question.body,
                generation_result=generation_result,
                evaluation_result=evaluation_result,
                processing_metadata=processing_metadata
            )
            
            # Log final summary
            logger.info(f"=== CAB EVALUATION COMPLETE FOR ISSUE {issue_data.id} ===")
            logger.info(f"User satisfied: {generation_result.user_satisfied}")
            logger.info(f"Final verdict: {evaluation_result.verdict.value}")
            logger.info(f"Conversation rounds: {generation_result.total_conversation_rounds}")
            logger.info(f"Total LLM calls: {sum(evaluation_result.llm_calls.values())}")
            
            if evaluation_result.alignment_score:
                logger.info(
                    f"Alignment score: {evaluation_result.alignment_score.satisfied}/"
                    f"{evaluation_result.alignment_score.total} conditions met "
                    f"({evaluation_result.alignment_score.percentage:.1f}%)"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in CAB evaluation workflow: {e}")
            raise CABEvaluationError(f"CAB workflow failed for issue {issue_data.id}: {str(e)}")
    
    async def process_dataset(
        self,
        dataset_path: str,
        target_language: Optional[str] = None,
        output_dir: str = "results",
        agent_model_mapping: Optional[Dict[str, str]] = None,
        batch_size: Optional[int] = None,
        resume_processing: bool = True
    ) -> Dict[str, Any]:
        """Process a complete dataset through CAB evaluation.
        
        Args:
            dataset_path: Path to dataset file
            target_language: Optional language filter
            output_dir: Output directory for results
            agent_model_mapping: Optional agent to model mapping
            batch_size: Batch size for processing (defaults to config)
            resume_processing: Whether to resume from previous processing
            
        Returns:
            Processing summary with statistics
        """
        batch_size = batch_size or self.config.workflow.batch_size
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Starting dataset processing: {dataset_path}")
        if target_language:
            logger.info(f"Filtering by language: {target_language}")
        
        # Load and filter dataset
        if target_language:
            issues = self.data_processor.filter_by_language(dataset_path, target_language)
        else:
            # Load all issues from dataset
            raw_data = self.data_processor.load_jsonl_data([dataset_path])
            issues = [self.data_processor.load_issue_data_from_dict(item) for item in raw_data]
        
        logger.info(f"Loaded {len(issues)} issues for processing")
        
        # Get already processed issues if resuming
        processed_issues = set()
        if resume_processing:
            processed_issues = self.data_processor.get_processed_issues(output_dir)
            logger.info(f"Found {len(processed_issues)} already processed issues")
        
        # Filter out already processed issues
        issues_to_process = [
            issue for issue in issues 
            if issue.id not in processed_issues
        ]
        
        logger.info(f"Processing {len(issues_to_process)} new issues")
        
        # Process issues in batches
        results = []
        batch_num = 1
        current_batch = []
        
        for i, issue_data in enumerate(issues_to_process):
            logger.info(f"Processing issue {i+1}/{len(issues_to_process)}: {issue_data.first_question.title}")
            
            try:
                # Run complete CAB evaluation
                cab_result = await self.run_complete_evaluation(issue_data, agent_model_mapping)
                
                # Convert to dictionary for saving
                result_dict = self._cab_result_to_dict(cab_result)
                current_batch.append(result_dict)
                
                logger.info(f"Issue {issue_data.id} processed successfully")
                
            except Exception as e:
                logger.error(f"Error processing issue {issue_data.id}: {e}")
                # Create error result
                error_result = {
                    'issue_id': issue_data.id,
                    'question_title': issue_data.first_question.title,
                    'question_body': issue_data.first_question.body,
                    'error': str(e),
                    'final_verdict': 'ERROR',
                    'user_satisfied': False,
                    'processing_metadata': {
                        'timestamp': timestamp,
                        'error_occurred': True,
                        'error_message': str(e)
                    }
                }
                current_batch.append(error_result)
            
            # Save batch when it reaches batch_size
            if len(current_batch) >= batch_size:
                self.data_processor.save_batch_results(
                    current_batch, batch_num, output_dir, timestamp
                )
                logger.info(f"Saved batch {batch_num} with {len(current_batch)} results")
                current_batch = []
                batch_num += 1
        
        # Save any remaining results
        if current_batch:
            self.data_processor.save_batch_results(
                current_batch, batch_num, output_dir, timestamp
            )
            logger.info(f"Saved final batch {batch_num} with {len(current_batch)} results")
        
        # Create processing summary
        summary = {
            'dataset_path': dataset_path,
            'target_language': target_language,
            'output_dir': output_dir,
            'timestamp': timestamp,
            'total_issues_in_dataset': len(issues),
            'already_processed': len(processed_issues),
            'newly_processed': len(issues_to_process),
            'total_batches': batch_num,
            'agent_model_mapping': agent_model_mapping or {},
            'processing_complete': True
        }
        
        logger.info(f"Dataset processing complete: {summary}")
        return summary
    
    def _cab_result_to_dict(self, cab_result: CABResult) -> Dict[str, Any]:
        """Convert CABResult to dictionary for serialization.
        
        Args:
            cab_result: CAB result object
            
        Returns:
            Dictionary representation
        """
        # Format conversation history
        formatted_conversation = self.data_processor.format_conversation_for_export(
            cab_result.generation_result.conversation_history
        )
        
        return {
            'issue_id': cab_result.issue_id,
            'question_title': cab_result.question_title,
            'question_body': cab_result.question_body,
            'timestamp': cab_result.timestamp.isoformat() if hasattr(cab_result.timestamp, 'isoformat') else str(cab_result.timestamp),
            
            # Generation results
            'initial_response': formatted_conversation[1]['content'] if len(formatted_conversation) > 1 else '',
            'final_response': self.evaluation_workflow._extract_final_maintainer_answer(formatted_conversation),
            'total_conversation_rounds': cab_result.generation_result.total_conversation_rounds,
            'original_conversation_length': cab_result.generation_result.original_comment_count,
            'user_satisfied': cab_result.generation_result.user_satisfied,
            'satisfaction_status': cab_result.generation_result.satisfaction_status.value,
            'satisfaction_reason': cab_result.generation_result.satisfaction_reason,
            'conversation_history': formatted_conversation,
            'exploration_log': cab_result.generation_result.exploration_log,
            'exploration_history': cab_result.generation_result.exploration_history,
            
            # Evaluation results
            'judgment': cab_result.evaluation_result.judgment,
            'final_verdict': cab_result.evaluation_result.verdict.value,
            'key_issues': cab_result.evaluation_result.key_issues,
            'llm_calls': cab_result.evaluation_result.llm_calls,
            
            # Alignment scores
            'final_alignment_score': (
                {
                    'satisfied': cab_result.evaluation_result.alignment_score.satisfied,
                    'total': cab_result.evaluation_result.alignment_score.total,
                    'percentage': cab_result.evaluation_result.alignment_score.percentage,
                    'conditions': [
                        {
                            'number': cond.number,
                            'satisfied': cond.satisfied,
                            'description': cond.description
                        }
                        for cond in cab_result.evaluation_result.alignment_score.conditions
                    ]
                } if cab_result.evaluation_result.alignment_score else None
            ),
            
            # Docker validation results
            'docker_validation': (
                {
                    'success': cab_result.evaluation_result.docker_results.success,
                    'logs': cab_result.evaluation_result.docker_results.logs,
                    'test_commands': cab_result.evaluation_result.docker_results.test_commands,
                    'error': cab_result.evaluation_result.docker_results.error
                } if cab_result.evaluation_result.docker_results else None
            ),
            
            # Prompt cache metrics
            'prompt_cache': {
                **cab_result.generation_result.prompt_cache,
                **cab_result.evaluation_result.prompt_cache
            },
            
            # Processing metadata
            'processing_metadata': cab_result.processing_metadata
        }
