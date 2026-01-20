"""Complete CAB workflow - combines generation and evaluation."""

import os
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
        """Initialize CAB workflow."""
        self.config = config or CABConfig()
        self.generation_workflow = GenerationWorkflow(self.config)
        self.evaluation_workflow = EvaluationWorkflow(self.config)
        self.data_processor = DataProcessor()
    
    def _flush_logger(self, log: logging.Logger):
        """Flush all logger handlers."""
        try:
            for handler in log.handlers:
                if hasattr(handler, 'flush'):
                    handler.flush()
        except Exception as e:
            logger.error(f"Error flushing logger: {e}")
        
    async def run_complete_evaluation(
        self,
        issue_data: IssueData,
        agent_model_mapping: Optional[Dict[str, str]] = None,
        agent_framework_mapping: Optional[Dict[str, str]] = None,
        issue_logger: Optional[logging.Logger] = None,
        enable_ast_tools: bool = True
    ) -> CABResult:
        """Run complete CAB evaluation workflow."""
        log = issue_logger or logger
        
        log.info(f"Starting complete CAB evaluation for issue: {issue_data.id}")
        self._flush_logger(log)
        
        if not self.data_processor.validate_issue_data(issue_data):
            raise CABEvaluationError(f"Invalid issue data for issue: {issue_data.id}")
        
        issue_summary = self.data_processor.create_issue_summary(issue_data)
        log.info(f"Processing issue: {issue_summary}")
        
        try:
            log.info("=== STARTING GENERATION WORKFLOW ===")
            self._flush_logger(log)
            generation_result = await self.generation_workflow.run_generation(
                issue_data, agent_model_mapping, agent_framework_mapping, issue_logger=issue_logger,
                enable_ast_tools=enable_ast_tools
            )
            log.info("=== GENERATION WORKFLOW COMPLETE ===")
            self._flush_logger(log)
            
            log.info("=== STARTING EVALUATION WORKFLOW ===")
            self._flush_logger(log)
            evaluation_result = await self.evaluation_workflow.run_evaluation(
                generation_result, agent_model_mapping, issue_logger=issue_logger
            )
            log.info("=== EVALUATION WORKFLOW COMPLETE ===")
            self._flush_logger(log)
            
            complete_framework_mapping = {
                "maintainer": agent_framework_mapping.get("maintainer", "strands") if agent_framework_mapping else "strands",
                "user": agent_framework_mapping.get("user", "strands") if agent_framework_mapping else "strands",
                "judge": agent_framework_mapping.get("judge", "strands") if agent_framework_mapping else "strands"
            }
            
            processing_metadata = {
                'workflow_version': '1.0.0',
                'agent_model_mapping': agent_model_mapping or {},
                'agent_framework_mapping': complete_framework_mapping,
                'issue_summary': issue_summary,
                'processing_time': datetime.now().isoformat(),
                'docker_validation_performed': issue_data.dockerfile is not None,
                'repository_type': self.data_processor.extract_repository_type(issue_data)
            }
            
            result = CABResult(
                issue_id=issue_data.id,
                question_title=issue_data.first_question.title,
                question_body=issue_data.first_question.body,
                generation_result=generation_result,
                evaluation_result=evaluation_result,
                processing_metadata=processing_metadata
            )
            
            log.info(f"=== CAB EVALUATION COMPLETE FOR ISSUE {issue_data.id} ===")
            log.info(f"User satisfied: {generation_result.user_satisfied}")
            log.info(f"Final verdict: {evaluation_result.verdict.value}")
            log.info(f"Conversation rounds: {generation_result.total_conversation_rounds}")
            log.info(f"Total LLM calls: {sum(evaluation_result.llm_calls.values())}")
            
            if evaluation_result.alignment_score:
                log.info(
                    f"Alignment score: {evaluation_result.alignment_score.satisfied}/"
                    f"{evaluation_result.alignment_score.total} conditions met "
                    f"({evaluation_result.alignment_score.percentage:.1f}%)"
                )
            
            self._flush_logger(log)
            return result
            
        except Exception as e:
            log.error(f"Error in CAB evaluation workflow: {e}", exc_info=True)
            raise CABEvaluationError(f"CAB workflow failed for issue {issue_data.id}: {str(e)}")
    
    async def process_dataset(
        self,
        dataset_path: str,
        target_language: Optional[str] = None,
        output_dir: str = "results",
        agent_model_mapping: Optional[Dict[str, str]] = None,
        agent_framework_mapping: Optional[Dict[str, str]] = None,
        batch_size: Optional[int] = None,
        resume_processing: bool = True,
        enable_ast_tools: bool = True
    ) -> Dict[str, Any]:
        """Process complete dataset through CAB evaluation."""
        batch_size = batch_size or self.config.workflow.batch_size
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Starting dataset processing: {dataset_path}")
        if target_language:
            logger.info(f"Filtering by language: {target_language}")
        
        if target_language:
            issues = self.data_processor.filter_by_language(dataset_path, target_language)
        else:
            raw_data = self.data_processor.load_jsonl_data([dataset_path])
            issues = [self.data_processor.load_issue_data_from_dict(item) for item in raw_data]
        
        logger.info(f"Loaded {len(issues)} issues for processing")
        
        processed_issues = set()
        if resume_processing:
            processed_issues = self.data_processor.get_processed_issues(output_dir)
            logger.info(f"Found {len(processed_issues)} already processed issues")
        
        issues_to_process = [
            issue for issue in issues 
            if issue.id not in processed_issues
        ]
        
        logger.info(f"Processing {len(issues_to_process)} new issues")
        
        log_dir = os.path.join(output_dir, f'logs_{timestamp}')
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f"📁 Log directory created: {log_dir}")
        
        successful_count = 0
        error_count = 0
        
        for i, issue_data in enumerate(issues_to_process):
            logger.info(f"\n{'='*80}")
            logger.info(f"📋 Processing issue {i+1}/{len(issues_to_process)}: {issue_data.first_question.title}")
            logger.info(f"{'='*80}")
            
            issue_logger = None
            
            try:
                issue_logger = self.data_processor.setup_issue_logger(issue_data, log_dir)
                log_filename = self.data_processor.create_issue_log_filename(issue_data)
                logger.info(f"📝 Logging to: {log_filename}")
                
                cab_result = await self.run_complete_evaluation(
                    issue_data, 
                    agent_model_mapping,
                    agent_framework_mapping,
                    issue_logger=issue_logger,
                    enable_ast_tools=enable_ast_tools
                )
                
                result_dict = self._cab_result_to_dict(cab_result)
                
                is_docker = issue_data.dockerfile is not None
                self.data_processor.save_single_result(
                    result_dict, 
                    output_dir, 
                    timestamp,
                    issue_number=i+1,
                    is_docker=is_docker
                )
                
                successful_count += 1
                logger.info(f"✅ Issue {issue_data.id} processed and saved successfully ({successful_count}/{i+1})")
                
                if issue_logger:
                    self.data_processor.flush_issue_logger(issue_logger)
                
            except Exception as e:
                error_count += 1
                logger.error(f"❌ Error processing issue {issue_data.id}: {e}")
                
                if issue_logger:
                    issue_logger.error(f"Fatal error during evaluation: {e}", exc_info=True)
                
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
                        'error_message': str(e),
                        'issue_number': i+1
                    }
                }
                
                is_docker = issue_data.dockerfile is not None
                self.data_processor.save_single_result(
                    error_result,
                    output_dir,
                    timestamp,
                    issue_number=i+1,
                    is_docker=is_docker
                )
                logger.info(f"⚠️  Error result saved for issue {issue_data.id} ({error_count} errors so far)")
                
                if issue_logger:
                    self.data_processor.flush_issue_logger(issue_logger)
            
            finally:
                if issue_logger:
                    self.data_processor.cleanup_issue_logger(issue_logger)
            
            if (i + 1) % 5 == 0 or (i + 1) == len(issues_to_process):
                logger.info(f"\n📊 Progress: {i+1}/{len(issues_to_process)} issues processed | ✅ {successful_count} successful | ❌ {error_count} errors")
        
        summary = {
            'dataset_path': dataset_path,
            'target_language': target_language,
            'output_dir': output_dir,
            'timestamp': timestamp,
            'total_issues_in_dataset': len(issues),
            'already_processed': len(processed_issues),
            'newly_processed': len(issues_to_process),
            'successful_count': successful_count,
            'error_count': error_count,
            'success_rate': f"{(successful_count / len(issues_to_process) * 100):.1f}%" if issues_to_process else "N/A",
            'agent_model_mapping': agent_model_mapping or {},
            'processing_complete': True,
            'save_mode': 'immediate'
        }
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🎉 Dataset processing complete!")
        logger.info(f"{'='*80}")
        logger.info(f"📊 Summary:")
        logger.info(f"   - Total issues in dataset: {len(issues)}")
        logger.info(f"   - Already processed: {len(processed_issues)}")
        logger.info(f"   - Newly processed: {len(issues_to_process)}")
        logger.info(f"   - ✅ Successful: {successful_count}")
        logger.info(f"   - ❌ Errors: {error_count}")
        logger.info(f"   - Success rate: {summary['success_rate']}")
        logger.info(f"   - Output directory: {output_dir}")
        logger.info(f"   - Results saved: Immediately after each issue (no batch waiting)")
        logger.info(f"{'='*80}\n")
        
        return summary
    
    def _cab_result_to_dict(self, cab_result: CABResult) -> Dict[str, Any]:
        """Convert CABResult to dictionary for serialization."""
        formatted_conversation = self.data_processor.format_conversation_for_export(
            cab_result.generation_result.conversation_history
        )
        
        return {
            'issue_id': cab_result.issue_id,
            'question_title': cab_result.question_title,
            'question_body': cab_result.question_body,
            'timestamp': cab_result.timestamp.isoformat() if hasattr(cab_result.timestamp, 'isoformat') else str(cab_result.timestamp),
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
            'judgment': cab_result.evaluation_result.judgment,
            'final_verdict': cab_result.evaluation_result.verdict.value,
            'key_issues': cab_result.evaluation_result.key_issues,
            'llm_calls': cab_result.evaluation_result.llm_calls,
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
            'docker_validation': (
                {
                    'success': cab_result.evaluation_result.docker_results.success,
                    'logs': cab_result.evaluation_result.docker_results.logs,
                    'test_commands': cab_result.evaluation_result.docker_results.test_commands,
                    'error': cab_result.evaluation_result.docker_results.error
                } if cab_result.evaluation_result.docker_results else None
            ),
            'prompt_cache': {
                **cab_result.generation_result.prompt_cache,
                **cab_result.evaluation_result.prompt_cache
            },
            'agent_framework_used': cab_result.generation_result.agent_framework_used,
            'qcli_metadata': cab_result.generation_result.qcli_metadata,
            'kiro_cli_metadata': cab_result.generation_result.kiro_cli_metadata,
            'processing_metadata': cab_result.processing_metadata
        }
