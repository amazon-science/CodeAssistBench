"""Data processing utilities for CAB evaluation."""

import json
import os
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import logging

from ..core.models import IssueData, CommitInfo, Question, Comment
from ..core.exceptions import CABEvaluationError

logger = logging.getLogger(__name__)


class DataProcessor:
    """Processes and manages CAB evaluation datasets."""
    
    def __init__(self, dataset_dir: str = "dataset"):
        """Initialize data processor.
        
        Args:
            dataset_dir: Directory containing datasets
        """
        self.dataset_dir = Path(dataset_dir)
        
    def load_jsonl_data(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Load data from multiple JSONL files.
        
        Args:
            file_paths: List of JSONL file paths
            
        Returns:
            List of loaded data entries
        """
        data = []
        for file_path in file_paths:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        # Extract JSON from line
                        start_idx = line.find('{')
                        end_idx = line.rfind('}') + 1
                        if start_idx != -1 and end_idx > start_idx:
                            json_content = line[start_idx:end_idx]
                            data.append(json.loads(json_content))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num} in {file_path}: {e}")
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num} in {file_path}: {e}")
                        continue
        
        logger.info(f"Loaded {len(data)} entries from {len(file_paths)} files")
        return data
    
    def load_issue_data_from_dict(self, data_dict: Dict[str, Any]) -> IssueData:
        """Convert dictionary to IssueData object.
        
        Args:
            data_dict: Raw data dictionary (supports both nested and flat JSONL formats)
            
        Returns:
            IssueData object
            
        Raises:
            CABEvaluationError: If required fields are missing
        """
        try:
            # Extract commit info - handle both new and legacy formats
            commit_info = self._extract_commit_info(data_dict)
            
            # Extract first question - handle both nested and flat formats
            if "first_question" in data_dict:
                # Nested format
                first_q = data_dict["first_question"]
                first_question = Question(
                    title=first_q.get("title", ""),
                    body=first_q.get("body", ""),
                    user=first_q.get("user", ""),
                    created_at=first_q.get("created_at", "")
                )
            else:
                # Flat JSONL format (common case)
                first_question = Question(
                    title=data_dict.get("title", ""),
                    body=data_dict.get("body", ""),
                    user=data_dict.get("author", ""),  # JSONL uses 'author' not 'user'
                    created_at=data_dict.get("created_at", "")
                )
            
            # Extract comments
            comments = []
            for comment_data in data_dict.get("comments", []):
                comments.append(Comment(
                    user=comment_data.get("user", ""),
                    body=comment_data.get("body", ""),
                    created_at=comment_data.get("created_at", "")
                ))
            
            # Extract issue ID - handle both formats  
            issue_id = (
                data_dict.get("id") or                                    # Nested format
                str(data_dict.get("number", f"issue_{hash(first_question.title)}"))  # Flat JSONL format
            )
            
            # Create IssueData object
            issue_data = IssueData(
                id=issue_id,
                language=data_dict.get("language", "unknown"),
                commit_info=commit_info,
                first_question=first_question,
                comments=comments,
                user_satisfaction_condition=data_dict.get("satisfaction_conditions", data_dict.get("user_satisfaction_condition", [])),
                dockerfile=data_dict.get("dockerfile"),
                extra_files=data_dict.get("extra_files")
            )
            
            return issue_data
            
        except Exception as e:
            raise CABEvaluationError(f"Error converting dictionary to IssueData: {e}")
    
    def _extract_commit_info(self, data_dict: Dict[str, Any]) -> CommitInfo:
        """Extract commit info from data dictionary, handling both formats.
        
        Args:
            data_dict: Raw data dictionary
            
        Returns:
            CommitInfo object
            
        Raises:
            CABEvaluationError: If commit info cannot be extracted
        """
        # Method 1: Complex format with nested commit_info structure
        commit_data = data_dict.get("commit_info", {})
        if isinstance(commit_data, dict) and "latest_commit" in commit_data:
            return CommitInfo(
                repository=commit_data["repository"],
                sha=commit_data["latest_commit"]["sha"],
                message=commit_data["latest_commit"].get("message", ""),
                author=commit_data["latest_commit"].get("author", ""),
                date=commit_data["latest_commit"].get("date", "")
            )
        
        # Method 2: Simple format with commit_id (backward compatibility)
        commit_id = data_dict.get("commit_id")
        if commit_id:
            # Extract repository from URL
            repository_url = data_dict.get("url", "")
            if repository_url:
                # Convert GitHub issue/PR URL to repository URL
                # Example: https://github.com/owner/repo/issues/123 -> https://github.com/owner/repo
                import re
                repo_match = re.match(r'(https://github\.com/[^/]+/[^/]+)', repository_url)
                repository = repo_match.group(1) if repo_match else repository_url
            else:
                repository = "unknown"
            
            return CommitInfo(
                repository=repository,
                sha=commit_id,
                message=data_dict.get("title", ""),  # Use issue title as message fallback
                author=data_dict.get("author", ""),
                date=data_dict.get("created_at", "")
            )
        
        # Method 3: Fallback for missing commit info
        logger.warning("No commit_info or commit_id found, creating default CommitInfo")
        repository_url = data_dict.get("url", "unknown")
        if repository_url and "github.com" in repository_url:
            import re
            repo_match = re.match(r'(https://github\.com/[^/]+/[^/]+)', repository_url)
            repository = repo_match.group(1) if repo_match else repository_url
        else:
            repository = "unknown"
            
        return CommitInfo(
            repository=repository,
            sha=f"unknown_{data_dict.get('number', 'no_id')}",
            message=data_dict.get("title", "Unknown issue"),
            author=data_dict.get("author", "unknown"),
            date=data_dict.get("created_at", "")
        )
    
    def filter_by_language(self, dataset_path: str, target_language: str) -> List[IssueData]:
        """Filter dataset by programming language.
        
        Args:
            dataset_path: Path to dataset file
            target_language: Target programming language
            
        Returns:
            List of filtered IssueData objects
        """
        filtered_issues = []
        
        try:
            with open(dataset_path, 'r', encoding='utf-8', errors='replace') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        # Extract JSON content
                        start_idx = line.find('{')
                        end_idx = line.rfind('}') + 1
                        if start_idx != -1 and end_idx > start_idx:
                            json_content = line[start_idx:end_idx]
                            issue_dict = json.loads(json_content)
                            
                            # Check language match
                            current_language = issue_dict.get('language', '').lower()
                            if current_language == target_language.lower():
                                issue_data = self.load_issue_data_from_dict(issue_dict)
                                filtered_issues.append(issue_data)
                                logger.debug(f"Added issue {issue_data.id} for language {current_language}")
                                
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num}: {e}")
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num}: {e}")
                        continue
            
            logger.info(f"Filtered {len(filtered_issues)} issues for language '{target_language}'")
            return filtered_issues
            
        except Exception as e:
            raise CABEvaluationError(f"Error filtering dataset by language: {e}")
    
    def get_processed_issues(self, output_dir: str) -> Set[str]:
        """Get already processed issue IDs from output directory.
        
        Args:
            output_dir: Output directory to scan
            
        Returns:
            Set of processed issue IDs
        """
        processed_issues = set()
        
        try:
            if not os.path.exists(output_dir):
                return processed_issues
            
            # Look for all JSONL files in output directory
            for filename in os.listdir(output_dir):
                if filename.endswith('.jsonl'):
                    filepath = os.path.join(output_dir, filename)
                    try:
                        with open(filepath, 'r') as f:
                            for line in f:
                                try:
                                    result = json.loads(line)
                                    # Get issue ID
                                    issue_id = result.get('issue_id') or result.get('id')
                                    if issue_id:
                                        processed_issues.add(issue_id)
                                except json.JSONDecodeError:
                                    continue
                    except Exception as e:
                        logger.error(f"Error reading file {filename}: {e}")
                        continue
                        
            logger.info(f"Found {len(processed_issues)} already processed issues")
            return processed_issues
            
        except Exception as e:
            logger.error(f"Error reading output directory: {e}")
            return set()
    
    def save_batch_results(
        self,
        results_batch: List[Dict[str, Any]],
        batch_num: int,
        results_dir: str,
        timestamp: str
    ):
        """Save a batch of results to files.
        
        Args:
            results_batch: Batch of results to save
            batch_num: Batch number
            results_dir: Results directory
            timestamp: Timestamp string
        """
        try:
            # Create results directory
            os.makedirs(results_dir, exist_ok=True)
            
            # Separate Docker and non-Docker results
            docker_results = []
            regular_results = []
            
            for result in results_batch:
                # Add metadata
                result['metadata'] = {
                    'timestamp': timestamp,
                    'batch_number': batch_num,
                    'processing_date': result.get('timestamp', timestamp),
                    'has_docker': 'docker_validation' in result,
                    'total_conversation_rounds': result.get('total_conversation_rounds', 'N/A')
                }
                
                # Ensure conversation_history is serializable
                if 'conversation_history' in result:
                    for message in result['conversation_history']:
                        if 'content' in message and not isinstance(
                            message['content'], (str, int, float, bool, type(None))
                        ):
                            message['content'] = str(message['content'])
                
                # Categorize results
                if 'docker_validation' in result:
                    docker_results.append(result)
                else:
                    regular_results.append(result)
            
            # Save Docker results
            if docker_results:
                docker_file = os.path.join(results_dir, f'docker_responses_{timestamp}_batch_{batch_num}.jsonl')
                with open(docker_file, 'w') as f:
                    for result in docker_results:
                        f.write(json.dumps(result) + '\n')
                logger.info(f"Saved {len(docker_results)} Docker results to batch {batch_num}")
            
            # Save regular results
            if regular_results:
                regular_file = os.path.join(results_dir, f'responses_{timestamp}_batch_{batch_num}.jsonl')
                with open(regular_file, 'w') as f:
                    for result in regular_results:
                        f.write(json.dumps(result) + '\n')
                logger.info(f"Saved {len(regular_results)} regular results to batch {batch_num}")
            
        except Exception as e:
            logger.error(f"Error saving batch {batch_num}: {e}")
            # Try emergency save with simplified data
            try:
                emergency_file = os.path.join(results_dir, f'emergency_save_{timestamp}_batch_{batch_num}.jsonl')
                with open(emergency_file, 'w') as f:
                    for result in results_batch:
                        simplified = {
                            'issue_id': result.get('issue_id', 'unknown'),
                            'question_title': result.get('question_title', ''),
                            'initial_verdict': result.get('initial_verdict', 'UNKNOWN'),
                            'final_verdict': result.get('final_verdict', 'UNKNOWN'),
                            'user_satisfied': result.get('user_satisfied', False)
                        }
                        f.write(json.dumps(simplified) + '\n')
                logger.info(f"Saved emergency backup of results to {emergency_file}")
            except Exception as e2:
                logger.error(f"Emergency save also failed: {e2}")
    
    def create_issue_summary(self, issue_data: IssueData) -> Dict[str, Any]:
        """Create a summary of an issue for logging/reporting.
        
        Args:
            issue_data: Issue data to summarize
            
        Returns:
            Summary dictionary
        """
        return {
            'issue_id': issue_data.id,
            'title': issue_data.first_question.title,
            'language': issue_data.language,
            'repository': issue_data.commit_info.repository,
            'commit_sha': issue_data.commit_info.sha,
            'has_dockerfile': issue_data.dockerfile is not None,
            'comment_count': len(issue_data.comments),
            'satisfaction_conditions_count': len(issue_data.user_satisfaction_condition),
            'has_extra_files': bool(issue_data.extra_files)
        }
    
    def validate_issue_data(self, issue_data: IssueData) -> bool:
        """Validate that issue data has required fields.
        
        Args:
            issue_data: Issue data to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields
            required_checks = [
                (issue_data.id, "Missing issue ID"),
                (issue_data.first_question.title, "Missing question title"),
                (issue_data.first_question.body, "Missing question body"),
                (issue_data.commit_info.repository, "Missing repository URL"),
                (issue_data.commit_info.sha, "Missing commit SHA")
            ]
            
            for value, error_msg in required_checks:
                if not value:
                    logger.error(f"Issue validation failed: {error_msg}")
                    return False
            
            # Validate satisfaction conditions
            if not issue_data.user_satisfaction_condition:
                logger.warning("No user satisfaction conditions found")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating issue data: {e}")
            return False
    
    def extract_repository_type(self, issue_data: IssueData) -> str:
        """Extract repository type from issue data.
        
        Args:
            issue_data: Issue data
            
        Returns:
            Repository type string
        """
        if issue_data.dockerfile:
            return "Docker"
        
        # Could add more detection logic based on files, language, etc.
        return "Standard"
    
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing for analysis.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        import re
        
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def format_conversation_for_export(
        self,
        conversation_history: List[Any]
    ) -> List[Dict[str, Any]]:
        """Format conversation history for export.
        
        Args:
            conversation_history: Raw conversation history
            
        Returns:
            Formatted conversation list
        """
        formatted_conversation = []
        
        for message in conversation_history:
            if hasattr(message, 'role'):
                # ConversationMessage object
                formatted_message = {
                    'role': message.role,
                    'content': message.content,
                    'timestamp': message.timestamp.isoformat() if hasattr(message.timestamp, 'isoformat') else str(message.timestamp),
                    'metadata': getattr(message, 'metadata', {})
                }
            elif isinstance(message, dict):
                # Dictionary format
                formatted_message = {
                    'role': message.get('role', 'unknown'),
                    'content': message.get('content', ''),
                    'timestamp': message.get('timestamp', ''),
                    'metadata': message.get('metadata', {})
                }
            else:
                # Fallback
                formatted_message = {
                    'role': 'unknown',
                    'content': str(message),
                    'timestamp': '',
                    'metadata': {}
                }
            
            formatted_conversation.append(formatted_message)
        
        return formatted_conversation
    
    def create_dataset_statistics(self, issues: List[IssueData]) -> Dict[str, Any]:
        """Create statistics for a dataset.
        
        Args:
            issues: List of issue data
            
        Returns:
            Statistics dictionary
        """
        if not issues:
            return {'total_issues': 0}
        
        # Count by language
        language_counts = {}
        docker_issues = 0
        total_comments = 0
        total_conditions = 0
        repositories = set()
        
        for issue in issues:
            # Language statistics
            lang = issue.language.lower()
            language_counts[lang] = language_counts.get(lang, 0) + 1
            
            # Docker issues
            if issue.dockerfile:
                docker_issues += 1
            
            # Comment and condition counts
            total_comments += len(issue.comments)
            total_conditions += len(issue.user_satisfaction_condition)
            
            # Repository tracking
            repositories.add(issue.commit_info.repository)
        
        return {
            'total_issues': len(issues),
            'language_distribution': language_counts,
            'docker_issues': docker_issues,
            'regular_issues': len(issues) - docker_issues,
            'total_comments': total_comments,
            'average_comments_per_issue': total_comments / len(issues) if issues else 0,
            'total_satisfaction_conditions': total_conditions,
            'average_conditions_per_issue': total_conditions / len(issues) if issues else 0,
            'unique_repositories': len(repositories),
            'repositories': list(repositories)[:10]  # First 10 for reference
        }
