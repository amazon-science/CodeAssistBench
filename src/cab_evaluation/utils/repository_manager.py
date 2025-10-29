"""Repository management utilities."""

import os
import subprocess
import tempfile
import shutil
import random
import time
from typing import Optional, Tuple
import logging

from ..core.exceptions import RepositoryError

logger = logging.getLogger(__name__)


def execute_command(repo_dir: str, command: str, timeout: int = 60) -> str:
    """Execute a command in the repository directory with a timeout.
    
    Args:
        repo_dir: Repository directory path
        command: Command to execute
        timeout: Command timeout in seconds
        
    Returns:
        Command output (stdout + stderr)
        
    Raises:
        RepositoryError: If command execution fails
    """
    try:
        logger.info(f"Executing command: {command}")
        start_time = time.time()
        
        try:
            result = subprocess.run(
                command,
                cwd=repo_dir,
                capture_output=True,
                text=False,  # Get bytes to handle encoding issues
                shell=True,
                timeout=timeout
            )
            
            elapsed_time = time.time() - start_time
            
            # Handle stdout and stderr as bytes with proper encoding
            stdout_bytes = result.stdout if result.stdout else b''
            stderr_bytes = result.stderr if result.stderr else b''
            
            # Try UTF-8 first, then fall back with error replacement
            try:
                stdout_text = stdout_bytes.decode('utf-8', errors='replace')
            except:
                stdout_text = stdout_bytes.decode('latin-1')
                
            try:
                stderr_text = stderr_bytes.decode('utf-8', errors='replace')
            except:
                stderr_text = stderr_bytes.decode('latin-1')
            
            # Count lines for logging
            stdout_lines = stdout_text.strip().split('\n') if stdout_text.strip() else []
            stderr_lines = stderr_text.strip().split('\n') if stderr_text.strip() else []
            
            logger.info(
                f"Command executed in {elapsed_time:.2f}s. "
                f"STDOUT: {len(stdout_lines)} lines, STDERR: {len(stderr_lines)} lines"
            )
            
            # Log stderr if it exists
            if stderr_lines:
                logger.warning(f"Command stderr: {stderr_text[:200]}{'...' if len(stderr_text) > 200 else ''}")
            
            return f"STDOUT:\n{stdout_text}\n\nSTDERR:\n{stderr_text}"
            
        except subprocess.TimeoutExpired:
            elapsed_time = time.time() - start_time
            logger.warning(f"Command timed out after {elapsed_time:.2f}s: {command}")
            return f"ERROR: Command timed out after {timeout} seconds"
            
    except subprocess.SubprocessError as e:
        # Try to decode stderr with error handling
        stderr = ""
        if hasattr(e, 'stderr') and e.stderr:
            try:
                stderr = e.stderr.decode('utf-8', errors='replace')
            except:
                stderr = e.stderr.decode('latin-1')
        
        logger.error(f"Error executing command '{command}': {stderr or str(e)}")
        return f"Error executing command: {stderr or str(e)}"


def read_file(repo_dir: str, file_path: str) -> str:
    """Read the entire file.
    
    Args:
        repo_dir: Repository directory
        file_path: Relative path to file
        
    Returns:
        File content
    """
    try:
        full_path = os.path.join(repo_dir, file_path)
        logger.info(f"Reading file: {file_path}")
        with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        logger.info(f"Successfully read file ({len(content)} bytes)")
        return content
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return f"Error reading file: {e}"


def read_file_lines(repo_dir: str, file_path: str, start_line: int, end_line: int) -> str:
    """Read specific lines from a file.
    
    Args:
        repo_dir: Repository directory
        file_path: Relative path to file
        start_line: Start line number (1-based)
        end_line: End line number (1-based, inclusive)
        
    Returns:
        File content for specified lines
    """
    try:
        full_path = os.path.join(repo_dir, file_path)
        logger.info(f"Reading lines {start_line} to {end_line} from file: {file_path}")
        with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        # Convert to 0-based indexing and ensure within bounds
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)
        
        result = ''.join(lines[start_idx:end_idx])
        logger.info(f"Successfully read {end_idx - start_idx} lines from file")
        return result
    except Exception as e:
        logger.error(f"Error reading lines from file {file_path}: {e}")
        return f"Error reading file lines: {e}"


def list_directory(repo_dir: str, dir_path: str = '.') -> str:
    """List contents of a directory.
    
    Args:
        repo_dir: Repository directory
        dir_path: Directory path relative to repo_dir
        
    Returns:
        Directory listing
    """
    try:
        full_path = os.path.join(repo_dir, dir_path)
        logger.info(f"Listing directory: {dir_path}")
        contents = os.listdir(full_path)
        logger.info(f"Found {len(contents)} items in directory")
        return '\n'.join(contents)
    except Exception as e:
        logger.error(f"Error listing directory {dir_path}: {e}")
        return f"Error listing directory: {e}"


def find_files(repo_dir: str, pattern: str) -> str:
    """Find files matching a pattern.
    
    Args:
        repo_dir: Repository directory
        pattern: File pattern to search for
        
    Returns:
        List of matching files
    """
    try:
        logger.info(f"Finding files matching pattern: {pattern}")
        result = subprocess.run(
            ['find', '.', '-type', 'f', '-name', pattern],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True
        )
        files_found = result.stdout.strip().split('\n') if result.stdout.strip() else []
        logger.info(f"Found {len(files_found)} files matching pattern")
        return result.stdout
    except subprocess.SubprocessError as e:
        logger.error(f"Error finding files with pattern {pattern}: {e}")
        return f"Error finding files: {e}"


class RepositoryManager:
    """Manages repository operations for CAB evaluation."""
    
    def __init__(self, max_retries: int = 5):
        """Initialize repository manager.
        
        Args:
            max_retries: Maximum retries for repository operations
        """
        self.max_retries = max_retries
    
    def parse_repo_name(self, repo_url: str) -> str:
        """Parse repository name from URL.
        
        Args:
            repo_url: Repository URL
            
        Returns:
            Parsed repository name
        """
        parts = repo_url.split('/')
        if len(parts) >= 5:
            return '/'.join(parts[:5])
        else:
            return '/'.join(parts)
    
    def clone_repository(self, repo_url: str, commit_hash: str) -> Optional[str]:
        """Clone repository with efficient fetching of a specific commit.
        
        Args:
            repo_url: Repository URL to clone
            commit_hash: Specific commit hash to checkout
            
        Returns:
            Temporary directory path or None if failed
            
        Raises:
            RepositoryError: If cloning fails after all retries
        """
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory: {temp_dir}")
        
        # Extract repo name from URL
        repo_name = self.parse_repo_name(repo_url)
        logger.info(f"Cloning repository: {repo_name}")
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt}/{self.max_retries} for cloning {repo_name}")
                    
                    # Clean up previous failed attempt
                    if os.path.exists(temp_dir):
                        try:
                            shutil.rmtree(temp_dir)
                            os.makedirs(temp_dir, exist_ok=True)
                        except Exception as e:
                            logger.warning(f"Failed to clean up directory: {e}")
                            temp_dir = tempfile.mkdtemp()
                
                logger.info(f"Initial shallow clone of {repo_name} (attempt {attempt+1}/{self.max_retries+1})")
                clone_result = subprocess.run(
                    ["git", "clone", "--quiet", "--depth=1", repo_name, temp_dir],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=300  # 5 minute timeout
                )

                # Fetch the specific commit
                logger.info(f"Fetching specific commit: {commit_hash}")
                fetch_result = subprocess.run(
                    ["git", "fetch", "--quiet", "--depth=1", "origin", commit_hash],
                    cwd=temp_dir,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=300
                )

                # Checkout the commit
                logger.info(f"Checking out commit: {commit_hash}")
                checkout_result = subprocess.run(
                    ["git", "checkout", "--quiet", commit_hash],
                    cwd=temp_dir,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=60
                )
                
                logger.info(f"Successfully cloned repository at commit {commit_hash}")
                return temp_dir
                
            except subprocess.TimeoutExpired as e:
                logger.warning(f"Timeout during git operation (attempt {attempt+1}): {e}")
                if attempt == self.max_retries:
                    self._cleanup_temp_dir(temp_dir)
                    raise RepositoryError(
                        f"Failed to clone repository after {self.max_retries+1} attempts due to timeouts",
                        repo_url=repo_url
                    )
                    
            except subprocess.SubprocessError as e:
                stderr = e.stderr.decode('utf-8', errors='replace') if hasattr(e, 'stderr') and e.stderr else str(e)
                logger.warning(f"Git operation failed (attempt {attempt+1}): {stderr}")
                
                # Check for temporary network errors
                if any(error in stderr.lower() for error in [
                    "could not resolve host", "connection timed out", "temporarily unavailable"
                ]):
                    if attempt == self.max_retries:
                        self._cleanup_temp_dir(temp_dir)
                        raise RepositoryError(f"Failed to clone repository after {self.max_retries+1} attempts: {stderr}", repo_url)
                        
                    # Exponential backoff with jitter
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Waiting {wait_time:.2f}s before retry...")
                    time.sleep(wait_time)
                else:
                    # Permanent error, no retry
                    self._cleanup_temp_dir(temp_dir)
                    raise RepositoryError(f"Failed to clone repository (permanent error): {stderr}", repo_url)
                    
            except Exception as e:
                logger.error(f"Unexpected error during repository cloning: {str(e)}")
                self._cleanup_temp_dir(temp_dir)
                raise RepositoryError(f"Unexpected error cloning repository: {str(e)}", repo_url)
        
        # Should never reach here
        self._cleanup_temp_dir(temp_dir)
        raise RepositoryError("Failed to clone repository: retry loop exited unexpectedly", repo_url)
    
    def _cleanup_temp_dir(self, temp_dir: str):
        """Clean up temporary directory.
        
        Args:
            temp_dir: Directory to clean up
        """
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Error cleaning up temporary directory {temp_dir}: {e}")
    
    def cleanup_repository(self, repo_dir: str):
        """Clean up cloned repository.
        
        Args:
            repo_dir: Repository directory to clean up
        """
        self._cleanup_temp_dir(repo_dir)
