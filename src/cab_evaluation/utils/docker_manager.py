"""Docker management utilities for CAB evaluation."""

import json
import os
import tempfile
import time
import shutil
from typing import Dict, List, Optional, Tuple
import logging

import docker
from docker.errors import DockerException, BuildError, APIError
import requests

from ..core.models import DockerValidationResult, IssueData
from ..core.exceptions import DockerValidationError
from ..core.config import DockerConfig

logger = logging.getLogger(__name__)


class DockerManager:
    """Manages Docker operations for CAB evaluation."""
    
    def __init__(self, config: Optional[DockerConfig] = None):
        """Initialize Docker manager.
        
        Args:
            config: Docker configuration
        """
        self.config = config or DockerConfig()
        self.client = None
        
    def setup_client(self) -> bool:
        """Initialize Docker client with error handling.
        
        Returns:
            True if client setup successful, False otherwise
        """
        try:
            self.client = docker.from_env()
            # Test the connection
            self.client.ping()
            logger.info("Docker client initialized successfully")
            return True
        except (DockerException, requests.exceptions.ConnectionError) as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            return False
    
    def write_file_in_context(self, temp_dir: str, file_path: str, content: str) -> bool:
        """Write a file to the Docker build context.
        
        Args:
            temp_dir: Temporary directory for Docker context
            file_path: File path relative to context
            content: File content
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure path is relative and sanitized
            safe_path = os.path.normpath(file_path)
            if safe_path.startswith(os.sep):
                safe_path = safe_path[1:]  # Remove leading slash
                
            full_path = os.path.join(temp_dir, safe_path)
            
            # Create directories if they don't exist
            dir_path = os.path.dirname(full_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Created file {safe_path} in Docker build context")
            return True
        except Exception as e:
            logger.error(f"Failed to create file {file_path}: {e}")
            return False
    
    def build_and_run_dockerfile(
        self,
        issue_data: IssueData,
        test_commands: List[str],
        extra_files: Optional[Dict[str, str]] = None
    ) -> DockerValidationResult:
        """Build and run Dockerfile with test commands.
        
        Args:
            issue_data: Issue data containing Dockerfile
            test_commands: Commands to test inside container
            extra_files: Additional files to include in build context
            
        Returns:
            DockerValidationResult with build and test results
        """
        if not self.client:
            if not self.setup_client():
                return DockerValidationResult(
                    success=False,
                    logs="Failed to setup Docker client",
                    error="Docker client initialization failed"
                )
        
        temp_dir = None
        image_id = None
        container = None
        
        try:
            # Create temporary directory for Docker context
            temp_dir = tempfile.mkdtemp()
            logger.info(f"Created Docker build context at: {temp_dir}")
            
            # Write Dockerfile
            dockerfile_path = os.path.join(temp_dir, 'Dockerfile')
            with open(dockerfile_path, 'w', encoding='utf-8') as f:
                f.write(issue_data.dockerfile)
                
            logger.info(f"Written Dockerfile ({len(issue_data.dockerfile)} bytes)")
            
            # Write any additional files
            if extra_files:
                for file_path, content in extra_files.items():
                    success = self.write_file_in_context(temp_dir, file_path, content)
                    if not success:
                        return DockerValidationResult(
                            success=False,
                            logs=f"Failed to create required file: {file_path}",
                            test_commands=test_commands,
                            extra_files=extra_files or {},
                            error=f"File creation failed: {file_path}"
                        )
            
            # Create test script
            if test_commands:
                test_script_path = os.path.join(temp_dir, 'docker_test.sh')
                with open(test_script_path, 'w', encoding='utf-8') as f:
                    f.write("#!/bin/bash\n")
                    f.write("set -e\n")  # Exit on first error
                    f.write("echo 'Starting Docker validation tests...'\n\n")
                    
                    for i, cmd in enumerate(test_commands):
                        # Skip docker commands as they won't work inside container
                        if cmd.startswith("docker "):
                            f.write(f"echo 'Skipping docker command: {cmd}'\n")
                            continue
                            
                        f.write(f"echo 'Test {i+1}: {cmd}'\n")
                        f.write(f"{cmd}\n")
                        f.write(f"echo 'Test {i+1} completed successfully'\n\n")
                    
                    f.write("echo 'All tests passed successfully!'\n")
                    
                # Make script executable
                os.chmod(test_script_path, 0o755)
                logger.info(f"Created test script with {len(test_commands)} commands")
                
                # Modify Dockerfile to include test script
                with open(dockerfile_path, 'a', encoding='utf-8') as f:
                    f.write("\n# Add test script\n")
                    f.write("COPY docker_test.sh /docker_test.sh\n")
                    f.write("RUN chmod +x /docker_test.sh\n\n")
                    f.write("# Test commands will run when container starts\n")
                    f.write("CMD [\"/docker_test.sh\"]\n")
            
            # Build Docker image
            logger.info(f"Building Docker image (timeout: {self.config.build_timeout}s)...")
            build_logs = []
            
            try:
                # Use low-level API to get build progress
                build_result = self.client.api.build(
                    path=temp_dir,
                    rm=True,
                    forcerm=True,
                    decode=True,
                    timeout=self.config.build_timeout
                )
                
                # Collect build logs
                for chunk in build_result:
                    if 'stream' in chunk:
                        build_logs.append(chunk['stream'].strip())
                        logger.debug(f"Build log: {chunk['stream'].strip()}")
                    elif 'error' in chunk:
                        error_msg = chunk['error'].strip()
                        build_logs.append(f"ERROR: {error_msg}")
                        logger.error(f"Build error: {error_msg}")
                        
                    # Check for image ID
                    if 'aux' in chunk and 'ID' in chunk['aux']:
                        image_id = chunk['aux']['ID']
                        
                # Try to extract image ID from logs if not found
                if not image_id:
                    for line in build_logs:
                        if line.startswith("Successfully built "):
                            image_id = line.split("Successfully built ")[1].strip()
                            break
                            
                if not image_id:
                    return DockerValidationResult(
                        success=False,
                        logs='\n'.join(build_logs),
                        test_commands=test_commands,
                        extra_files=extra_files or {},
                        error="Failed to get image ID after build"
                    )
                    
                logger.info(f"Docker image built successfully: {image_id}")
                
            except (BuildError, APIError) as e:
                logger.error(f"Docker build failed: {str(e)}")
                build_logs.append(f"BUILD FAILED: {str(e)}")
                return DockerValidationResult(
                    success=False,
                    logs='\n'.join(build_logs),
                    test_commands=test_commands,
                    extra_files=extra_files or {},
                    error=f"Docker build failed: {str(e)}"
                )
            
            # Run the container
            logger.info(f"Running Docker container (timeout: {self.config.run_timeout}s)...")
            
            try:
                # Create and start container
                container = self.client.containers.create(image_id)
                container.start()
                
                # Wait for completion with timeout
                start_time = time.time()
                
                while time.time() - start_time < self.config.run_timeout:
                    container.reload()
                    if container.status != 'running':
                        break
                    time.sleep(1)
                
                # Get results
                container.reload()
                logs = container.logs().decode('utf-8', errors='replace')
                
                # Check if timed out
                if time.time() - start_time >= self.config.run_timeout:
                    logger.warning(f"Container timed out after {self.config.run_timeout}s")
                    container.stop(timeout=10)
                    success = False
                    logs += "\n\nERROR: Container execution timed out"
                else:
                    # Check exit code
                    exit_code = container.attrs['State']['ExitCode']
                    success = exit_code == 0
                    logger.info(f"Container exited with code {exit_code}")
                
                # Format result
                result_logs = f"Build logs:\n{chr(10).join(build_logs)}\n\nContainer logs:\n{logs}"
                
                return DockerValidationResult(
                    success=success,
                    logs=result_logs,
                    test_commands=test_commands,
                    extra_files=extra_files or {},
                    modified_dockerfile=False
                )
                
            except APIError as e:
                logger.error(f"Docker API error during container run: {str(e)}")
                return DockerValidationResult(
                    success=False,
                    logs=f"Docker run error: {str(e)}\n\nBuild logs:\n{chr(10).join(build_logs)}",
                    test_commands=test_commands,
                    extra_files=extra_files or {},
                    error=f"Container run failed: {str(e)}"
                )
                
        except Exception as e:
            logger.error(f"Docker operation failed: {str(e)}")
            return DockerValidationResult(
                success=False,
                logs=f"Docker operation failed: {str(e)}",
                test_commands=test_commands,
                extra_files=extra_files or {},
                error=str(e)
            )
            
        finally:
            # Cleanup resources
            self._cleanup_docker_resources(container, image_id, temp_dir)
    
    def validate_docker_solution(
        self,
        issue_data: IssueData,
        test_commands: Optional[List[str]] = None
    ) -> DockerValidationResult:
        """Validate a Docker solution by building and testing.
        
        Args:
            issue_data: Issue data with Dockerfile
            test_commands: Optional test commands to run
            
        Returns:
            DockerValidationResult with validation results
        """
        if not self.client:
            if not self.setup_client():
                return DockerValidationResult(
                    success=False,
                    logs="Failed to setup Docker client",
                    error="Docker client initialization failed"
                )
        
        temp_dir = None
        image_id = None
        
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            logger.info(f"Created Docker build context at: {temp_dir}")
            
            # Write Dockerfile
            dockerfile_path = os.path.join(temp_dir, 'Dockerfile')
            with open(dockerfile_path, 'w', encoding='utf-8') as f:
                f.write(issue_data.dockerfile)
            
            # Add extra files if present
            if hasattr(issue_data, 'extra_files') and issue_data.extra_files:
                for file_path, content in issue_data.extra_files.items():
                    success = self.write_file_in_context(temp_dir, file_path, content)
                    if not success:
                        return DockerValidationResult(
                            success=False,
                            logs=f"Failed to create required file: {file_path}",
                            error=f"File creation failed: {file_path}"
                        )
            
            # Build the image
            logger.info("Building Docker image to validate Dockerfile...")
            try:
                image, build_logs = self.client.images.build(
                    path=temp_dir,
                    rm=True,
                    forcerm=True,
                    timeout=self.config.build_timeout
                )
                image_id = image.id
                logger.info(f"Docker image built successfully: {image_id}")
            except Exception as e:
                logger.error(f"Docker build failed: {str(e)}")
                return DockerValidationResult(
                    success=False,
                    logs=f"Docker build failed: {str(e)}",
                    error=f"Build failed: {str(e)}"
                )
            
            # Test container startup
            logger.info("Running basic container startup test...")
            try:
                result = self.client.containers.run(
                    image.id,
                    command="echo 'Container starts successfully'",
                    remove=True,
                    timeout=60
                )
                logger.info("Basic container test succeeded")
            except Exception as e:
                logger.error(f"Container startup test failed: {str(e)}")
                return DockerValidationResult(
                    success=False,
                    logs=f"Container startup test failed: {str(e)}",
                    error=f"Container startup failed: {str(e)}"
                )
            
            # Run additional test commands
            test_results = []
            if test_commands:
                logger.info(f"Running {len(test_commands)} test commands...")
                
                for i, cmd in enumerate(test_commands):
                    try:
                        # Skip docker build/run commands
                        if cmd.startswith(("docker build", "docker run")):
                            test_results.append(f"Skipping Docker command: {cmd}")
                            continue
                        
                        # Execute command inside container
                        container_cmd = f"bash -c '{cmd}'"
                        logger.info(f"Running test {i+1}: {container_cmd}")
                        
                        output = self.client.containers.run(
                            image.id,
                            command=container_cmd,
                            remove=True,
                            timeout=self.config.run_timeout
                        )
                        output_text = output.decode('utf-8', errors='replace')
                        test_results.append(f"Test {i+1} passed: {cmd}\nOutput: {output_text}")
                        logger.info(f"Test {i+1} succeeded")
                    except Exception as e:
                        error_msg = f"Test {i+1} failed: {cmd}\nError: {str(e)}"
                        test_results.append(error_msg)
                        logger.error(f"Test {i+1} failed: {str(e)}")
                        return DockerValidationResult(
                            success=False,
                            logs="\n".join(test_results),
                            test_commands=test_commands,
                            error=f"Test command failed: {cmd}"
                        )
                
                logger.info("All tests passed successfully")
            
            return DockerValidationResult(
                success=True,
                logs="\n".join(test_results) if test_results else "Docker validation succeeded",
                test_commands=test_commands or []
            )
            
        except Exception as e:
            logger.error(f"Docker validation failed: {str(e)}")
            return DockerValidationResult(
                success=False,
                logs=f"Docker validation failed: {str(e)}",
                error=str(e)
            )
            
        finally:
            # Cleanup
            self._cleanup_docker_resources(None, image_id, temp_dir)
    
    def generate_test_commands(
        self,
        issue_data: IssueData,
        maintainer_response: str,
        exploration_results: Optional[str] = None,
        llm_service = None
    ) -> List[str]:
        """Generate test commands based on maintainer response.
        
        Args:
            issue_data: Issue data
            maintainer_response: Maintainer's response
            exploration_results: Repository exploration results
            llm_service: LLM service for generating commands
            
        Returns:
            List of test commands
        """
        # Import here to avoid circular imports
        from ..prompts.constants import TaskPrompts
        
        dockerfile = issue_data.dockerfile or ''
        
        # Create context from exploration results
        docker_context = ""
        if exploration_results:
            docker_context = f"""
            EXPLORATION RESULTS:
            The following information was gathered during repository exploration in a temporary directory.
            IMPORTANT: These results show the repository structure, but paths will be DIFFERENT inside the Docker container.
            Use this information to understand the codebase, but adapt paths according to the Dockerfile context.
            
            {exploration_results}
            """
        
        system_prompt = f"""
        {TaskPrompts.TEST_COMMAND_GENERATION}
        
        {docker_context}
        
        Carefully analyze the Dockerfile:
        ```
        {dockerfile}
        ```
        
        Based on the Dockerfile above:
        1. Only reference paths and directories that definitely exist in the container
        2. Only use commands that would be available in the container
        3. Check for critical files or services mentioned in the issue
        
        Return ONLY a JSON array of commands.
        """
        
        user_prompt = f"""
        Maintainer's response:
        {maintainer_response}
        
        Dockerfile content:
        {dockerfile}
        
        Generate test commands that will verify the maintainer's response works correctly.
        These commands will run INSIDE the Docker container.
        
        REMEMBER: Only reference paths and resources that will definitely exist in the container.
        Also, only generate the minimal number of commands needed to verify the solution.
        
        Return ONLY a JSON array of commands.
        """
        
        if not llm_service:
            logger.warning("No LLM service provided for test command generation")
            return []
        
        try:
            # This would need to be implemented with proper async handling
            # For now, return empty list as placeholder
            logger.warning("Test command generation not fully implemented")
            return []
        except Exception as e:
            logger.error(f"Failed to generate test commands: {e}")
            return []
    
    def _cleanup_docker_resources(
        self,
        container,
        image_id: Optional[str],
        temp_dir: Optional[str]
    ):
        """Clean up Docker resources.
        
        Args:
            container: Container object to remove
            image_id: Image ID to remove
            temp_dir: Temporary directory to remove
        """
        try:
            # Clean up container
            if container:
                try:
                    logger.info(f"Removing container: {container.id}")
                    container.remove(force=True)
                except Exception as e:
                    logger.warning(f"Error removing container: {e}")
            
            # Clean up image
            if image_id and self.client:
                try:
                    logger.info(f"Removing image: {image_id}")
                    self.client.images.remove(image_id, force=True)
                except Exception as e:
                    logger.warning(f"Error removing image: {e}")
            
            # Clean up temp directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    logger.info(f"Removing temporary directory: {temp_dir}")
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Error removing temporary directory: {e}")
                    
        except Exception as e:
            logger.warning(f"Error during Docker cleanup: {e}")
