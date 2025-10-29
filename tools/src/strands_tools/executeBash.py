import os
import subprocess
from typing import Any, Dict, List, Optional

from strands.types.tools import ToolResult, ToolUse

TOOL_SPEC = {
    "name": "executeBash",
    "description": "Execute the specified command on the system shell (bash on Unix/Linux/macOS, cmd.exe on Windows).",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "explanation": {
                    "description": "One sentence explanation as to why this tool is being used, and how it contributes to the goal.",
                    "type": "string"
                },
                "command": {
                    "type": "string",
                    "description": "Command to execute on the system shell. On Windows, this will run in cmd.exe; on Unix-like systems, this will run in bash."
                },
                "cwd": {
                    "type": "string",
                    "description": "Parameter to set the current working directory for the command execution."
                }
            },
            "required": ["command", "cwd"]
        }
    },
}


def execute_bash_command(command: str, cwd: str) -> Dict[str, Any]:
    """Execute a bash command and return its output and status.

    Args:
        command: The bash command to execute
        cwd: The current working directory for the command

    Returns:
        Dict containing the stdout, stderr, and return code
    """
    # Ensure the working directory exists
    if not os.path.exists(cwd):
        raise FileNotFoundError(f"Working directory does not exist: {cwd}")
    
    # Execute the command using bash
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        executable="/bin/bash",
        universal_newlines=True
    )
    
    # Get the stdout and stderr
    stdout, stderr = process.communicate()
    return_code = process.returncode
    
    return {
        "stdout": stdout,
        "stderr": stderr,
        "return_code": return_code
    }


def executeBash(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Execute a bash command in the specified directory.

    Args:
        tool: Tool use information containing input parameters
        **kwargs: Additional keyword arguments

    Returns:
        ToolResult: Tool execution result with command output
    """
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]
    
    try:
        # Extract parameters
        command = tool_input["command"]
        cwd = tool_input["cwd"]
        explanation = tool_input.get("explanation", "")
        
        # Execute the command
        result = execute_bash_command(command, cwd)
        
        # Build the response
        stdout = result["stdout"]
        stderr = result["stderr"]
        return_code = result["return_code"]
        
        response_text = []
        
        # Add explanation if provided
        if explanation:
            response_text.append(f"Explanation: {explanation}")
        
        # Add command and its working directory
        response_text.append(f"Command: {command}")
        response_text.append(f"Working directory: {cwd}")
        
        # Add command output
        response_text.append("\nOutput:")
        response_text.append(stdout if stdout else "(no standard output)")
        
        # Add error output if any
        if stderr:
            response_text.append("\nErrors:")
            response_text.append(stderr)
        
        # Add return code
        response_text.append(f"\nReturn code: {return_code}")
        
        # Determine status based on return code
        status = "success" if return_code == 0 else "error"
        
        return {
            "toolUseId": tool_use_id,
            "status": status,
            "content": [{"text": "\n".join(response_text)}]
        }
    
    except Exception as e:
        # Return error with details
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Error executing bash command: {str(e)}"}]
        }