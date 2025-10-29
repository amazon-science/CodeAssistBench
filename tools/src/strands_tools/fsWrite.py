import os
from typing import Any, Dict, Optional

from strands.types.tools import ToolResult, ToolUse

TOOL_SPEC = {
            "name": 'fsWrite',
            "description":
                'A tool for creating and editing a file.\n\n' +
                '## Overview\n' +
                'This tool provides multiple commands for file operations including creating, replacing, inserting, and appending content.\n\n' +
                '## When to use\n' +
                '- When creating new files (create)\n' +
                '- When replacing specific text in existing files (strReplace)\n' +
                '- When inserting text at a specific line (insert)\n' +
                '- When adding text to the end of a file (append)\n\n' +
                '## When not to use\n' +
                '- When you only need to read file content (use fsRead instead)\n' +
                '- When you need to delete a file (no delete operation is available)\n' +
                '- When you need to rename or move a file\n\n' +
                '## Command details\n' +
                '- The `create` command will override the file at `path` if it already exists as a file, and otherwise create a new file. Use this command for initial file creation, such as scaffolding a new project. You should also use this command when overwriting large boilerplate files where you want to replace the entire content at once.\n' +
                '- The `insert` command will insert `newStr` after `insertLine` and place it on its own line.\n' +
                '- The `append` command will add content to the end of an existing file, automatically adding a newline if the file does not end with one.\n' +
                '- The `strReplace` command will replace `oldStr` in an existing file with `newStr`.\n\n' +
                '## IMPORTANT Notes for using the `strReplace` command\n' +
                '- Use this command to delete code by using empty `newStr` parameter.\n' +
                '- If you need to make small changes to an existing file, consider using `strReplace` command to avoid unnecessary rewriting the entire file.\n' +
                '- Prefer the `create` command if the complexity or number of changes would make `strReplace` unwieldy or error-prone.\n' +
                '- The `oldStr` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces! Include just the changing lines, and a few surrounding lines if needed for uniqueness. Do not include long runs of unchanging lines in `oldStr`.\n' +
                '- The `newStr` parameter should contain the edited lines that should replace the `oldStr`.\n' +
                '- When multiple edits to the same file are needed, combine them into a single call whenever possible. This improves efficiency by reducing the number of tool calls and ensures the file remains in a consistent state.',
            "inputSchema": {
                "type": 'object',
                "properties": {
                    "command": {
                        "type": 'string',
                        "enum": ["create", "strReplace", "insert", "append"],
                        "description":
                            'The commands to run. Allowed options are: `create`, `strReplace`, `insert`, `append`.',
                    },
                    "explanation": {
                        "description":
                            'One sentence explanation as to why this tool is being used, and how it contributes to the goal.',
                        "type": 'string',
                    },
                    "fileText": {
                        "description":
                            'Required parameter of `create` command, with the content of the file to be created.',
                        "type": 'string',
                    },
                    "insertLine": {
                        "description":
                            'Required parameter of `insert` command. The `newStr` will be inserted AFTER the line `insertLine` of `path`.',
                        "type": 'number',
                    },
                    "newStr": {
                        "description":
                            'Required parameter of `strReplace` command containing the new string. Required parameter of `insert` command containing the string to insert. Required parameter of `append` command containing the content to append to the file.',
                        "type": 'string',
                    },
                    "oldStr": {
                        "description":
                            'Required parameter of `strReplace` command containing the string in `path` to replace.',
                        "type": 'string',
                    },
                    "path": {
                        "description":
                            'Absolute path to a file, e.g. `/repo/file.py` for Unix-like system including Unix/Linux/macOS or `d:\\repo\\file.py` for Windows.',
                        "type": 'string',
                    },
                },
                "required": ['command', 'path'],
            },
        }

def create_file(path: str, content: str) -> str:
    """Create a new file or overwrite an existing one with the given content.

    Args:
        path: Path to the file to create or overwrite
        content: Content to write to the file

    Returns:
        A message indicating the result
    """
    # Create directories if they don't exist
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    # Write content to file
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    
    return f"File created successfully at {path}"


def append_to_file(path: str, content: str) -> str:
    """Append content to an existing file.

    Args:
        path: Path to the existing file
        content: Content to append to the file

    Returns:
        A message indicating the result

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    if not os.path.exists(path) or not os.path.isfile(path):
        raise FileNotFoundError(f"Cannot append to non-existent file: {path}")
    
    # Check if the file ends with a newline
    with open(path, "r", encoding="utf-8") as f:
        existing_content = f.read()
    
    # Append a newline if the file doesn't end with one
    with open(path, "a", encoding="utf-8") as f:
        if existing_content and not existing_content.endswith("\n"):
            f.write("\n")
        f.write(content)
    
    return f"Content appended successfully to {path}"


def string_replace(path: str, old_str: str, new_str: str) -> str:
    """Replace a specific string in a file with a new string.

    Args:
        path: Path to the file
        old_str: The string to replace
        new_str: The string to replace with

    Returns:
        A message indicating the result

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the old string isn't unique or isn't found
    """
    if not os.path.exists(path) or not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check if old_str exists in the file
    if old_str not in content:
        raise ValueError(f"String to replace was not found in file {path}")
    
    # Check if old_str is unique in the file
    if content.count(old_str) > 1:
        raise ValueError(f"String to replace is not unique in file {path}")
    
    # Replace old_str with new_str
    new_content = content.replace(old_str, new_str)
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    return f"String replacement successful in {path}"


def insert_at_line(path: str, insert_line: int, new_str: str) -> str:
    """Insert a string after a specific line in a file.

    Args:
        path: Path to the file
        insert_line: The line number after which to insert the new string
        new_str: The string to insert

    Returns:
        A message indicating the result

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the line number is invalid
    """
    if not os.path.exists(path) or not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    if insert_line < 0 or insert_line >= len(lines):
        raise ValueError(f"Invalid line number {insert_line} for file with {len(lines)} lines")
    
    # Ensure new_str has a trailing newline
    if not new_str.endswith("\n"):
        new_str += "\n"
    
    # Insert new_str after the specified line
    lines.insert(insert_line + 1, new_str)
    
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    
    return f"Content inserted successfully after line {insert_line} in {path}"


def fsWrite(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """File system write tool for creating and editing files.

    Args:
        tool: Tool use information containing input parameters
        **kwargs: Additional keyword arguments

    Returns:
        ToolResult: Tool execution result with success or error status
    """
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]
    
    try:
        # Extract common parameters
        command = tool_input["command"]
        path = tool_input["path"]
        explanation = tool_input.get("explanation", "")
        
        # Process based on command type
        if command == "create":
            if "fileText" not in tool_input:
                raise ValueError("Missing required 'fileText' parameter for 'create' command")
            result = create_file(path, tool_input["fileText"])
        
        elif command == "append":
            if "newStr" not in tool_input:
                raise ValueError("Missing required 'newStr' parameter for 'append' command")
            result = append_to_file(path, tool_input["newStr"])
        
        elif command == "strReplace":
            if "oldStr" not in tool_input or "newStr" not in tool_input:
                raise ValueError("Missing required 'oldStr' or 'newStr' parameters for 'strReplace' command")
            result = string_replace(path, tool_input["oldStr"], tool_input["newStr"])
        
        elif command == "insert":
            if "insertLine" not in tool_input or "newStr" not in tool_input:
                raise ValueError("Missing required 'insertLine' or 'newStr' parameters for 'insert' command")
            result = insert_at_line(path, tool_input["insertLine"], tool_input["newStr"])
        
        else:
            raise ValueError(f"Unknown command: {command}")
        
        # Return success
        message = f"{result}" + (f" - {explanation}" if explanation else "")
        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": message}]
        }
    
    except Exception as e:
        # Return error with details
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Error: {str(e)}"}]
        }