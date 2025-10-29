import os
from typing import Any, Dict, List, Optional

from strands.types.tools import ToolResult, ToolUse

TOOL_SPEC = {
            "name": 'fsRead',
            "description":
                'A tool for reading files.\n\n' +
                '## Overview\n' +
                'This tool returns the contents of files, with optional line range specification.\n\n' +
                '## When to use\n' +
                '- When you need to examine the content of a file or multiple files\n' +
                '- When you need to read specific line ranges from files\n' +
                '- When you need to analyze code or configuration files\n\n' +
                '## When not to use\n' +
                '- When you need to search for patterns across multiple files\n' +
                '- When you need to process files in binary format\n\n' +
                '## Notes\n' +
                '- Prioritize reading multiple files at once by passing in multiple paths rather than calling this tool with a single path multiple times\n' +
                '- When reading multiple files, the total characters combined cannot exceed 400K characters, break the step into smaller chunks if it happens\n' +
                '- This tool is more effective than running a command like `head -n` using `executeBash` tool\n' +
                '- If a file exceeds 200K characters, this tool will only read the first 200K characters of the file with a `truncated=true` in the output\n' +
                '- For large files (>200K characters), you may need to make multiple calls with specific `readRange` values, but ONLY do this if:\n' +
                '  * The initial read was truncated (indicated by `truncated=true` in the output)\n' +
                '  * A specific `readRange` is needed to focus on relevant sections\n' +
                '  * The user explicitly asks to read more of the file\n' +
                '- DO NOT re-read the file again using `readRange` unless explicitly asked by the user',
            "inputSchema": {
                "type": 'object',
                "properties": {
                    "paths": {
                        "description": 'List of file paths to read in a sequence',
                        "type": 'array',
                        "items": {
                            "type": 'object',
                            "properties": {
                                "path": {
                                    "description": 'Absolute path to a file, e.g. `/repo/file.py` for Unix-like system including Unix/Linux/macOS or `d:\\repo\\file.py` for Windows.',
                                    "type": 'string',
                                },
                                "readRange": {
                                    "description": 'Optional parameter when reading files.\n * If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[startLine, -1]` shows all lines from `startLine` to the end of the file.',
                                    "type": 'array',
                                    "items": {
                                        "type": 'number',
                                    },
                                },
                            },
                            "required": ['path'],
                        },
                    },
                },
                "required": ['paths'],
            }
}

MAX_FILE_SIZE = 200000  # 200K characters per file
MAX_TOTAL_SIZE = 400000  # 400K characters total

def read_single_file(file_path: str, read_range: Optional[List[int]] = None) -> Dict[str, Any]:
    """Read a single file with optional line range.
    
    Args:
        file_path (str): Path to the file
        read_range (Optional[List[int]]): Optional line range to read
        
    Returns:
        Dict[str, Any]: File content and metadata
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        return {
            "path": file_path,
            "status": "error",
            "error": f"File not found: {file_path}",
            "content": None
        }
        
    # Check if the path is a directory
    if os.path.isdir(file_path):
        return {
            "path": file_path,
            "status": "error",
            "error": f"Cannot read directory as a file: {file_path}",
            "content": None
        }
        
    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            if read_range:
                # Get all lines from the file
                all_lines = f.readlines()
                
                # Calculate the start and end lines based on the range
                start_line = max(0, read_range[0] - 1)  # Convert from 1-indexed to 0-indexed
                
                if len(read_range) > 1 and read_range[1] != -1:
                    end_line = min(read_range[1], len(all_lines))
                else:
                    end_line = len(all_lines)
                    
                # Extract the specified lines
                selected_lines = all_lines[start_line:end_line]
                content = ''.join(selected_lines)
                truncated = False
            else:
                # Read the whole file with character limit
                content = f.read(MAX_FILE_SIZE + 1)  # Read one extra character to check if truncation is needed
                truncated = len(content) > MAX_FILE_SIZE
                
                if truncated:
                    content = content[:MAX_FILE_SIZE]
        
        return {
            "path": file_path,
            "status": "success",
            "content": content,
            "truncated": truncated
        }
    except Exception as e:
        return {
            "path": file_path,
            "status": "error",
            "error": f"Error reading file: {str(e)}",
            "content": None
        }

def fsRead(tool: ToolUse, **kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]
    
    try:
        # Extract file paths array
        file_requests_raw = tool_input["paths"]
        
        # Handle empty paths array
        if not file_requests_raw:
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": "No file paths provided"}]
            }
        
        # Normalize file_requests to handle both string paths and object paths
        file_requests = []
        for req in file_requests_raw:
            if isinstance(req, str):
                # If the request is just a string path
                file_requests.append({"path": req, "readRange": None})
            else:
                # If the request is already an object
                file_requests.append(req)
        
        # Read each file
        file_results = []
        total_size = 0
        
        for file_request in file_requests:
            file_path = file_request["path"]
            read_range = file_request.get("readRange")
            
            result = read_single_file(file_path, read_range)
            
            # Add to file results
            file_results.append(result)
            
            # Update total size if successful
            if result["status"] == "success":
                total_size += len(result["content"])
        
        # Check if total size exceeds the limit
        if total_size > MAX_TOTAL_SIZE:
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": "Files are too large, please break the file read into smaller chunks"}]
            }
        
        # Format the response
        formatted_content = ""
        truncated_files = []
        error_files = []
        
        for result in file_results:
            formatted_content += f"\n\n# File: {result['path']}\n"
            
            if result["status"] == "success":
                formatted_content += result["content"]
                if result.get("truncated", False):
                    truncated_files.append(result["path"])
            else:
                formatted_content += f"ERROR: {result.get('error', 'Unknown error')}"
                error_files.append(result["path"])
        
        # Add notes about truncated files
        if truncated_files:
            truncation_notes = "\n\n[NOTE: The following files were truncated because they exceed 200K characters: "
            truncation_notes += ", ".join(truncated_files)
            truncation_notes += "]"
            formatted_content += truncation_notes
        
        # Add summary info to the content instead of using a separate field
        summary = f"\n\nSummary: Read {len(file_results) - len(error_files)} files successfully"
        if error_files:
            summary += f", {len(error_files)} files had errors"
        if truncated_files:
            summary += f", {len(truncated_files)} files were truncated"
        formatted_content += summary
        
        # Create response with only the allowed fields
        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": formatted_content.strip()}]
        }
        
    except Exception as e:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Error reading files: {str(e)}"}]
        }