import os
import re
import stat
from typing import Any, Dict, List, Optional, Set, Tuple

from strands.types.tools import ToolResult, ToolUse

TOOL_SPEC = {
            "name": 'listDirectory',
            "description":
                'List the contents of a directory and its subdirectories in a tree-like format.\n\n' +
                '## Overview\n' +
                'This tool recursively lists directory contents in a visual tree structure, ignoring common build and dependency directories.\n\n' +
                '## When to use\n' +
                '- When exploring a codebase or project structure\n' +
                '- When you need to discover files in a directory hierarchy\n' +
                '- When you need to understand the organization of a project\n' +
                '- When you need to filter files based on specific patterns\n\n' +
                '## When not to use\n' +
                '- When you already know the exact file path you need\n' +
                '- When you need to confirm the existence of files you may have created (the user will let you know if files were created successfully)\n\n' +
                '## Notes\n' +
                '- This tool will ignore directories such as `build/`, `out/`, `dist/` and `node_modules/`\n' +
                '- This tool is more effective than running a command like `ls` using `executeBash` tool\n' +
                '- Results are displayed in a tree format with directories ending in `/` and symbolic links ending in `@`\n' +
                '- Use the `maxDepth` parameter to control how deep the directory traversal goes\n' +
                '- Use the `includePatterns` parameter to only show files/directories matching these regex patterns\n' +
                '- Use the `excludePatterns` parameter to hide files/directories matching these regex patterns',
            "inputSchema": {
                "type": 'object',
                "properties": {
                    "path": {
                        "type": 'string',
                        "description":
                            'Absolute path to a directory, e.g. `/repo` for Unix-like system including Unix/Linux/macOS or `d:\\repo\\` for Windows',
                    },
                    "maxDepth": {
                        "type": 'number',
                        "description":
                            'Maximum depth to traverse when listing directories. Use `0` to list only the specified directory, `1` to include immediate subdirectories, etc. If it is not provided, it will list all subdirectories recursively.',
                    },
                    "includePatterns": {
                        "type": 'array',
                        "items": {"type": "string"},
                        "description": 
                            'List of regex patterns. Only files/directories matching at least one of these patterns will be shown. If not provided, all files will be included (subject to excludePatterns).',
                    },
                    "excludePatterns": {
                        "type": 'array',
                        "items": {"type": "string"},
                        "description": 
                            'List of regex patterns. Files/directories matching any of these patterns will be excluded from the listing.',
                    }
                },
                "required": ['path'],
            }
}
# Define directories to be excluded from listing
EXCLUDED_DIRS = {
    "node_modules", "dist", "build", "out"
}


def is_excluded_dir(path: str) -> bool:
    """Check if a directory should be excluded from listing.
    
    Args:
        path: Path to check
        
    Returns:
        True if the directory should be excluded, False otherwise
    """
    basename = os.path.basename(path)
    
    # Check if the directory name is in the excluded list
    if basename in EXCLUDED_DIRS:
        return True
        
    # Check if the path contains excluded directories
    parts = path.split(os.path.sep)
    return any(part in EXCLUDED_DIRS for part in parts)


def get_file_type_prefix(path: str) -> str:
    """Get a prefix that indicates the file type (file, directory, symlink).
    
    Args:
        path: Path to check
        
    Returns:
        String prefix indicating the file type: [F], [D], or [L]
    """
    if os.path.islink(path):
        return "[L]"
    elif os.path.isdir(path):
        return "[D]"
    else:
        return "[F]"


def matches_patterns(path: str, 
                    include_patterns: Optional[List[str]] = None, 
                    exclude_patterns: Optional[List[str]] = None) -> bool:
    """Check if a path matches the given patterns.
    
    Args:
        path: Path to check
        include_patterns: List of regex patterns to include
        exclude_patterns: List of regex patterns to exclude
        
    Returns:
        True if the path should be included, False otherwise
    """
    basename = os.path.basename(path)
    
    # If exclude patterns are provided and the path matches any, exclude it
    if exclude_patterns:
        for pattern in exclude_patterns:
            if re.search(pattern, basename) or re.search(pattern, path):
                return False
    
    # If include patterns are provided, the path must match at least one
    if include_patterns:
        return any(re.search(pattern, basename) or re.search(pattern, path) for pattern in include_patterns)
    
    # If no include patterns are provided, include everything that wasn't excluded
    return True


def list_directory_contents(
    path: str, 
    max_depth: Optional[int] = None, 
    current_depth: int = 0,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None
) -> List[str]:
    """List the contents of a directory recursively.
    
    Args:
        path: Path to the directory to list
        max_depth: Maximum depth to traverse (None for unlimited)
        current_depth: Current recursive depth
        include_patterns: List of regex patterns to include
        exclude_patterns: List of regex patterns to exclude
        
    Returns:
        List of formatted directory entry strings
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory not found: {path}")
    
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Path is not a directory: {path}")
    
    result = []
    
    # For level 0, add the root directory itself
    if current_depth == 0:
        result.append(f"{get_file_type_prefix(path)} {path}")
        
    # If max_depth is specified and we've reached it, stop recursion
    if max_depth is not None and current_depth >= max_depth:
        return result
    
    try:
        # Get all entries in the directory, sorted alphabetically
        entries = sorted(os.listdir(path))
        indent = "  " * (current_depth + 1)
        
        for entry in entries:
            entry_path = os.path.join(path, entry)
            
            # Check if this is a directory that should be excluded
            if os.path.isdir(entry_path) and is_excluded_dir(entry_path):
                continue
                
            # Check if this entry matches the pattern filters
            entry_matches = matches_patterns(entry_path, include_patterns, exclude_patterns)
            
            # Add this entry to the result if it matches the patterns
            if entry_matches:
                prefix = get_file_type_prefix(entry_path)
                result.append(f"{indent}{prefix} {entry}")
            
            # If this is a directory, recursively add its contents
            if os.path.isdir(entry_path) and not os.path.islink(entry_path):
                subdir_results = list_directory_contents(
                    entry_path, 
                    max_depth, 
                    current_depth + 1,
                    include_patterns,
                    exclude_patterns
                )
                
                # If we found matches in the subdirectory and this directory wasn't already added
                if subdir_results and len(subdir_results) > 0 and not entry_matches:
                    # Add the parent directory so the tree structure makes sense
                    prefix = get_file_type_prefix(entry_path)
                    result.append(f"{indent}{prefix} {entry}")
                
                # Add all the subdirectory results
                result.extend(subdir_results)
    
    except PermissionError:
        result.append(f"{indent}[Permission denied]")
    except Exception as e:
        result.append(f"{indent}[Error: {str(e)}]")
    
    return result


def listDirectory(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """List the contents of a directory and its subdirectories.
    
    Args:
        tool: Tool use information containing input parameters
        **kwargs: Additional keyword arguments
        
    Returns:
        ToolResult: Tool execution result with directory listing
    """
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]
    
    try:
        # Extract parameters
        path = tool_input["path"]
        max_depth = tool_input.get("maxDepth")
        include_patterns = tool_input.get("includePatterns")
        exclude_patterns = tool_input.get("excludePatterns")
        
        # List directory contents
        listing = list_directory_contents(
            path, 
            max_depth, 
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns
        )
        
        # Build the response
        header = f"Directory listing for: {path}"
        if max_depth is not None:
            header += f" (max depth: {max_depth})"
            
        depth_info = ""
        if max_depth is None:
            depth_info = "\nListing all subdirectories recursively (no depth limit)"
        elif max_depth == 0:
            depth_info = "\nListing only the specified directory (depth 0)"
        else:
            depth_info = f"\nListing subdirectories up to depth {max_depth}"
            
        exclusion_info = "\nExcluded build outputs and dependency directories such as: build/, out/, dist/, node_modules/, etc."
        
        # Add pattern info if applicable
        pattern_info = ""
        if include_patterns:
            pattern_info += f"\nIncluding only files/directories matching: {', '.join(include_patterns)}"
        if exclude_patterns:
            pattern_info += f"\nExcluding files/directories matching: {', '.join(exclude_patterns)}"
            
        legend = "\nLegend: [F] File, [D] Directory, [L] Symlink"
        
        # Format the full listing output
        full_listing = "\n".join(listing)
        
        response_text = f"{header}{depth_info}{exclusion_info}{pattern_info}{legend}\n\n{full_listing}"
        
        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": response_text}]
        }
    
    except Exception as e:
        # Return error with details
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Error listing directory: {str(e)}"}]
        }