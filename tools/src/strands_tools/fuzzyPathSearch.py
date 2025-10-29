import os
import logging
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz
from strands.types.tools import ToolResult, ToolUse
from tools.src.strands_tools.listDirectory import is_excluded_dir, get_file_type_prefix

TOOL_SPEC = {
    "name": "fuzzyPathSearch",
    "description": (
        "Search for files and directories in a target path using fuzzy name matching.\n\n"
        "## Overview\n"
        "This tool recursively lists all files and folders starting from a given path,\n"
        "then performs fuzzy matching against the full list based on the provided query.\n\n"
        "## When to use\n"
        "- When you want to find files or folders by partial names\n"
        "- When the structure is too large for exact path recall\n"
        "- When skipping intermediate listing steps is desired\n\n"
        "## Limitations\n"
        "- Does not search file contents\n"
        "- Ignores some directories like `build/`, `dist/`, `node_modules/`, etc.\n"
        "- Does not follow symlinks\n\n"
        "## Notes\n"
        "- Use `maxDepth` to limit traversal\n"
        "- `matchThreshold` controls how loose or strict fuzzy matching is (default: 80)\n"
        "- `caseSensitive` is false by default"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to start the search from."
            },
            "queryName": {
                "type": "string",
                "description": "Name fragment to fuzzy match against file and directory names."
            },
            "maxDepth": {
                "type": "number",
                "description": "Maximum depth to traverse from the root directory."
            },
            "caseSensitive": {
                "type": "boolean",
                "description": "Enable case-sensitive matching. Default is false."
            },
            "matchThreshold": {
                "type": "number",
                "description": "Fuzzy match threshold (0-100). Higher = stricter match. Default is 80.",
                "default": 80
            }
        },
        "required": ["path", "queryName"]
    }
}

def is_fuzzy_match(name: str, query: str, threshold: int = 80, case_sensitive: bool = False) -> bool:
    if not case_sensitive:
        name = name.lower()
        query = query.lower()
    return fuzz.partial_ratio(name, query) >= threshold

def collect_paths(
    root: str,
    max_depth: Optional[int],
    current_depth: int = 0
) -> List[str]:
    if not os.path.exists(root):
        raise FileNotFoundError(f"Directory not found: {root}")
    if not os.path.isdir(root):
        raise NotADirectoryError(f"Not a directory: {root}")

    results = []
    if max_depth is not None and current_depth > max_depth:
        return results

    try:
        for entry in sorted(os.listdir(root)):
            full_path = os.path.join(root, entry)

            if os.path.isdir(full_path) and is_excluded_dir(full_path):
                continue

            results.append(full_path)

            if os.path.isdir(full_path) and not os.path.islink(full_path):
                results.extend(
                    collect_paths(full_path, max_depth, current_depth + 1)
                )
    except Exception as e:
        results.append(f"[Error reading {root}: {str(e)}]")

    return results

def fuzzyPathSearch(tool: ToolUse, **kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]

    try:
        path = tool_input["path"]
        query = tool_input["queryName"]
        max_depth = tool_input.get("maxDepth")
        case_sensitive = tool_input.get("caseSensitive", False)
        threshold = tool_input.get("matchThreshold", 80)

        all_paths = collect_paths(path, max_depth)
        matches = []

        for p in all_paths:
            name = os.path.basename(p)
            if is_fuzzy_match(name, query, threshold, case_sensitive):
                prefix = get_file_type_prefix(p)
                matches.append(f"{prefix} {p}")

        if not matches:
            message = f"No items similar to \"{query}\" found in {path}"
        else:
            message = f"Found {len(matches)} matching items in {path}:\n\n" + "\n".join(matches)

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": message}]
        }

    except Exception as e:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Error during search: {str(e)}"}]
        }
