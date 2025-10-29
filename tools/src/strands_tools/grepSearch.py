"""
grepSearch ― ripgrep‑backed full‑text search
"""

import os
import subprocess
from typing import Any, Dict, List, Optional

from strands.types.tools import ToolResult, ToolUse


# --------------------------------------------------------------------------- #
#  Constants                                                                   #
# --------------------------------------------------------------------------- #
DEFAULT_MAX_RESULTS = 20

# --------------------------------------------------------------------------- #
#  Tool specification (exposed to the LLM)                                    #
# --------------------------------------------------------------------------- #
TOOL_SPEC: Dict[str, Any] = {
    "name": "grepSearch",
    "description": (
        'A tool for searching text patterns across files.\n\n' +
                '## Overview\n' +
                'This tool searches for text content in files within a directory and its subdirectories.\n\n' +
                '## When to use\n' +
                '- When you need to find specific text patterns across multiple files\n' +
                '- When you need to locate code implementations, function definitions, or specific strings\n' +
                '- When you need to identify where certain features or components are used\n\n' +
                '## When not to use\n' +
                '- When you need to read the full content of specific files (use `fsRead` instead)\n' +
                '- When you need to search within binary files\n' +
                '- When you need to perform complex regex operations beyond simple text matching\n\n' +
                '## Notes\n' +
                '- Results include file paths, line numbers, and matching content\n' +
                '- Case sensitivity can be controlled with the caseSensitive parameter\n' +
                '- Include and exclude patterns can be specified to narrow down the search scope\n' +
                '- Results are limited to 20 matches per file to prevent overwhelming output\n' +
                '- This tool is more effective than running commands like `grep` or `find` using `executeBash` tool'
    ),
    "inputSchema": {
                "type": 'object',
                "properties": {
                    "path": {
                        "type": 'string',
                        "description":
                            'Absolute path to a directory to search in, e.g. `/repo` for Unix-like system including Unix/Linux/macOS or `d:\\repo` for Windows. If not provided, the current workspace folder will be used.',
                    },
                    "query": {
                        "type": 'string',
                        "description":
                            'The text pattern to search for in files. Can be a simple string or a regular expression pattern.',
                    },
                    "caseSensitive": {
                        "type": 'boolean',
                        "description": 'Whether the search should be case-sensitive. Defaults to false if not provided.',
                    },
                    "includePattern": {
                        "type": 'string',
                        "description":
                            'Comma-separated glob patterns to include in the search, e.g., "*.js,*.ts,src/**/*.jsx". Only files matching these patterns will be searched.',
                    },
                    "excludePattern": {
                        "type": 'string',
                        "description":
                            'Comma-separated glob patterns to exclude from the search, e.g., "*.min.js,*.d.ts,**/*.test.*". Files matching these patterns will be ignored.',
                    },
                },
                "required": ['query'],
            },
}

# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _clean_path(raw_path: str) -> str:
    """Strip any ':start-end' slice and return an absolute path."""
    return os.path.abspath(raw_path.split(":", 1)[0])


# --------------------------------------------------------------------------- #
#  Implementation                                                              #
# --------------------------------------------------------------------------- #
def grepSearch(tool: ToolUse, **kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    inp = tool["input"]

    try:
        raw_path: str = inp.get("path", ".")
        path: str = _clean_path(raw_path)
        query: str = inp["query"]

        case_sensitive: bool = inp.get("caseSensitive", False)
        include_pattern: Optional[str] = inp.get("includePattern")
        exclude_pattern: Optional[str] = inp.get("excludePattern")
        max_results: int = int(inp.get("maxResults", DEFAULT_MAX_RESULTS))
        if max_results <= 0:
            max_results = DEFAULT_MAX_RESULTS

        if not os.path.exists(path):
            raise FileNotFoundError(f"Invalid search path: {raw_path}")

        rg_cmd: List[str] = [
            "rg",
            "-n",
            "--no-heading",
            "--color=never",
            "-m",
            str(max_results),
        ]

        if not case_sensitive:
            rg_cmd.append("-i")

        # Auto‑restrict to file basename when a single file is given.
        if os.path.isfile(path) and not include_pattern:
            include_pattern = os.path.basename(path)

        if include_pattern:
            for patt in include_pattern.split(","):
                rg_cmd.extend(["-g", patt.strip()])

        if exclude_pattern:
            for patt in exclude_pattern.split(","):
                rg_cmd.extend(["-g", f"!{patt.strip()}"])

        rg_cmd.extend([query, path])

        proc = subprocess.run(
            rg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode not in (0, 1):  # 0=matches, 1=no matches
            raise RuntimeError(f"ripgrep error: {proc.stderr.strip()}")

        stdout = proc.stdout.strip()
        if not stdout:
            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [{"text": "Found 0 matches."}],
            }

        match_total = 0
        by_file: Dict[str, List[Dict[str, str]]] = {}

        for line in stdout.splitlines():
            try:
                fp, ln, text = line.split(":", 2)
            except ValueError:
                continue
            match_total += 1
            by_file.setdefault(fp, []).append({"lineNum": ln, "content": text})

        parts: List[str] = [f"Found {match_total} matches "
                            f"(showing up to {max_results}):"]
        for fp, mlist in sorted(by_file.items(),
                                key=lambda kv: len(kv[1]), reverse=True):
            parts.append(f"\n{fp}:")
            for m in mlist:
                parts.append(f"  Line {m['lineNum']}: {m['content']}")

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": "\n".join(parts)}],
        }

    except Exception as exc:   # pylint: disable=broad-except
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"grepSearch error: {exc}"}],
        }
