from __future__ import annotations

"""fs_write – Python port mirroring the Rust implementation.

Supported commands
------------------
* **create**      – create/overwrite a file with `file_text` (or fallback `new_str`).
* **str_replace** – replace *exactly* one occurrence of `old_str` with `new_str`.
* **insert**      – insert `new_str` *after* line `insert_line` (0‑based = prepend).
* **append**      – append `new_str` to the file (adds newline when needed).

The single entry‑point is `fs_write(tool: ToolUse) -> ToolResult`.
"""

import os
from pathlib import Path
from typing import Any, Dict

from strands.types.tools import ToolResult, ToolUse

# -----------------------------------------------------------------------------
# Tool specification (copy of the JSON schema provided by Rust side)
TOOL_SPEC: Dict[str, Any] = {
    "name": "fs_write",
    "description": (
        "A tool for creating and editing files\n * The `create` command will override the file at `path` if it already exists as a file, and otherwise create a new file\n * The `append` command will add content to the end of an existing file, automatically adding a newline if the file doesn't end with one. The file must exist.\n Notes for using the `str_replace` command:\n * The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!\n * If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique\n * The `new_str` parameter should contain the edited lines that should replace the `old_str`."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": ["create", "str_replace", "insert", "append"],
            },
            "file_text": {"type": "string"},
            "insert_line": {"type": "integer"},
            "new_str": {"type": "string"},
            "old_str": {"type": "string"},
            "path": {"type": "string"},
        },
        "required": ["command", "path"],
    },
}

# -----------------------------------------------------------------------------
# Helpers

def _expand_path(ctx_cwd: Path, raw: str) -> Path:
    p = Path(os.path.expanduser(raw))
    if not p.is_absolute():
        p = ctx_cwd / p
    return p.absolute()


def _ensure_endswith_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


# -----------------------------------------------------------------------------
# Core implementation

def fs_write(tool: ToolUse, **_kwargs: Any) -> ToolResult:  # noqa: N802 (Rust naming)
    tool_use_id = tool["toolUseId"]
    inp = tool["input"]
    command = inp.get("command")
    cwd = Path(os.getcwd())
    try:
        path = _expand_path(cwd, inp["path"])
    except KeyError:
        raise ValueError("'path' is required")

    try:
        if command == "create":
            content = inp.get("file_text") or inp.get("new_str") or ""
            if not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_ensure_endswith_newline(content))
            payload = {"text": "file created"}

        elif command == "str_replace":
            old_str = inp["old_str"]
            new_str = inp["new_str"]

            if not path.is_file():
                raise ValueError("file must exist for str_replace")

            text = path.read_text()                     # ← read the file first

            count = text.count(old_str)
            if count == 0:
                raise ValueError(f'no occurrences of "{old_str}" found')
            if count > 1:
                raise ValueError(f'{count} occurrences of old_str found; expected exactly 1')

            updated = text.replace(old_str, new_str, 1)
            path.write_text(updated)
            payload = {"text": "replacement done"}

        elif command == "insert":
            insert_line = int(inp["insert_line"])
            new_str = inp["new_str"]
            if not path.is_file():
                raise ValueError("file must exist for insert")
            text = path.read_text()
            lines = text.splitlines(keepends=True)
            # clamp 0..=len(lines)
            insert_line = max(0, min(insert_line, len(lines)))
            char_index = sum(len(l) for l in lines[:insert_line])
            updated = text[:char_index] + new_str + text[char_index:]
            path.write_text(_ensure_endswith_newline(updated))
            payload = {"text": "inserted"}

        elif command == "append":
            new_str = inp["new_str"]
            if not path.is_file():
                raise ValueError("file must exist for append")
            text = path.read_text()
            if not text.endswith("\n"):
                text += "\n"
            text += new_str
            path.write_text(_ensure_endswith_newline(text))
            payload = {"text": "appended"}

        else:
            raise ValueError(f"unknown command '{command}'")

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [payload],
        }

    except Exception as exc:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"fs_write error: {exc}"}],
        }
