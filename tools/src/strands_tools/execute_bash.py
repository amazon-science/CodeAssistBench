import os
import shlex
import subprocess
from typing import Any, Dict, List

from strands.types.tools import ToolResult, ToolUse

# ---- Spec ----
TOOL_SPEC = {
    "name": "execute_bash",
    "description": "Execute the specified bash command.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Bash command to execute",
            },
            "summary": {
                "type": "string",
                "description": "Optional short description of why the command is run",
            },
        },
        "required": ["command"],
    },
}

# ---- Validation helpers (ported from Rust) ----
READONLY_COMMANDS: List[str] = [
    "ls", "cat", "echo", "pwd", "which", "head", "tail", "find", "grep"
]
DANGEROUS_PATTERNS: List[str] = ["<(", "$(", "`", ">", "&&", "||", "&", ";"]


def _requires_acceptance(cmd: str) -> bool:
    try:
        args = shlex.split(cmd)
    except ValueError:  # malformed quoting
        return True

    # obvious red‑flags
    if any(p in a for a in args for p in DANGEROUS_PATTERNS):
        return True

    # split the pipeline into sub‑commands
    pipeline: List[List[str]] = []
    current: List[str] = []
    for a in args:
        if a == "|":
            if current:
                pipeline.append(current)
            current = []
        elif "|" in a:  # `echo foo|rm bar`
            return True
        else:
            current.append(a)
    if current:
        pipeline.append(current)

    # each segment must start with a whitelisted command and,
    # for `find`, must not contain exec/delete
    for seg in pipeline:
        if not seg:
            return True
        first = seg[0]
        if first == "find" and any("-exec" in s or "-delete" in s for s in seg):
            return True
        if first not in READONLY_COMMANDS:
            return True
    return False


# ---- Core executor ----
def _execute_bash_command(command: str) -> Dict[str, Any]:
    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        executable="/bin/bash",
        universal_newlines=True,
    )
    stdout, stderr = proc.communicate()
    return {
        "exit_status": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


# ---- Tool entry‑point ----
def execute_bash(tool: ToolUse, **kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    inp = tool["input"]

    command: str = inp["command"]
    summary: str | None = inp.get("summary")

    # Check if unrestricted mode is enabled via environment variable
    allow_unrestricted = os.getenv("EXECUTE_BASH_UNRESTRICTED", "false").lower() == "true"
    
    # safety gate (bypass if unrestricted mode is enabled)
    if not allow_unrestricted and _requires_acceptance(command):
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{
                "text": "Refused: command requires explicit acceptance or is not read‑only."
            }],
        }

    try:
        result = _execute_bash_command(command)
        status = "success" if result["exit_status"] == 0 else "error"

        # Rust returns JSON; we embed the same dict in content
        return {
            "toolUseId": tool_use_id,
            "status": status,
            "content": [{
                "json": {
                    "exit_status": result["exit_status"],
                    "stdout": result["stdout"],
                    "stderr": result["stderr"],
                }
            }],
        }

    except Exception as e:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{
                "text": f"Error executing bash command: {e}"
            }],
        }
