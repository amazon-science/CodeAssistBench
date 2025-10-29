from __future__ import annotations

"""thinking – a *no‑op* Python tool that fulfils the internal‑reasoning
contract.

The orchestration layer calls this tool when it needs to store or surface an
intermediate reasoning step.  It simply accepts the `thought` string and
returns success without exposing any user‑visible payload (mirroring the idea
that these thoughts are internal).
"""

from strands.types.tools import ToolResult, ToolUse
from typing import Any

# -----------------------------------------------------------------------------
# Tool specification (verbatim)
TOOL_SPEC = {
    "name": "thinking",
    "description": (
        "Thinking is an internal reasoning mechanism improving the quality of "
        "complex tasks by breaking their atomic actions down; use it "
        "specifically for multi‑step problems requiring step‑by‑step "
        "dependencies, reasoning through multiple constraints, synthesizing "
        "results from previous tool calls, planning intricate sequences of "
        "actions, troubleshooting complex errors, or making decisions involving "
        "multiple trade‑offs. Avoid using it for straightforward tasks, basic "
        "information retrieval, summaries; always clearly define the reasoning "
        "challenge, structure thoughts explicitly, consider multiple "
        "perspectives, and summarize key insights before important decisions or "
        "complex tool interactions."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "thought": {
                "type": "string",
                "description": (
                    "A reflective note or intermediate reasoning step such as "
                    "\"The user needs to prepare their application for production. "
                    "I need to complete three major asks including 1: building "
                    "their code from source, 2: bundling their release artifacts "
                    "together, and 3: signing the application bundle.\""
                ),
            }
        },
        "required": ["thought"],
    },
}

# -----------------------------------------------------------------------------
# Simple data holder

class Thinking:
    def __init__(self, thought: str) -> None:
        self.thought = thought

    def invoke(self, tid: str) -> ToolResult:
        # Internal reasoning: do not expose thought to end‑user.
        return {"toolUseId": tid, "status": "success", "content": []}


# -----------------------------------------------------------------------------
# Entry‑point for the orchestration layer

def thinking(tool: ToolUse, **_kw: Any) -> ToolResult:  # noqa: N802
    tid = tool["toolUseId"]
    inp = tool["input"]
    thought = inp.get("thought")
    if thought is None:
        return {
            "toolUseId": tid,
            "status": "error",
            "content": [{"text": "thinking error: 'thought' field is required"}],
        }

    return Thinking(thought).invoke(tid)
