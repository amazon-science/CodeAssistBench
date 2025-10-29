from __future__ import annotations

"""gh_issue – Python port that now mirrors the Rust `GhIssue` tool **more strictly**.

Key alignments added
--------------------
* **Side‑effect only**: opens the pre‑filled GitHub URL in the user’s default
  browser (via `webbrowser.open`) and returns an *empty* content list, just as
  the Rust code returns `Default::default()`.
* **Token counts** instead of raw bytes in the `[chat-context]` file listing,
  using a simple whitespace‑token heuristic (`_count_tokens`).
* All markdown sections/newlines built with `\n` for clarity.

Limitations remain the same: the ContextManager interface is expected to be
provided by the host environment; token counting is approximate.
"""

from dataclasses import dataclass, field
from collections import deque
from typing import Any, Deque, Dict, List
from urllib.parse import quote_plus
import webbrowser

from strands.types.tools import ToolResult, ToolUse

# -----------------------------------------------------------------------------
# Constants / limits
MAX_TRANSCRIPT_CHAR_LEN = 3_000
GITHUB_NEW_ISSUE_URL = "https://github.com/aws/amazon-q-developer-cli/issues/new"  # replace with real repo

# -----------------------------------------------------------------------------
# Tool specification
TOOL_SPEC = {
    "name": "report_issue",
    "description": (
        "Opens the browser to a pre-filled gh (GitHub) issue template to report chat issues, bugs, or feature requests. Pre-filled information includes the conversation transcript, chat context, and chat request IDs from the service."    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "expected_behavior": {"type": "string"},
            "actual_behavior": {"type": "string"},
            "steps_to_reproduce": {"type": "string"},
        },
        "required": ["title"],
    },
}

# -----------------------------------------------------------------------------
# Context objects (lightweight mirrors of Rust structs)

@dataclass
class ToolPermission:
    trusted: bool = False


@dataclass
class GhIssueContext:
    context_manager: Any | None = None  # expects same interface as Rust’s
    transcript: Deque[str] = field(default_factory=deque)
    failed_request_ids: List[str] = field(default_factory=list)
    tool_permissions: Dict[str, ToolPermission] = field(default_factory=dict)
    interactive: bool = False


# -----------------------------------------------------------------------------
# Helpers mirroring Rust logic

def _count_tokens(text: str) -> int:
    """Rough approximation of token count (whitespace‑separated)."""
    return len(text.split())


def _trim_transcript(lines: Deque[str]) -> str:
    out: List[str] = []
    remaining = MAX_TRANSCRIPT_CHAR_LEN
    for line in reversed(lines):
        if remaining <= 0:
            break
        seg = line[:remaining]
        remaining -= len(seg)
        out.append(seg.replace("```", r"\```"))
    body = "No chat history found." if not out else "\n\n".join(reversed(out))
    if remaining <= 0:
        body += "\n\n(...truncated)"
    return f"```\n[chat-transcript]\n{body}\n```"


def _format_request_ids(failed_ids: List[str]) -> str:
    ids_block = "none" if not failed_ids else "\n".join(failed_ids)
    return f"[chat-failed_request_ids]\n{ids_block}"


def _gather_chat_settings(ctx: GhIssueContext) -> str:
    rows = [f"interactive={ctx.interactive}", "", "[chat-trusted_tools]"]
    rows.extend(f"\n{tool}={perm.trusted}" for tool, perm in ctx.tool_permissions.items())
    return "[chat-settings]\n" + "".join(rows)


def _gather_context(ctx: GhIssueContext) -> str:
    cm = ctx.context_manager
    if cm is None:
        return "[chat-context]\nNo context available."

    lines: List[str] = ["[chat-context]", f"current_profile={cm.current_profile}\n"]

    profiles = getattr(cm, "list_profiles", lambda: [])()
    lines.append("profiles=none\n" if not profiles else "profiles=\n" + "\n".join(profiles) + "\n")

    gc_paths = getattr(cm.global_config, "paths", [])
    pc_paths = getattr(cm.profile_config, "paths", [])
    lines.append("global_context=none\n" if not gc_paths else "global_context=\n" + "\n".join(gc_paths) + "\n")
    lines.append("profile_context=none\n" if not pc_paths else "profile_context=\n" + "\n".join(pc_paths) + "\n")

    get_ctx_files = getattr(cm, "get_context_files", None)
    files = []
    if callable(get_ctx_files):
        try:
            files = get_ctx_files()
        except Exception:  # silent fail‑open
            files = []

    if files:
        total = 0
        file_rows = []
        for f, content in files:
            size = _count_tokens(content)
            total += size
            file_rows.append(f"{f}, {size} tkns")
        lines.append("files=\n" + "\n".join(file_rows) + f"\ntotal context size={total} tkns")
    else:
        lines.append("files=none")

    return "".join(lines)


# -----------------------------------------------------------------------------
# GitHub URL builder

def _build_issue_url(title: str, body: str) -> str:
    return f"{GITHUB_NEW_ISSUE_URL}?title={quote_plus(title)}&body={quote_plus(body)}"


# -----------------------------------------------------------------------------
# Public tool callable (mirrors Rust `invoke` semantics)

def report_issue(tool: ToolUse, **_kwargs: Any) -> ToolResult:  # noqa: N802
    tid = tool["toolUseId"]
    data = tool["input"]

    title = data["title"]
    expected = data.get("expected_behavior")
    actual = data.get("actual_behavior")
    steps = data.get("steps_to_reproduce")

    ctx: GhIssueContext | None = tool.get("context")  # type: ignore[index]

    additional_env_parts: List[str] = []
    if ctx is not None:
        additional_env_parts.extend([
            _gather_chat_settings(ctx),
            _format_request_ids(ctx.failed_request_ids),
            _gather_context(ctx),
        ])
        transcript = _trim_transcript(ctx.transcript)
    else:
        transcript = "[chat-transcript]\nUnavailable"

    # Build issue body
    body_sections: List[str] = []
    if expected:
        body_sections.append(f"### Expected behavior\n{expected}")
    if actual:
        body_sections.append(f"### Actual behavior\n{actual}\n\n{transcript}")
    else:
        body_sections.append(f"### Transcript\n{transcript}")
    if steps:
        body_sections.append(f"### Steps to reproduce\n{steps}")
    if additional_env_parts:
        body_sections.append("### Additional environment\n" + "\n\n".join(additional_env_parts))

    body = "\n\n".join(body_sections)
    url = _build_issue_url(title, body)

    # Side‑effect (open browser) just like IssueCreator::create_url() in Rust
    try:
        webbrowser.open(url)
    except Exception:
        # Failing to open the browser should not mark the tool as failed
        pass

    # Return success with **no content**, mirroring Rust's Default::default()
    return {"toolUseId": tid, "status": "success", "content": []}
