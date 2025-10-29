from __future__ import annotations

"""
fs_read – Python implementation aligned with the Rust tool.

Changes vs. the previous draft
------------------------------
* **Schema** – `path` is no longer globally required (it is optional for
  `Image` mode just like in Rust).
* **Symlink policy** – `_expand_path()` now returns an *absolute* path
  without resolving symlinks, mirroring `sanitize_path_tool_arg()` on the
  Rust side.
* **Early validation** in `Line`, `Search`, and `Directory` modes to match
  Rust's explicit checks and error wording.
* **Search mode size‑guard** – rejects responses that would exceed
  `MAX_TOOL_RESPONSE_SIZE` (64 kB).
* Tiny clean‑ups (removed unused `_truncate`, updated doc‑strings).
"""

import base64
import json
import os
import stat
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from strands.types.tools import ToolResult, ToolUse

# -----------------------------------------------------------------------------
# Constants — match Rust implementation
MAX_TOOL_RESPONSE_SIZE = 64_000
CONTEXT_LINE_PREFIX = "  "
MATCHING_LINE_PREFIX = "→ "
DEFAULT_CONTEXT_LINES = 2
DEFAULT_DEPTH = 0

IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".tif", ".heic",
}

# -----------------------------------------------------------------------------
# Tool specification (used by orchestration layer)
TOOL_SPEC = {
    "name": "fs_read",
    "description": (
        "Tool for reading files (e.g. `cat -n`), directories (`ls -la`) and "
        "images. The behaviour is controlled by the `mode` argument: \n"
        "- Line      : read (optionally ranged) lines from a text file\n"
        "- Directory : recursive long‑format listing\n"
        "- Search    : grep‑like search with context\n"
        "- Image     : validate/collect images (supply `image_paths`)\n"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "path": {
                "description": "Path to a file or directory. Required for all "
                               "modes *except* `Image`. Accepts `~` and "
                               "relative paths.",
                "type": "string",
            },
            # "image_paths": {
            #     "description": "List of paths to images (required for Image mode)",
            #     "type": "array",
            #     "items": {"type": "string"},
            # },
            "mode": {
                "type": "string",
                "enum": ["Line", "Directory", "Search", "Image"],
            },
            "start_line": {
                "type": "integer",
                "description": "Starting line (1‑based, negative = from EOF)",
                "default": 1,
            },
            "end_line": {
                "type": "integer",
                "description": "Ending line (inclusive, 1‑based, negative = from EOF)",
                "default": -1,
            },
            "pattern": {
                "type": "string",
                "description": "Case‑insensitive substring to search for (Search mode)",
            },
            "context_lines": {
                "type": "integer",
                "description": "Context lines around each match (Search mode)",
                "default": DEFAULT_CONTEXT_LINES,
            },
            "depth": {
                "type": "integer",
                "description": "Recursive depth for Directory mode",
                "default": DEFAULT_DEPTH,
            },
        },
        "required": ["mode"],
    },
}

# -----------------------------------------------------------------------------
# Helper functions

def _convert_negative_index(line_count: int, idx: int | None, *, default: int) -> int:
    """Convert 1‑based index (accepting negatives) to 0‑based Python index."""
    if idx is None:
        idx = default
    return max(line_count + idx, 0) if idx <= 0 else idx - 1


def _expand_path(ctx_cwd: Path, raw: str) -> Path:
    """Return an *absolute* path without resolving symlinks (Rust parity)."""
    p = Path(os.path.expanduser(raw))
    if not p.is_absolute():
        p = ctx_cwd / p
    return p.absolute()


def _format_mode(mode: int) -> str:
    table = ("---", "--x", "-w-", "-wx", "r--", "r-x", "rw-", "rwx")
    perm = stat.S_IMODE(mode)
    return "".join(table[(perm >> shift) & 0b111] for shift in (6, 3, 0))


def _format_long_entry(p: Path, md: os.stat_result) -> str:
    ftype = "d" if stat.S_ISDIR(md.st_mode) else "-" if stat.S_ISREG(md.st_mode) else "l"
    mtime = time.strftime("%b %d %H:%M", time.localtime(md.st_mtime))
    return f"{ftype}{_format_mode(md.st_mode)} {md.st_nlink} {md.st_uid} {md.st_gid} {md.st_size:>8} {mtime} {p}"

# -----------------------------------------------------------------------------
# Mode implementations

def _line_mode(path: Path, start_line: int | None, end_line: int | None) -> str:
    if not path.exists():
        raise ValueError(f"'{path}' does not exist")
    if not path.is_file():
        raise ValueError(f"'{path}' is not a file")

    text = path.read_text(errors="replace")
    lines = text.splitlines()
    lc = len(lines)
    start = _convert_negative_index(lc, start_line, default=1)
    end = _convert_negative_index(lc, end_line, default=-1)
    end = max(end, start)  # inclusive range validity

    if start >= lc:
        raise ValueError(
            f"start_line {start_line} is beyond EOF (file has {lc} lines)"
        )

    snippet = "\n".join(lines[start : end + 1])
    if len(snippet.encode()) > MAX_TOOL_RESPONSE_SIZE:
        raise ValueError(
            f"response would exceed {MAX_TOOL_RESPONSE_SIZE} bytes; try a smaller line range"
        )
    return snippet


def _directory_mode(path: Path, depth: int) -> str:
    if not path.exists():
        raise ValueError(f"Directory not found: {path}")
    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    entries: List[str] = []
    queue: List[Tuple[Path, int]] = [(path, 0)]

    while queue:
        current, d = queue.pop(0)
        if d > depth:
            continue
        for child in sorted(current.iterdir()):
            md = child.lstat()
            entries.append(_format_long_entry(child, md))
            if child.is_dir() and d < depth:
                queue.append((child, d + 1))

    listing = "\n".join(entries)
    if len(listing.encode()) > MAX_TOOL_RESPONSE_SIZE:
        raise ValueError(
            f"listing is {len(listing)} bytes, exceeds {MAX_TOOL_RESPONSE_SIZE}"
        )
    return listing


def _search_mode(path: Path, pattern: str, ctx_lines: int) -> str:
    if not path.exists():
        raise ValueError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    if not pattern:
        raise ValueError("pattern cannot be empty")

    content = path.read_text(errors="replace")
    lines = content.splitlines(keepends=True)
    results: List[Dict[str, Union[int, str]]] = []

    pattern_lower = pattern.lower()
    for i, line in enumerate(lines):
        if pattern_lower in line.lower():
            start = max(i - ctx_lines, 0)
            end = min(i + ctx_lines, len(lines) - 1)
            ctx_block = []
            for j in range(start, end + 1):
                prefix = MATCHING_LINE_PREFIX if j == i else CONTEXT_LINE_PREFIX
                ctx_block.append(f"{prefix}{j+1}: {lines[j]}")
            results.append({"line_number": i + 1, "context": "".join(ctx_block)})

    payload = json.dumps(results, ensure_ascii=False)
    if len(payload.encode()) > MAX_TOOL_RESPONSE_SIZE:
        raise ValueError(
            f"response would exceed {MAX_TOOL_RESPONSE_SIZE} bytes; narrow your search or lower context_lines"
        )
    return payload


def _image_mode(image_paths: List[str], ctx_cwd: Path) -> List[str]:
    if not image_paths:
        raise ValueError("'image_paths' cannot be empty for Image mode")

    imgs: List[str] = []
    for raw in image_paths:
        p = _expand_path(ctx_cwd, raw)
        if not p.is_file():
            raise ValueError(f"'{raw}' is not a file")
        if p.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"'{raw}' does not look like a supported image type")
        imgs.append(base64.b64encode(p.read_bytes()).decode())
    return imgs

# -----------------------------------------------------------------------------
# Entry‑point callable by the orchestration layer

def fs_read(tool: ToolUse, **_kwargs: Any) -> ToolResult:  # noqa: N802 (Rust naming)
    tool_use_id = tool["toolUseId"]
    inp = tool["input"]
    ctx_cwd = Path(os.getcwd())

    mode = inp.get("mode")
    try:
        if mode == "Line":
            path = _expand_path(ctx_cwd, inp["path"])
            snippet = _line_mode(path, inp.get("start_line"), inp.get("end_line"))
            payload = {"text": snippet}

        elif mode == "Directory":
            dir_path = _expand_path(ctx_cwd, inp["path"])
            depth = int(inp.get("depth", DEFAULT_DEPTH))
            listing = _directory_mode(dir_path, depth)
            payload = {"text": listing}

        elif mode == "Search":
            file_path = _expand_path(ctx_cwd, inp["path"])
            ctx_lines = int(inp.get("context_lines", DEFAULT_CONTEXT_LINES))
            pattern = inp["pattern"]
            matches_json = _search_mode(file_path, pattern, ctx_lines)
            payload = {"text": matches_json}

        elif mode == "Image":
            images_b64 = _image_mode(inp["image_paths"], ctx_cwd)
            payload = {"image": images_b64}

        else:
            raise ValueError(f"unknown mode '{mode}'")

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [payload],
        }

    except Exception as exc:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"fs_read error: {exc}"}],
        }
