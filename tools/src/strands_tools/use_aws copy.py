from __future__ import annotations

"""use_aws – Python port of the Rust `UseAws` tool.

Now includes the **TOOL_SPEC** block identical to the provided JSON schema so
that the orchestration layer can introspect the specification directly from
this file.
"""

import json as _json
import os
import re
import subprocess
from typing import Any, Dict, Optional

from strands.types.tools import ToolResult, ToolUse

# -----------------------------------------------------------------------------
# Tool specification
TOOL_SPEC = {
    "name": "use_aws",
    "description": (
        "Make an AWS CLI api call with the specified service, operation, and "
        "parameters. All arguments MUST conform to the AWS CLI specification. "
        "Should the output indicate a malformed command, invoke help to obtain "
        "the correct command."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "service_name": {
                "type": "string",
                "description": (
                    "The name of the AWS service. If you want to query s3, you "
                    "should use s3api if possible."
                ),
            },
            "operation_name": {
                "type": "string",
                "description": "The name of the operation to perform.",
            },
            "parameters": {
                "type": "object",
                "description": (
                    "The parameters for the operation. Keys MUST conform to the "
                    "AWS CLI specification. Prefer JSON syntax; for boolean "
                    "flags use empty string values; prefer kebab‑case."
                ),
            },
            "region": {
                "type": "string",
                "description": "Region name for calling the operation on AWS.",
            },
            "profile_name": {
                "type": "string",
                "description": (
                    "Optional: AWS profile name from ~/.aws/credentials; defaults "
                    "to default profile if omitted."
                ),
            },
            "label": {
                "type": "string",
                "description": "Human‑readable description of the API call.",
            },
        },
        "required": ["region", "service_name", "operation_name", "label"],
    },
}

# -----------------------------------------------------------------------------
# Constants matching Rust side

READONLY_OPS = {
    "get", "describe", "list", "ls", "search", "batch_get",
}
USER_AGENT_ENV_VAR = "AWS_EXECUTION_ENV"
USER_AGENT_APP_NAME = "AmazonQ-For-CLI"
USER_AGENT_VERSION_KEY = "Version"
USER_AGENT_VERSION_VALUE = "0.0.0"  # TODO: inject package version
MAX_TOOL_RESPONSE_SIZE = 64_000

# -----------------------------------------------------------------------------
# Helper: camelCase / snake_case to kebab‑case

def _to_kebab(name: str) -> str:
    name = name.lstrip("-")
    s1 = re.sub("([a-z0-9])([A-Z])", r"\1-\2", name)
    kebab = s1.replace("_", "-").lower()
    return f"--{kebab}"

# -----------------------------------------------------------------------------

class UseAws:
    def __init__(
        self,
        service_name: str,
        operation_name: str,
        region: str,
        parameters: Optional[Dict[str, Any]] = None,
        profile_name: Optional[str] = None,
        label: Optional[str] = None,
    ) -> None:
        self.service_name = service_name
        self.operation_name = operation_name
        self.region = region
        self.parameters = parameters or {}
        self.profile_name = profile_name
        self.label = label

    # ------------------------- parity helpers -------------------------
    def requires_acceptance(self) -> bool:
        return not any(self.operation_name.startswith(op) for op in READONLY_OPS)

    def cli_parameters(self) -> list[tuple[str, str]]:
        params: list[tuple[str, str]] = []
        for key, val in self.parameters.items():
            cli_name = _to_kebab(key)
            cli_val: str
            if isinstance(val, str):
                cli_val = val  # can be empty string flag
            else:
                cli_val = _json.dumps(val)
            params.append((cli_name, cli_val))
        return params

    # ------------------------- invocation ----------------------------
    def invoke(self, tool: ToolUse, **_kwargs: Any) -> ToolResult:
        tid = tool["toolUseId"]

        cmd = ["aws", "--region", self.region]
        if self.profile_name:
            cmd.extend(["--profile", self.profile_name])
        cmd.extend([self.service_name, self.operation_name])
        for name, val in self.cli_parameters():
            cmd.append(name)
            if val:
                cmd.append(val)

        env = os.environ.copy()
        agent_suffix = f"{USER_AGENT_APP_NAME} {USER_AGENT_VERSION_KEY}/{USER_AGENT_VERSION_VALUE}"
        env[USER_AGENT_ENV_VAR] = (
            (env.get(USER_AGENT_ENV_VAR, "") + " " + agent_suffix).strip()
        )

        try:
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        except FileNotFoundError as exc:
            return {
                "toolUseId": tid,
                "status": "error",
                "content": [{"text": f"use_aws error: {exc}"}],
            }

        def _truncate(txt: str) -> str:
            limit = MAX_TOOL_RESPONSE_SIZE // 3
            return txt[:limit] + (" ... truncated" if len(txt) > limit else "")

        stdout_t = _truncate(proc.stdout)
        stderr_t = _truncate(proc.stderr)

        if proc.returncode == 0:
            payload = {
                "exit_status": str(proc.returncode),
                "stdout": stdout_t,
                "stderr": stderr_t,
            }
            return {
                "toolUseId": tid,
                "status": "success",
                "content": [{"json": payload}],
            }
        return {
            "toolUseId": tid,
            "status": "error",
            "content": [{"text": stderr_t or f"AWS CLI exited with status {proc.returncode}"}],
        }


# -----------------------------------------------------------------------------
# Orchestrator‑visible entry‑point

def use_aws(tool: ToolUse, **_kw: Any) -> ToolResult:  # noqa: N802
    inp = tool["input"]
    try:
        cmd = UseAws(
            service_name=inp["service_name"],
            operation_name=inp["operation_name"],
            region=inp["region"],
            parameters=inp.get("parameters"),
            profile_name=inp.get("profile_name"),
            label=inp.get("label"),
        )
    except KeyError as exc:
        return {
            "toolUseId": tool["toolUseId"],
            "status": "error",
            "content": [{"text": f"use_aws error: missing field {exc}"}],
        }

    return cmd.invoke(tool)
