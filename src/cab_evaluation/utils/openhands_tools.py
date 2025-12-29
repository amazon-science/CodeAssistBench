"""Custom OpenHands tools for AST-based code reading and editing."""

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Literal

from pydantic import Field

if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState

# Import OpenHands SDK components
try:
    from openhands.sdk.tool import (
        Action,
        Observation,
        ToolAnnotations,
        ToolDefinition,
        ToolExecutor,
        register_tool,
    )
    OPENHANDS_AVAILABLE = True
except ImportError:
    OPENHANDS_AVAILABLE = False
    class Action:
        pass
    class Observation:
        pass
    class ToolDefinition:
        pass
    class ToolExecutor:
        pass
    class ToolAnnotations:
        pass
    def register_tool(name, factory):
        pass

SWE_AGENT_TOOLS = Path("/home/mysoo/CodeStruct/SWE-agent/tools")
READ_CODE_BIN = SWE_AGENT_TOOLS / "ast_read" / "bin" / "readCode"
EDIT_CODE_BIN = SWE_AGENT_TOOLS / "ast_edit" / "bin" / "editCode"

READ_CODE_DESCRIPTION = """Read code with AST parsing and fuzzy search. Language auto-detected from file extension.

PATHS: Supports both relative paths (e.g., "./README.md", "src/main.py") and absolute paths.

BEHAVIOR:
• Small files (<10K chars): Returns FULL content
• Large files (≥10K chars): Returns signatures (function/class names & line numbers)

MOST EFFICIENT USAGE: read_code(path)  # ALWAYS START WITH THIS - gets full content OR signatures
If you get signatures and need specific code:
1. read_code(path, selector="name")     # Read specific function/class
2. read_code(path, line_start=10, line_end=50)  # Read specific line range

CRITICAL RULES:
• NEVER use empty path '' - always specify a file or directory
• ALWAYS call read_code(path) first - it auto-adapts to file size
• Read each file ONCE - never repeat with different parameters
• Use selector or line_start/end ONLY if first call returns signatures
• Selector is just the name: "is_fits" NOT "def is_fits"

EXAMPLES:
read_code("/workspace/small.py")                              # Returns: full content
read_code("/workspace/large_file.py")                         # Returns: signatures only
read_code("/workspace/large_file.py", selector="is_fits")     # Returns: specific function
read_code("/workspace/config.yaml", line_start=1, line_end=50)  # Returns: specific lines
"""

EDIT_CODE_DESCRIPTION = """AST-based code editor. Auto-detects language, handles indentation.

PATHS: Supports both relative paths (e.g., "./README.md", "src/main.py") and absolute paths.

USE edit_code FOR:
• Adding NEW functions or methods (insert_node)
• Deleting entire functions/classes (delete_node)
• Complete function rewrites where >30% of lines change (replace_node)

CRITICAL - AVOID THESE MISTAKES:
• Do NOT include import statements in replacement - they already exist
• Do NOT rewrite entire functions to change 1-2 lines

Operations: replace_node, insert_node, delete_node
Possible Selectors: "ClassName", "functionName", "ClassName.methodName", "start", "end"

EXAMPLES:
• edit_code("/workspace/main.py", "insert_node", "Calculator", replacement="def add(self, a, b):\\n    return a + b")
• edit_code("/workspace/main.py", "delete_node", "oldFunc")
• edit_code("/workspace/main.py", "replace_node", "my_func", replacement="def my_func():\\n    return 42")
"""


if OPENHANDS_AVAILABLE:
    class ReadCodeAction(Action):
        path: str = Field(description="File or directory path (relative or absolute)")
        selector: Optional[str] = Field(default=None, description="Function/class name")
        line_start: Optional[int] = Field(default=None, description="Start line number (1-indexed)")
        line_end: Optional[int] = Field(default=None, description="End line number (1-indexed)")

    class ReadCodeObservation(Observation):
        pass

    class ReadCodeExecutor(ToolExecutor[ReadCodeAction, ReadCodeObservation]):
        def __call__(self, action: ReadCodeAction, conversation=None) -> ReadCodeObservation:
            cmd = ["python3", str(READ_CODE_BIN), action.path]
            if action.selector:
                cmd.extend(["--selector", action.selector])
            if action.line_start:
                cmd.extend(["--line-start", str(action.line_start)])
            if action.line_end:
                cmd.extend(["--line-end", str(action.line_end)])
            
            import os
            env = os.environ.copy()
            ast_lib_path = str(SWE_AGENT_TOOLS / "ast_read" / "lib")
            env["PYTHONPATH"] = f"{ast_lib_path}:{env.get('PYTHONPATH', '')}"
            
            # Get workspace from conversation object
            cwd = os.getcwd()
            if conversation:
                ws = None
                if hasattr(conversation, 'workspace'):
                    ws = conversation.workspace
                elif hasattr(conversation, '_workspace'):
                    ws = conversation._workspace
                elif hasattr(conversation, 'state') and hasattr(conversation.state, 'workspace'):
                    ws = conversation.state.workspace
                # Convert LocalWorkspace to string path
                if ws is not None:
                    cwd = str(ws.working_dir) if hasattr(ws, 'working_dir') else str(ws)
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                                       cwd=cwd, env=env)
                output = result.stdout or result.stderr or "No output"
                is_error = result.returncode != 0
            except subprocess.TimeoutExpired:
                output = "Error: Command timed out after 60 seconds"
                is_error = True
            except Exception as e:
                output = f"Error reading code: {e}"
                is_error = True
            
            return ReadCodeObservation.from_text(output, is_error=is_error)

    class ReadCodeTool(ToolDefinition[ReadCodeAction, ReadCodeObservation]):
        name = "read_code"
        
        @classmethod
        def create(cls, conv_state: "ConversationState") -> Sequence["ReadCodeTool"]:
            return [cls(
                action_type=ReadCodeAction,
                observation_type=ReadCodeObservation,
                description=READ_CODE_DESCRIPTION,
                annotations=ToolAnnotations(title="read_code", readOnlyHint=True,
                                           destructiveHint=False, idempotentHint=True, openWorldHint=False),
                executor=ReadCodeExecutor(),
            )]

    class EditCodeAction(Action):
        path: str = Field(description="File path to modify (relative or absolute)")
        operation: Literal["replace_node", "insert_node", "delete_node"] = Field(description="AST operation type")
        selector: str = Field(description="Target selector")
        replacement: Optional[str] = Field(default=None, description="New code for replace_node/insert_node")

    class EditCodeObservation(Observation):
        pass

    class EditCodeExecutor(ToolExecutor[EditCodeAction, EditCodeObservation]):
        def __call__(self, action: EditCodeAction, conversation=None) -> EditCodeObservation:
            cmd = ["python3", str(EDIT_CODE_BIN), action.path, action.operation, action.selector]
            if action.replacement:
                cmd.extend(["--replacement", action.replacement])
            
            import os
            env = os.environ.copy()
            ast_lib_path = str(SWE_AGENT_TOOLS / "ast_edit" / "lib")
            env["PYTHONPATH"] = f"{ast_lib_path}:{env.get('PYTHONPATH', '')}"
            
            # Get workspace from conversation object
            cwd = os.getcwd()
            if conversation:
                ws = None
                if hasattr(conversation, 'workspace'):
                    ws = conversation.workspace
                elif hasattr(conversation, '_workspace'):
                    ws = conversation._workspace
                elif hasattr(conversation, 'state') and hasattr(conversation.state, 'workspace'):
                    ws = conversation.state.workspace
                # Convert LocalWorkspace to string path
                if ws is not None:
                    cwd = str(ws.working_dir) if hasattr(ws, 'working_dir') else str(ws)
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                                       cwd=cwd, env=env)
                output = result.stdout or result.stderr or "No output"
                is_error = result.returncode != 0
            except subprocess.TimeoutExpired:
                output = "Error: Command timed out after 60 seconds"
                is_error = True
            except Exception as e:
                output = f"Error editing code: {e}"
                is_error = True
            
            return EditCodeObservation.from_text(output, is_error=is_error)

    class EditCodeTool(ToolDefinition[EditCodeAction, EditCodeObservation]):
        name = "edit_code"
        
        @classmethod
        def create(cls, conv_state: "ConversationState") -> Sequence["EditCodeTool"]:
            return [cls(
                action_type=EditCodeAction,
                observation_type=EditCodeObservation,
                description=EDIT_CODE_DESCRIPTION,
                annotations=ToolAnnotations(title="edit_code", readOnlyHint=False,
                                           destructiveHint=True, idempotentHint=False, openWorldHint=False),
                executor=EditCodeExecutor(),
            )]

    def register_ast_tools():
        from openhands.sdk.tool import list_registered_tools
        registered = list_registered_tools()
        if ReadCodeTool.name not in registered:
            register_tool(ReadCodeTool.name, ReadCodeTool)
        if EditCodeTool.name not in registered:
            register_tool(EditCodeTool.name, EditCodeTool)

    register_ast_tools()

else:
    ReadCodeTool = None
    EditCodeTool = None
    def register_ast_tools():
        pass
