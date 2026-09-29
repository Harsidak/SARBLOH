import os
import  json

def py(**kwargs):
    IPYTHON_TOOL = {
        "type": "function",
        "function": {
            "name": "ipython",
            "description": "Execute Python code in a persistent Python REPL. Top-level `await` is supported. Variables, "
                           "imports, and loaded data persist across calls. Run shell commands with `bash('cmd')` / "
                           "`await bash('cmd')`.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python code to execute in the persistent "
                                                                         "Python REPL."}},
                "required": ["code"],
            },
        },
    }

    return IPYTHON_TOOL