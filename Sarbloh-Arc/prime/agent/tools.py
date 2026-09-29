"""Tool schemas the host offers the model. Upstream Prime Agent exposes one tool: the persistent ``ipython`` REPL."""

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
