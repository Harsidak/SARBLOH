"""``show_source(name)``: print the code of something the agent defined in an earlier cell. Pre-imported by the kernel.

The runtime keeps every cell's code in ``linecache`` under ``<cell-N>`` (``rlm/repl.py``), so ``inspect.getsource``
works for functions and classes; for a variable (or anything else) the newest cell that assigned the name is searched.
Used when old calls' code is replaced by a short note in the context (config ``stub_old_code``).
"""

from __future__ import annotations

import ast
import inspect
import linecache
import re
from typing import Any

_CELL = re.compile(r"^<cell-(\d+)>$")


def _defines(node: ast.stmt, name: str) -> bool:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name == name
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return any(isinstance(n, ast.Name) and n.id == name for t in targets for n in ast.walk(t))
    return False


def _from_cells(name: str) -> str | None:
    """The newest top-level statement that defined ``name``, from the cells' code."""
    cells = sorted(((int(m.group(1)), key) for key in list(linecache.cache)
                    if isinstance(key, str) and (m := _CELL.match(key))), reverse=True)
    for _, key in cells:
        code = "".join(linecache.cache[key][2])
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        hits = [n for n in tree.body if _defines(n, name)]
        if hits:
            return "\n\n".join(ast.get_source_segment(code, n) or ast.unparse(n) for n in hits)
    return None


def show_source(name: Any) -> None:
    """Print the current code of a function, class or variable you defined (give its name as a string, or the
    object). Free: it reads the kernel, not the game."""
    obj, label = name, getattr(name, "__name__", None)
    if isinstance(name, str):
        label = name
        caller = inspect.currentframe().f_back
        obj = caller.f_globals.get(name, caller.f_locals.get(name)) if caller is not None else None
    text = None
    if obj is not None and (inspect.isfunction(obj) or inspect.isclass(obj) or inspect.ismethod(obj)):
        try:
            text = inspect.getsource(obj)
        except (OSError, TypeError):
            text = None
    if text is None and label:
        text = _from_cells(label)
    print(text.rstrip("\n") if text else f"show_source: no code found for {label or name!r} in your earlier cells")
