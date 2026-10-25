"""``graph()``: the live map of the code the agent wrote in its kernel. Pre-imported by the kernel.

The map is read from the namespace as it is now, never from old cells, so it cannot go stale: a function that was
deleted, or a name rebound to a plain value, is simply gone from it. A node is a function or a class whose code comes
from a cell (``<cell-N>``); a class lists its methods. A node's edges are the agent's own functions and classes it calls
or names, found with ``ast`` in its source (the runtime keeps every cell's code in ``linecache``). A name the code calls
that is defined nowhere (not local, not in the kernel, not a builtin) is listed as ``missing``: calling it would raise a
NameError.

    graph()            # the whole map: callers first, each followed by what it calls
    graph("solve")     # one node: its signature, what it calls, what calls it, and what it calls that is missing

Host side (``AgentSession``): after every ipython call the host takes a ``snapshot()``, compares it with the one before
(``delta``) and adds one line to that call's output: ``+`` a new node, ``~`` a changed one, ``-`` a removed one, and
``!`` a node that now calls something missing. After a drain the whole map (``render``) goes back into the context once.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import inspect
import sys
import textwrap
from typing import Any

SIG_CHARS = 60          # a signature longer than this is cut
DELTA_CHARS = 500       # the change line of one call
RENDER_CHARS = 4000     # the whole map
_BUILTINS = set(dir(builtins))
_CACHE: dict[Any, dict[str, Any]] = {}   # code object -> what its source says (a code object never changes)


def _ns() -> dict[str, Any]:
    return sys.modules["__main__"].__dict__


def _from_cell(fn: Any) -> bool:
    code = getattr(fn, "__code__", None)
    return code is not None and str(code.co_filename).startswith("<cell-")


def _function(v: Any) -> Any:
    """The plain function behind ``v`` (through decorators such as ``functools.lru_cache``), or None."""
    try:
        v = inspect.unwrap(v)
    except ValueError:
        return None
    return v if inspect.isfunction(v) and _from_cell(v) else None


def _signature(name: str, fn: Any) -> str:
    try:
        sig = str(inspect.signature(fn))
    except (TypeError, ValueError):
        sig = "(...)"
    return name + (sig if len(sig) <= SIG_CHARS else sig[:SIG_CHARS - 4] + "...)")


def _locals(tree: ast.AST) -> set[str]:
    """Every name bound inside the code: arguments, assignments, loop and comprehension targets, nested functions and
    classes, imports and ``except ... as`` names (minus the names it declares ``global``)."""
    bound: set[str] = set()
    declared: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.arg):
            bound.add(n.arg)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            bound.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name).split(".")[0] for a in n.names)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, ast.Global):
            declared.update(n.names)
    return bound - declared


def _read(fn: Any) -> dict[str, Any]:
    """What the source of ``fn`` says: the names it calls, the names it reads, the methods it calls on ``self``, its
    locals, its first docstring line, its size and a hash of the code. Without source: the code object's names."""
    code = fn.__code__
    if code in _CACHE:
        return _CACHE[code]
    try:
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
    except (OSError, TypeError, SyntaxError, IndentationError):
        src, tree = None, None
    if tree is None:
        names = set(code.co_names)
        info = {"called": names, "named": names, "self_calls": set(), "locals": set(code.co_varnames),
                "doc": "", "lines": 0, "hash": hashlib.blake2b(code.co_code, digest_size=4).hexdigest()}
    else:
        node = tree.body[0] if tree.body else tree
        body = node.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else [node]
        called, named, self_calls = set(), set(), set()
        for stmt in body:
            for n in ast.walk(stmt):
                if isinstance(n, ast.Call):
                    if isinstance(n.func, ast.Name):
                        called.add(n.func.id)
                    elif isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) \
                            and n.func.value.id in ("self", "cls"):
                        self_calls.add(n.func.attr)
                elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                    named.add(n.id)
        doc = ast.get_docstring(node) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None
        info = {"called": called, "named": named | called, "self_calls": self_calls, "locals": _locals(node),
                "doc": ((doc or "").strip().splitlines() or [""])[0][:80], "lines": src.count("\n") + 1,
                "hash": hashlib.blake2b(src.encode(), digest_size=4).hexdigest()}
    _CACHE[code] = info
    return info


def _methods(cls: type) -> dict[str, Any]:
    out = {}
    for k, v in vars(cls).items():
        if isinstance(v, (staticmethod, classmethod)):
            v = v.__func__
        elif isinstance(v, property):
            v = v.fget
        fn = _function(v) if v is not None else None
        if fn is not None:
            out[k] = fn
    return out


def snapshot() -> dict[str, dict[str, Any]]:
    """The map now, as plain data: {name: {"kind", "sig", "calls", "missing", "hash", "lines", "doc"}} (a class also
    has "methods"). Names starting with "_" and the names the kernel was booted with are left out."""
    ns = _ns()
    boot = ns.get("_BOOT_NAMES") or set()
    found: dict[str, tuple[str, Any]] = {}
    for k, v in list(ns.items()):
        if k.startswith("_") or k in boot:
            continue
        if inspect.isclass(v) and getattr(v, "__module__", None) == "__main__":   # defined in a cell
            found[k] = ("class", v)
        else:
            fn = _function(v)
            if fn is not None:
                found[k] = ("function", fn)
    nodes: dict[str, dict[str, Any]] = {}
    for k, (kind, v) in found.items():
        fns = list(_methods(v).items()) if kind == "class" else [(k, v)]
        refs, missing, hashes, lines = set(), set(), [], 0
        for name, fn in fns:
            info = _read(fn)
            refs |= info["named"]
            missing |= {n for n in info["called"] - info["locals"]
                        if n not in ns and n not in _BUILTINS and n not in found}
            hashes.append(info["hash"])
            lines += info["lines"]
        if kind == "class":
            try:
                src = inspect.getsource(v)
                h = hashlib.blake2b(src.encode(), digest_size=4).hexdigest()
                lines = src.count("\n") + 1
            except (OSError, TypeError):   # no source for the class itself: its methods and attribute names
                h = hashlib.blake2b(("".join(hashes) + ",".join(sorted(vars(v)))).encode(),
                                    digest_size=4).hexdigest()
            node = {"kind": "class", "sig": f"class {k}", "methods": sorted(n for n, _ in fns), "hash": h,
                    "doc": (inspect.getdoc(v) or "").strip().splitlines()[0][:80] if inspect.getdoc(v) else ""}
        else:
            info = _read(v)
            node = {"kind": "function", "sig": _signature(k, v), "hash": hashes[0], "doc": info["doc"]}
        node.update({"calls": sorted(n for n in refs if n in found and n != k), "missing": sorted(missing),
                     "lines": lines})
        nodes[k] = node
    return nodes


def _callers(nodes: dict[str, dict[str, Any]], name: str) -> list[str]:
    return sorted(k for k, n in nodes.items() if name in n["calls"])


def _head(n: dict[str, Any]) -> str:
    head = n["sig"] + (f" [{', '.join(n['methods'])}]" if n["kind"] == "class" else "")
    return head + (f" -> {', '.join(n['calls'])}" if n["calls"] else "")


def _changes(before: dict[str, dict[str, Any]], now: dict[str, dict[str, Any]]) -> tuple[list[str], list[str],
                                                                                         list[str]]:
    """The names added, changed (its code, signature or edges) and removed between two snapshots."""
    gone = [k for k in before if k not in now]

    def changed(k: str) -> bool:   # its own code changed, or it calls something new (a removed callee is a "!")
        a, b = before[k], now[k]
        return any(a.get(f) != b.get(f) for f in ("hash", "sig", "methods")) or \
            set(b["calls"]) != set(a["calls"]) - set(gone)

    return [k for k in now if k not in before], [k for k in now if k in before and changed(k)], gone


def delta(before: dict[str, dict[str, Any]] | None, now: dict[str, dict[str, Any]],
          limit: int = DELTA_CHARS) -> str:
    """One line of what changed between two snapshots, or "" when nothing did: ``+`` added, ``~`` changed (its code,
    signature or edges), ``-`` removed, ``!`` a node that now calls a missing name."""
    before = before or {}
    added, changed, gone = _changes(before, now)
    parts = [f"+{_head(now[k])}" for k in added]
    parts += [f"~{_head(now[k])}" for k in changed]
    parts += [f"-{k}" for k in gone]
    for k, n in now.items():
        new = [m for m in n["missing"] if m not in (before.get(k) or {}).get("missing", [])]
        if new:
            parts.append(f"! {k} calls missing {', '.join(new)}")
    if not parts:
        return ""
    line, shown = "", 0
    for p in parts:
        if len(line) + len(p) + 3 > limit:
            break
        line, shown = (f"{line} | {p}" if line else p), shown + 1
    more = f" | ... {len(parts) - shown} more: graph() shows all" if shown < len(parts) else ""
    return f"[code graph: {line}{more}]"


def cell_map(before: dict[str, dict[str, Any]] | None, now: dict[str, dict[str, Any]],
             limit: int = RENDER_CHARS) -> str:
    """What one cell defined, changed or removed, one line per function or class as it is now (its signature, what it
    calls and its docstring), or "" when it touched none. The stored reply holds this in place of the cell's code."""
    before = before or {}
    added, changed, gone = _changes(before, now)
    out = []
    for k in added + changed:
        doc = f"  # {now[k]['doc']}" if now[k].get("doc") else ""
        out.append(_head(now[k]) + doc)
        if now[k]["missing"]:
            out.append(f"! {k} calls missing {', '.join(now[k]['missing'])}")
    out += [f"removed {k}" for k in gone]
    text = "\n".join(out)
    if len(text) > limit:
        text = text[:limit].rsplit("\n", 1)[0] + "\n... (cut: graph() shows all)"
    return text


def render(nodes: dict[str, dict[str, Any]], limit: int = RENDER_CHARS) -> str:
    """The whole map as an indented tree: every node that nothing else calls first, each followed by what it calls; a
    node already shown is named again with "(above)". Then the missing names."""
    if not nodes:
        return "code graph: no functions or classes defined yet"
    n_fn = sum(1 for n in nodes.values() if n["kind"] == "function")
    out = [f"code graph: {n_fn} functions, {len(nodes) - n_fn} classes (a name, then what it calls, indented)"]
    shown: set[str] = set()

    def walk(k: str, depth: int) -> None:
        if k in shown:
            out.append("  " * depth + f"{k} (above)")
            return
        shown.add(k)
        n = nodes[k]
        doc = f"  # {n['doc']}" if n.get("doc") else ""
        out.append("  " * depth + (n["sig"] + (f" [{', '.join(n['methods'])}]" if n["kind"] == "class" else "")) + doc)
        for c in n["calls"]:
            walk(c, depth + 1)

    roots = sorted(k for k in nodes if not _callers(nodes, k))
    for k in roots + sorted(nodes):   # then anything only reachable through a cycle
        if k not in shown:
            walk(k, 0)
    out += [f"! {k} calls missing {', '.join(n['missing'])}" for k, n in sorted(nodes.items()) if n["missing"]]
    text = "\n".join(out)
    if len(text) > limit:
        text = text[:limit].rsplit("\n", 1)[0] + "\n... (cut: graph(\"name\") shows one node)"
    return text


def describe(nodes: dict[str, dict[str, Any]], name: str) -> str:
    """One node: its signature, size, docstring, what it calls, what calls it and what it calls that is missing."""
    n = nodes.get(name)
    if n is None:
        close = sorted(k for k in nodes if name.lower() in k.lower())
        return f"graph: {name!r} is not a function or class in your kernel now" + \
            (f" (similar: {', '.join(close)})" if close else "")
    out = [f"{n['sig']}  ({n['lines']} lines)" + (f"  # {n['doc']}" if n.get("doc") else "")]
    if n["kind"] == "class":
        out.append(f"methods: {', '.join(n['methods'])}")
    out.append(f"calls: {', '.join(n['calls']) or '(none of your functions)'}")
    out.append(f"called by: {', '.join(_callers(nodes, name)) or '(nothing)'}")
    if n["missing"]:
        out.append(f"missing: {', '.join(n['missing'])} (not defined anywhere: calling it raises NameError)")
    out.append(f'code: show_source("{name}")')
    return "\n".join(out)


def graph(name: Any = None) -> None:
    """Print the map of your functions and classes as they are in the kernel now, or one of them with its callers and
    callees (give its name as text, or the function). Free: it reads the kernel, not the game."""
    nodes = snapshot()
    if name is None:
        print(render(nodes))
    else:
        print(describe(nodes, name if isinstance(name, str) else getattr(name, "__name__", str(name))))
