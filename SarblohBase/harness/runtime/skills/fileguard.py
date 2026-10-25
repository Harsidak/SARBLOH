"""The kernel's file guard: the agent's code may open and list files only in its own game folder, the memory folder
that ``recall()`` reads, the temp folder and the Python libraries, and may not start other programs.

Why: in the 9-hour run the agent walked ``/kaggle/input`` with ``os.walk`` in 10 games, found the game files and spent
up to 113 requests reading them. Installed once by ``BOOT_CODE`` (``runtime/kernel.py``) when the host sets
``SARBLOH_FILE_GUARD=1`` (config ``agent.file_guard``), as a ``sys.addaudithook`` hook, which cannot be removed again.

Allowed roots: the kernel's working folder (its game folder), ``RLM_SESSION_DIR``, ``SARBLOH_MEMORY_DIR``, the temp
folder, ``sys.prefix`` / ``sys.base_prefix`` / ``sys.exec_prefix`` and every ``sys.path`` folder at install time (so
imports work), plus ``/dev``, ``/proc`` and ``/sys`` on POSIX. Denied roots, which win over allowed ones: every folder in
``SARBLOH_GAME_FILES`` (the environments folder, set by ``games.make_arcade``), and the agent's own code folder (the one
that holds ``harness/``, on ``sys.path`` so the kernel can import it): only the import system may read that one. The
4-hour run spent 56 requests reading the harness code (play(), the game wrapper, this guard) from the code dataset.
An allowed root that is the file system root, the home folder, or holds a denied root is dropped. ``os.walk`` swallows
the error and yields nothing; ``open`` and ``os.listdir`` raise ``PermissionError``; ``inspect.getsource`` of a harness
function fails, and a traceback shows no source line for harness frames.
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Any

# Audit events that touch a path, and which of their arguments are paths.
_PATH_ARGS = {"open": (0,), "os.listdir": (0,), "os.scandir": (0,), "os.chdir": (0,), "os.remove": (0,),
              "os.rmdir": (0,), "os.mkdir": (0,), "os.rename": (0, 1), "shutil.copyfile": (0, 1),
              "shutil.copytree": (0, 1), "shutil.rmtree": (0,)}
# Audit events that start another program (a shell could read any file).
_PROGRAM_PREFIXES = ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.startfile",
                     "pty.spawn")

_installed = False
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _norm(path: Any) -> str | None:
    if path is None or isinstance(path, int):
        return None
    try:
        text = os.fsdecode(path)
    except TypeError:
        return None
    return os.path.normcase(os.path.realpath(text))


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _importing() -> bool:
    """True when the import system is the caller (it reads and lists the code folder to import a module)."""
    f = sys._getframe(2)
    while f is not None:
        if f.f_code.co_filename.startswith("<frozen importlib"):
            return True
        f = f.f_back
    return False


def roots() -> tuple[list[str], list[str], list[str]]:
    """(allowed, denied, code) roots, normalised. The code roots are denied too, except to the import system."""
    denied = [n for d in os.environ.get("SARBLOH_GAME_FILES", "").split(os.pathsep) if d.strip()
              if (n := _norm(d.strip()))]
    code = [n for n in (_norm(CODE_ROOT),) if n]
    candidates = [os.getcwd(), os.environ.get("RLM_SESSION_DIR"), os.environ.get("SARBLOH_MEMORY_DIR"),
                  tempfile.gettempdir(), sys.prefix, sys.base_prefix, sys.exec_prefix,
                  *[p or os.getcwd() for p in sys.path]]
    if os.name != "nt":
        candidates += ["/dev", "/proc", "/sys"]
    too_wide = {_norm(os.sep), _norm(os.path.expanduser("~")), _norm(os.path.splitdrive(os.getcwd())[0] + os.sep)}
    allowed: list[str] = []
    for c in candidates:
        n = _norm(c) if c else None
        if not n or n in too_wide or n in allowed or any(_under(d, n) for d in denied):
            continue
        allowed.append(n)
    return allowed, denied, code


def install() -> None:
    """Adds the audit hook. A second call does nothing."""
    global _installed
    if _installed:
        return
    allowed, denied, code = roots()
    busy = [False]

    def allowed_path(n: str) -> bool:
        if not any(_under(n, a) for a in allowed) or any(_under(n, d) for d in denied):
            return False
        return not any(_under(n, c) for c in code) or _importing()

    def hook(event: str, args: tuple) -> None:
        if busy[0]:
            return
        if event.startswith(_PROGRAM_PREFIXES):
            raise PermissionError("starting another program is not allowed here; do the work in Python")
        where = _PATH_ARGS.get(event)
        if where is None:
            return
        busy[0] = True
        try:
            for i in where:
                if i < len(args) and (n := _norm(args[i])) is not None and not allowed_path(n):
                    raise PermissionError(f"{os.fsdecode(args[i])}: your code may only use files in its own folder "
                                          "for this game")
        finally:
            busy[0] = False

    sys.addaudithook(hook)
    _installed = True
