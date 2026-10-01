"""Load an ALTERNATE agent build by file path.

Two things make the alternate builds unimportable by name:
  * their filenames contain a space (`my_agent original.py`), and
  * they are verbatim notebook cells, so line 1 is the `%%writefile` magic.

Both are handled here, in ONE place, so the eval harness and the component tests
cannot drift apart on how they load the same file.
"""
import os
import sys
import hashlib
import tempfile
import importlib.util


def _strip_magics(src: str) -> str:
    """Blank the leading Jupyter magics. Blanked, not deleted: keeping the line
    count identical means a traceback still points at the right source line."""
    lines = src.splitlines(keepends=True)
    i = 0
    while i < len(lines) and (lines[i].lstrip().startswith("%")
                              or lines[i].lstrip().startswith("!")):
        lines[i] = "\n"
        i += 1
    return "".join(lines)


_CACHE = {}


def load_agent_module(path: str):
    """Import `path` as a module and return it. Cached per path: re-executing a
    7k-line agent per game would reset its module-level state and cost seconds."""
    path = os.path.abspath(path)
    if path in _CACHE:
        return _CACHE[path]
    with open(path, encoding="utf-8") as f:
        src = _strip_magics(f.read())

    # Write the cleaned source to a real file and import it normally, so the
    # module gets a genuine __file__/__spec__ and tracebacks stay readable.
    #
    # The pid is part of the name because bench.py runs workers as PROCESSES:
    # keyed on the source path alone, every worker truncates and rewrites the
    # same temp file while its neighbours are exec'ing it, so they import a
    # half-written module and raise `no attribute 'MyAgent'`. One file per
    # process, so there is no shared writable file to race over.
    stem = ("alt_agent_" + hashlib.md5(path.encode()).hexdigest()[:8]
            + "_%d" % os.getpid())
    tmp = os.path.join(tempfile.gettempdir(), stem + ".py")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(src)

    spec = importlib.util.spec_from_file_location(stem, tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[stem] = mod
    spec.loader.exec_module(mod)
    _CACHE[path] = mod
    return mod
