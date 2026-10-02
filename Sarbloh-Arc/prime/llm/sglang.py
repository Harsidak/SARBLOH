"""E109, E112: SGLang as a prebuilt runtime for the shared serving layer (``ModelSpec.runtimes``), model-free.

The wheelhouse is an offline Kaggle dataset: ``wheels/`` + ``requirements.lock`` (dfranzen/pennyroyal-v253), or the wheels
and the lock flat at the dataset root (banwait13/sglangwheels, the same Pennyroyal v2.5.3 build); the lock pins sglang
itself, torch, flashinfer, uv and the CUDA 13 toolkit wheels. It is installed once per process into its own venv
in /tmp with the wheelhouse's own ``uv`` (torch 2.13 must not touch the notebook's Python). ``nvidia/cu13`` from those
wheels becomes ``CUDA_HOME``: flashinfer and SGLang JIT-compile kernels at start and need nvcc, libcudart and the
driver's libcuda.so.

``prime.llm.vllm.VllmServer`` launches the server with the returned ``python`` and ``serve`` args and reads
``sglang:*`` metrics (``backend``). Settings follow the documented Pennyroyal launch of the milestone-2 reference
notebook (kaggle/reference/, licence UNCONFIRMED): this file is our own code, written from those settings.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

PREFIX = Path("/tmp/sarbloh-sglang")
SERVE = ["-m", "sglang.launch_server", "--model-path"]


def _log(msg: str) -> None:
    print(f"[sglang {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _run(cmd: list[str], env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess:
    _log("$ " + " ".join(map(str, cmd))[:400])
    out = subprocess.run(list(map(str, cmd)), env=env, capture_output=True, text=True)
    if check and out.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({out.returncode}): {(out.stderr or out.stdout)[-3000:]}")
    return out


def wheelhouse(root: Path) -> tuple[Path, Path]:
    """(wheels dir, requirements.lock): ``wheels/`` + lock at the dataset root or one level down, or both flat."""
    for base in (root, *sorted(p for p in root.iterdir() if p.is_dir())):
        if (base / "wheels").is_dir() and (base / "requirements.lock").is_file():
            return base / "wheels", base / "requirements.lock"
        if (base / "requirements.lock").is_file() and any(base.glob("sglang-*.whl")):
            return base, base / "requirements.lock"
    raise FileNotFoundError(f"no wheels/ + requirements.lock (or flat *.whl + requirements.lock) under {root}")


def precache(paths: list[Path], threads: int = 16, chunk: int = 32 << 20) -> dict[str, Any]:
    """Read every file under ``paths`` once so the page cache holds it: /kaggle/input is a slow network mount and the
    weight load (and the venv install from the wheels) is otherwise bound by it. Run in a background thread."""
    t, jobs = time.time(), []
    for root in paths:
        files = [root] if root.is_file() else [f for f in sorted(root.rglob("*")) if f.is_file()]
        for f in files:
            size = f.stat().st_size
            jobs += [(f, off, min(chunk, size - off)) for off in range(0, max(size, 1), chunk)]

    def read(job: tuple[Path, int, int]) -> int:
        f, off, n = job
        try:
            with f.open("rb") as fh:
                fh.seek(off)
                return len(fh.read(n))
        except OSError:
            return 0

    with ThreadPoolExecutor(threads) as pool:
        total = sum(pool.map(read, jobs))
    info = {"gb": round(total / 1e9, 1), "seconds": round(time.time() - t), "files": len({j[0] for j in jobs})}
    _log(f"precache done: {info}")
    return info


def uv_binary(wheels: Path, dest: Path) -> Path:
    """The Linux ``uv`` executable out of its wheel (no pip, no ensurepip, no internet)."""
    whl = sorted(wheels.glob("uv-*.whl"))
    if len(whl) != 1:
        raise FileNotFoundError(f"expected one uv wheel in {wheels}, found {len(whl)}")
    with zipfile.ZipFile(whl[0]) as z:
        names = [n for n in z.namelist() if Path(n).name == "uv" and not n.endswith("/")]
        if len(names) != 1:
            raise RuntimeError(f"no single uv executable in {whl[0].name}")
        data = z.read(names[0])
    if not data.startswith(b"\x7fELF"):
        raise RuntimeError("the uv in the wheel is not a Linux binary")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    dest.chmod(0o755)
    return dest


def _libcuda() -> Path | None:
    """The driver's libcuda.so.* (the JIT linker needs an unversioned libcuda.so next to libcudart)."""
    try:
        out = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if "libcuda.so" in line and "=>" in line and "x86-64" in line:
                p = Path(line.rsplit("=>", 1)[1].strip())
                if p.exists():
                    return p
    except Exception:  # noqa: BLE001
        pass
    for d in ("/usr/local/nvidia/lib64", "/usr/local/nvidia/lib", "/usr/lib/x86_64-linux-gnu", "/usr/lib64"):
        hits = sorted(Path(d).glob("libcuda.so*"))
        if hits:
            return hits[0]
    return None


def cuda_home(venv: Path) -> Path:
    """``nvidia/cu13`` from the wheels, with the symlinks the JIT linker expects (lib64, *.so, libcuda.so)."""
    hits = sorted(venv.glob("lib/python*/site-packages/nvidia/cu13"))
    if not hits or not (hits[0] / "bin" / "nvcc").is_file():
        raise FileNotFoundError("CUDA 13 toolkit (nvidia/cu13/bin/nvcc) missing from the SGLang venv")
    home = hits[0]
    lib, lib64 = home / "lib", home / "lib64"
    if not lib64.exists() and not lib64.is_symlink():
        lib64.symlink_to("lib")
    for so in sorted(lib.glob("*.so.*")):
        link = lib / re.sub(r"\.so\..*$", ".so", so.name)
        if not link.exists() and not link.is_symlink():
            link.symlink_to(so.name)
    if not (lib / "libcuda.so").exists():
        drv = _libcuda()
        if drv is None:
            raise FileNotFoundError("GPU driver libcuda.so not found")
        if (lib / "libcuda.so").is_symlink():
            (lib / "libcuda.so").unlink()
        (lib / "libcuda.so").symlink_to(drv.resolve())
    return home


def server_env(venv: Path, home: Path) -> dict[str, str]:
    cxx = next((c for c in ("g++-14", "g++-13", "g++-12", "g++-11", "g++") if shutil.which(c)), "g++")
    cache = PREFIX / "cache"
    for name in ("huggingface", "torch", "torchinductor", "triton", "flashinfer", "cuda", "sglang/jit"):
        (cache / name).mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}   # the notebook's site must not leak in
    env.update({
        "PYTHONNOUSERSITE": "1", "PYTHONFAULTHANDLER": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "CUDA_HOME": str(home), "CUDACXX": str(home / "bin" / "nvcc"),
        "PATH": f"{home / 'bin'}:{venv / 'bin'}:" + env.get("PATH", ""),
        "LD_LIBRARY_PATH": f"{home / 'lib64'}:{home / 'lib'}:" + env.get("LD_LIBRARY_PATH", ""),
        "CC": cxx.replace("g++", "gcc"), "CXX": cxx, "CUDAHOSTCXX": cxx, "TORCH_CUDA_ARCH_LIST": "12.0",
        "HF_HOME": str(cache / "huggingface"), "XDG_CACHE_HOME": str(cache), "TORCH_HOME": str(cache / "torch"),
        "CUDA_CACHE_PATH": str(cache / "cuda"), "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
        "TRITON_CACHE_DIR": str(cache / "triton"), "FLASHINFER_WORKSPACE_BASE": str(cache / "flashinfer"),
        "SGLANG_CACHE_DIR": str(cache / "sglang"), "SGLANG_JIT_CACHE_DIR": str(cache / "sglang" / "jit"),
        "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1", "SGLANG_MM_PREPROCESS_DEVICE": "cpu",
        "SGLANG_ENABLE_SM120_LOWM_BF16_GEMM": "1", "SGLANG_NUMA_BIND_V2": "false",
        "SGLANG_SM120_ONLINE_MXFP8": "0", "SGLANG_SM120_LOWM_FP8_WEIGHT": "0", "SGLANG_SM120_LM_HEAD_FP8": "0",
        "SGLANG_MAMBA_CONV_DTYPE": "bfloat16", "CMAKE_BUILD_PARALLEL_LEVEL": "8", "FLASHINFER_NINJA_JOBS": "8",
        "TORCHINDUCTOR_COMPILE_THREADS": "8",
        "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "TOKENIZERS_PARALLELISM": "false", "MAX_JOBS": "8",
        "FLASHINFER_NVCC_THREADS": "2", "NUMPY_MADVISE_HUGEPAGE": "0",
    })
    return env


def relink(wheels: Path, lock: Path, dest: Path) -> Path:
    """A find-links dir of symlinks to ``wheels`` with each local version's ``+`` restored. Kaggle drops ``+`` from
    uploaded file names (``torch-2.13.0cu130-...whl``), so uv cannot match the lock's ``torch==2.13.0+cu130``."""
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    names = {p.name: p for p in wheels.glob("*.whl")}
    for line in lock.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9._-]+)==([^+\s;]+)\+([A-Za-z0-9.]+)", line.strip())
        if not m:
            continue
        norm, ver, local = re.sub(r"[-_.]+", "_", m[1]).lower(), m[2], m[3]
        stripped = f"{norm}-{ver}{local}-"  # how the name looks on Kaggle, e.g. torch-2.13.0cu130-
        for name in [n for n in names if n.lower().startswith(stripped.lower())]:
            fixed = f"{name[:len(norm)]}-{ver}+{local}-{name[len(stripped):]}"
            names[fixed] = names.pop(name)
    for name, src in names.items():
        (dest / name).symlink_to(src)
    return dest


def install(dataset: str, find_input: Callable[[str], Path]) -> Path:
    """The venv's python, installed once (a stamp names the dataset and the sglang wheel)."""
    wheels, lock = wheelhouse(find_input(dataset))
    sgl = sorted(wheels.glob("sglang-*.whl"))
    if len(sgl) != 1:
        raise FileNotFoundError(f"expected one sglang wheel in {wheels}, found {[p.name for p in sgl]}")
    venv, stamp = PREFIX / "venv", PREFIX / ".sarbloh-stamp"
    py = venv / "bin" / "python"
    want = json.dumps([dataset, sgl[0].name])
    if py.exists() and stamp.is_file() and stamp.read_text(encoding="utf-8") == want:
        _log(f"install cached at {venv}")
        return py
    shutil.rmtree(venv, ignore_errors=True)
    env = {**{k: v for k, v in os.environ.items() if k != "PYTHONPATH"}, "UV_OFFLINE": "1",
           "UV_PYTHON_DOWNLOADS": "never", "PYTHONNOUSERSITE": "1", "UV_CACHE_DIR": str(PREFIX / "uv-cache")}
    uv = str(uv_binary(wheels, PREFIX / "bin" / "uv"))
    links = relink(wheels, lock, PREFIX / "links")
    sgl_link = next(links.glob("sglang-*.whl"))  # the one sglang wheel checked above, its "+" restored
    _run([uv, "venv", "--python", sys.executable, str(venv)], env=env)
    _run([uv, "pip", "install", "--python", str(py), "--no-index", "--find-links", str(links), "-r", str(lock)],
         env=env)
    _run([uv, "pip", "install", "--python", str(py), "--no-index", "--find-links", str(links), "--reinstall",
          "--no-deps", str(sgl_link)], env=env)
    stamp.write_text(want, encoding="utf-8")
    return py


def runtime(dataset: str,
            precache_datasets: tuple[str, ...] = ()) -> Callable[[Path, Callable[[str], Path]], dict[str, Any]]:
    """A ``ModelSpec.runtimes`` entry: install the wheelhouse ``dataset`` and return the SGLang server runtime.
    ``precache_datasets`` (the model weights) are read into the page cache in the background meanwhile."""
    def prepare(working_dir: Path, find_input: Callable[[str], Path]) -> dict[str, Any]:
        t = time.time()
        for ds in precache_datasets:
            try:
                threading.Thread(target=precache, args=([find_input(ds)],), name=f"precache-{ds}",
                                 daemon=True).start()
            except Exception as exc:  # noqa: BLE001 - a missing optional input only loses the warm cache
                _log(f"precache {ds} skipped: {exc!r}")
        py = install(dataset, find_input)
        venv = py.parent.parent
        env = server_env(venv, cuda_home(venv))
        versions = ("import sglang, torch, flashinfer; print(sglang.__version__, torch.__version__, "
                    "flashinfer.__version__, torch.cuda.is_available())")
        check = _run([str(py), "-c", versions], env=env)
        return {"env": env, "python": str(py), "serve": list(SERVE), "backend": "sglang",
                "info":{"dataset": dataset, "versions": check.stdout.strip(), "install_s": round(time.time() - t)}}

    return prepare
