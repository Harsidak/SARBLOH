"""Build the sarbloh Kaggle dataset and push it (and the launcher notebook) with the Kaggle CLI.

    uv run python scripts/kaggle_push.py build                 # stage kaggle/datasets/sarbloh-agent/sarbloh, no upload
    uv run python scripts/kaggle_push.py dataset -m "E001 ..."  # build + new dataset version
    uv run python scripts/kaggle_push.py kernel                 # push kaggle/main.ipynb (starts a run)
    uv run python scripts/kaggle_push.py status | output        # poll the run / download its outputs
    uv run python scripts/kaggle_push.py prime                  # push kaggle/experimental/002_prime.ipynb (Prime Agent)
    uv run python scripts/kaggle_push.py prime-status | prime-output
    uv run python scripts/kaggle_push.py experimentation        # push kaggle/experimental/sarbloh_experimentation.ipynb
    uv run python scripts/kaggle_push.py experimentation-status | experimentation-output
    uv run python scripts/kaggle_push.py deepseek               # kaggle/experimental/deepseek_experimentation.ipynb (DeepSeek-V4)
    uv run python scripts/kaggle_push.py deepseek-status | deepseek-output
    uv run python scripts/kaggle_push.py wheels                 # upload kaggle/datasets/sarbloh-wheels-gemma4
    uv run python scripts/kaggle_push.py wheels-build           # run kaggle/wheels.ipynb (CPU, internet on): vLLM wheelhouse
    uv run python scripts/kaggle_push.py wheels-build-status | wheels-build-output
    uv run python scripts/kaggle_push.py vllm-bench             # E007: push kaggle/experimental/007_vllm_bench.ipynb
    uv run python scripts/kaggle_push.py vllm-bench-status | vllm-bench-output
    uv run python scripts/kaggle_push.py alignment              # Alignment/alignment.ipynb (LoRA SFT + RL) on a T4
    uv run python scripts/kaggle_push.py alignment-status | alignment-output
    uv run python scripts/kaggle_push.py alignment-sync         # only refresh the notebook's generated source cells

Credentials: the repo .env (gitignored, loaded automatically), ~/.kaggle/kaggle.json, or KAGGLE_* env vars. The dataset holds only authored code plus the
vendored Duck baseline; model weights and the vLLM wheelhouse are separate, already-public datasets.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KAGGLE = REPO / "kaggle"
DATASET_DIR = KAGGLE / "datasets" / "sarbloh-agent"  # holds LICENSE + dataset-metadata.json; sarbloh/ is rebuilt
KERNEL_DIR = KAGGLE / ".push" / "main"
NOTEBOOK = KAGGLE / "main.ipynb"
SETTINGS = REPO / "scripts" / "kaggle_settings.json"
EXCLUDE = {"legacy", "__pycache__"}


def load_dotenv(path: Path = REPO / ".env") -> None:
    """KEY=value lines from the gitignored repo .env; real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#") and value.strip():
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def settings() -> dict:
    s = json.loads(SETTINGS.read_text(encoding="utf-8"))
    if "<" in s["username"]:
        sys.exit(f"Set your Kaggle username in {SETTINGS}")
    return s


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def build() -> Path:
    shutil.rmtree(DATASET_DIR / "sarbloh", ignore_errors=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        REPO / "sarbloh",
        DATASET_DIR / "sarbloh",
        ignore=lambda d, names: [n for n in names if n in EXCLUDE or n.endswith((".pyc", ".pkl"))],
    )
    dirty = bool(git("status", "--porcelain", "--", "sarbloh"))
    marker = {
        "schema_version": 1,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git("rev-parse", "--short", "HEAD"),
        "dirty": dirty,
    }
    (DATASET_DIR / "sarbloh-bundle.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    files = sum(1 for p in DATASET_DIR.rglob("*") if p.is_file())
    print(f"built {DATASET_DIR} ({files} files) {marker}")
    if dirty:
        print("WARNING: sarbloh/ has uncommitted changes; the run will not map to a git SHA exactly.")
    return DATASET_DIR


def kaggle(*args: str) -> None:
    if shutil.which("kaggle") is None:
        sys.exit("kaggle CLI not found: uv sync (it is in the default groups) and run via uv run")
    print("$ kaggle " + " ".join(args), flush=True)
    # the CLI echoes kernel logs; on a cp1252 console a non-ASCII char kills the output download
    subprocess.run(["kaggle", *args], check=True, env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})


def check_inputs(datasets: list[str]) -> None:
    """Refuse to push when an input dataset is unreachable: Kaggle drops it from the kernel silently and the run
    fails later (2026-10-01: driessmit1/arc3-vllm-h100-wheelhouse-v3 went 403 and vLLM had nothing to install)."""
    bad = [d for d in datasets if subprocess.run(["kaggle", "datasets", "files", d, "--page-size", "1"],
                                                 capture_output=True, text=True).returncode != 0]
    if bad:
        sys.exit(f"input datasets unreachable (deleted, private or misspelled), not pushing: {bad}")


def push_dataset(message: str) -> None:
    s = settings()
    build()
    ref = f"{s['username']}/{s['dataset_slug']}"
    meta_path = DATASET_DIR / "dataset-metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["id"] = ref
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    exists = subprocess.run(["kaggle", "datasets", "status", ref], capture_output=True, text=True).returncode == 0
    if exists:
        kaggle("datasets", "version", "-p", str(DATASET_DIR), "-m", message, "-r", "zip")
    else:
        kaggle("datasets", "create", "-p", str(DATASET_DIR), "-r", "zip")
    for _ in range(60):  # the kernel must not start before the new version is ready
        out = subprocess.run(["kaggle", "datasets", "status", ref], capture_output=True, text=True).stdout.strip()
        print(f"dataset status: {out}", flush=True)
        if out == "ready":
            return
        time.sleep(10)
    sys.exit("dataset did not become ready")


def push_kernel() -> None:
    s = settings()
    shutil.rmtree(KERNEL_DIR, ignore_errors=True)
    KERNEL_DIR.mkdir(parents=True)
    shutil.copy(NOTEBOOK, KERNEL_DIR / NOTEBOOK.name)
    meta = {
        "id": f"{s['username']}/{s['kernel_slug']}",
        "title": s["kernel_title"],
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        # main.ipynb (the Duck launch) does not import sarbloh yet; attach the agent dataset only when asked.
        "dataset_sources": [*([f"{s['username']}/{s['dataset_slug']}"] if s.get("attach_sarbloh") else []),
                            *s["extra_datasets"]],
        "kernel_sources": [],
        "model_sources": s.get("model_sources", []),
        "machine_shape": s["machine_shape"],
    }
    (KERNEL_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    check_inputs(meta["dataset_sources"])
    kaggle("kernels", "push", "-p", str(KERNEL_DIR))


def push_prime(key: str = "prime") -> None:
    """Prime Agent notebook: its own kernel slug and inputs (settings[key]); the agent code comes from the
    git-linked dataset the owner updates, not from build(). key "experimentation" runs Sarbloh-Experimentation, and
    "deepseek" runs it on DeepSeek-V4-Flash."""
    s = settings()
    p = s[key]
    kdir = KAGGLE / ".push" / key
    shutil.rmtree(kdir, ignore_errors=True)
    kdir.mkdir(parents=True)
    nb = REPO / p["notebook"]
    shutil.copy(nb, kdir / nb.name)
    meta = {
        "id": f"{s['username']}/{p['kernel_slug']}", "title": p["kernel_title"], "code_file": nb.name,
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
        "enable_tpu": False, "enable_internet": False, "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "dataset_sources": p["datasets"], "kernel_sources": [], "model_sources": p.get("model_sources", []),
        "machine_shape": s["machine_shape"],
    }
    if p.get("docker_image"):
        # the cp312 SGLang 0.5.20 wheelhouse needs the Python 3.12 image (see push_vllm_bench)
        meta["docker_image"] = p["docker_image"]
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    check_inputs(meta["dataset_sources"])
    check_models(meta["model_sources"])
    kaggle("kernels", "push", "-p", str(kdir))


def push_vllm_bench(key: str = "vllm_bench", root: str = "Sarbloh") -> None:
    """A serving bench (E007, or the ``dsv4_probe`` server probe). ``root``/harness/llm (plus the harness modules it
    imports) is embedded into the notebook (SOURCES dict), so the run tests the working tree without a sarblohagent
    dataset update; its git SHA and dirty flag are printed by the notebook."""
    s = settings()
    b = s[key]
    kdir = KAGGLE / ".push" / key
    shutil.rmtree(kdir, ignore_errors=True)
    kdir.mkdir(parents=True)
    prime = REPO / root / "harness"
    files = [prime / "__init__.py", *sorted(prime.glob("sysmon.py")), *sorted((prime / "llm").glob("*.py"))]
    sources = {f.relative_to(prime.parent).as_posix(): f.read_text(encoding="utf-8") for f in files}
    meta = {"git_sha": git("rev-parse", "--short", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--", f"{root}/harness/llm")),
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "files": sorted(sources)}
    nb_path = REPO / b["notebook"]
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    marker = "# @@PRIME_LLM_SOURCES@@"
    hits = 0
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if marker in src:
            # repr(), not json.dumps: the cell is Python, and JSON's true/false/null are NameErrors there
            src = src.replace(marker, f"SOURCES = {sources!r}\nSOURCES_META = {meta!r}")
            cell["source"] = src.splitlines(True)
            hits += 1
    if hits != 1:
        sys.exit(f"{nb_path}: expected one {marker!r} cell, found {hits}")
    (kdir / nb_path.name).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    kmeta = {
        "id": f"{s['username']}/{b['kernel_slug']}", "title": b["kernel_title"], "code_file": nb_path.name,
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
        "enable_tpu": False, "enable_internet": False,
        # the RTX Pro 6000 is reserved for ARC-AGI-3: without the competition attached, v2 got 2x T4 (UNCONFIRMED cause)
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "dataset_sources": b["datasets"], "kernel_sources": [], "model_sources": b.get("model_sources", []),
        "machine_shape": s["machine_shape"],
    }
    if b.get("docker_image"):
        # a kernel without one gets Kaggle's newest image (Python 3.13 on 2026-10-26), where the cp312 wheelhouses
        # cannot install; the pin keeps the image sarbloh-experimentation ran SGLang on
        kmeta["docker_image"] = b["docker_image"]
    (kdir / "kernel-metadata.json").write_text(json.dumps(kmeta, indent=2), encoding="utf-8")
    print(f"embedded {len(sources)} files {meta}")
    check_inputs(kmeta["dataset_sources"])
    check_models(kmeta["model_sources"])
    kaggle("kernels", "push", "-p", str(kdir))


def check_models(models: list[str]) -> None:
    """Every model_sources ref (owner/model/framework/variation/version) must list its files. An unchecked ref is
    UNCONFIRMED as the cause of the three dsv4 probes that died at start with an empty log."""
    bad = [m for m in models if subprocess.run(["kaggle", "models", "instances", "versions", "files", m],
                                               capture_output=True, text=True).returncode != 0]
    if bad:
        sys.exit(f"model sources unreachable (wrong framework/variation/version?), not pushing: {bad}")


def push_glm(key: str = "glm", root: str = "Sarbloh-Experimentation") -> None:
    """GLM-5.3-Flash on llama.cpp: settings[key]'s notebook with the whole ``root`` tree (harness/ + rlm/) embedded
    as a base64 tar.gz, so the run tests the working tree without a sarblohagent dataset update."""
    import base64
    import io
    import tarfile

    s = settings()
    b = s[key]
    kdir = KAGGLE / ".push" / key
    shutil.rmtree(kdir, ignore_errors=True)
    kdir.mkdir(parents=True)
    files = sorted(f for f in (REPO / root).rglob("*")
                   if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for f in files:
            tf.add(f, arcname=f.relative_to(REPO).as_posix())
    meta = {"git_sha": git("rev-parse", "--short", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", root)),
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "files": [f.relative_to(REPO).as_posix() for f in files]}
    nb_path = REPO / b["notebook"]
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    marker = "# @@AGENT_SOURCE@@"
    hits = 0
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if marker in src:
            blob = base64.b64encode(buf.getvalue()).decode("ascii")
            cell["source"] = src.replace(marker, f"AGENT_TGZ = {blob!r}\nAGENT_SOURCE_META = {meta!r}").splitlines(True)
            hits += 1
    if hits != 1:
        sys.exit(f"{nb_path}: expected one {marker!r} cell, found {hits}")
    (kdir / nb_path.name).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    kmeta = {
        "id": f"{s['username']}/{b['kernel_slug']}", "title": b["kernel_title"], "code_file": nb_path.name,
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
        "enable_tpu": False, "enable_internet": False, "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "dataset_sources": b["datasets"], "kernel_sources": [], "model_sources": b.get("model_sources", []),
        "machine_shape": s["machine_shape"],
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(kmeta, indent=2), encoding="utf-8")
    print(f"embedded {len(files)} files, {len(buf.getvalue()) / 1e6:.2f} MB gzip, "
          f"{ {k: v for k, v in meta.items() if k != 'files'} }")
    check_inputs(kmeta["dataset_sources"])
    check_models(kmeta["model_sources"])
    kaggle("kernels", "push", "-p", str(kdir))


ALIGN_MARK = "# @@ALIGNMENT_SOURCES@@"
ALIGN_WRITE = "%%writefile alignment_src/"


def _cell(cid: str, kind: str, text: str) -> dict:
    c = {"cell_type": kind, "id": cid, "metadata": {}, "source": text.splitlines(True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def sync_alignment(nb_path: Path) -> dict:
    """Refresh, in place, the notebook's generated source cells: one marker cell (makes the dirs, writes
    SOURCES_META.json) and one ``%%writefile alignment_src/<file>`` cell per Alignment/lora_*.py and data/*.jsonl.
    Everything else in the notebook is hand-edited and left alone, so the file on disk is exactly what gets pushed."""
    root = REPO / "Alignment"
    files = [*sorted(root.glob("lora_*.py")), *sorted((root / "data").glob("*.jsonl"))]
    rels = [f.relative_to(root).as_posix() for f in files]
    meta = {"git_sha": git("rev-parse", "--short", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--", "Alignment")),
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "files": rels}
    dirs = sorted({str(Path("alignment_src", r).parent.as_posix()) for r in rels})
    marker = "\n".join([
        f"{ALIGN_MARK} generated by `scripts/kaggle_push.py alignment` from Alignment/: edit those files, not this",
        "# cell or the %%writefile cells after it.",
        "import json, os",
        *(f"os.makedirs({d!r}, exist_ok=True)" for d in dirs),
        f"SOURCES_META = {meta!r}",
        "with open('alignment_src/SOURCES_META.json', 'w', encoding='utf-8') as fh:",
        "    json.dump(SOURCES_META, fh, indent=1)",
        "print(SOURCES_META)"])
    new = [_cell("alignment-src-meta", "code", marker)]
    for f, r in zip(files, rels):
        text = f.read_text(encoding="utf-8")
        new.append(_cell("alignment-src-" + r.replace("/", "-").replace(".", "-"), "code",
                         ALIGN_WRITE + r + "\n" + text.rstrip("\n")))
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    at = [i for i, c in enumerate(cells) if "".join(c["source"]).startswith(ALIGN_MARK)]
    if len(at) != 1:
        sys.exit(f"{nb_path}: expected exactly one cell starting with {ALIGN_MARK!r}, found {len(at)}")
    end = at[0] + 1
    while end < len(cells) and "".join(cells[end]["source"]).startswith(ALIGN_WRITE):
        end += 1
    nb["cells"] = cells[: at[0]] + new + cells[end:]
    nb.setdefault("metadata", {}).setdefault(
        "kernelspec", {"display_name": "Python 3", "language": "python", "name": "python3"})
    nb["metadata"].setdefault("language_info", {"name": "python"})
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"synced {len(files)} source cells into {nb_path.relative_to(REPO)}: {meta}")
    return meta


def push_alignment() -> None:
    """Alignment/alignment.ipynb: LoRA SFT (E107) + LoRA RL (E108) in one notebook. The notebook is edited directly;
    its generated source cells are refreshed from Alignment/ first, then that exact file is pushed. T4 with internet
    on (the model comes from HF): training is not the offline submission, and the RTX Pro 6000 stays reserved."""
    s = settings()
    b = s["alignment"]
    nb_path = REPO / b["notebook"]
    sync_alignment(nb_path)
    kdir = KAGGLE / ".push" / "alignment"
    shutil.rmtree(kdir, ignore_errors=True)
    kdir.mkdir(parents=True)
    shutil.copy(nb_path, kdir / nb_path.name)
    kmeta = {
        "id": f"{s['username']}/{b['kernel_slug']}", "title": b["kernel_title"], "code_file": nb_path.name,
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
        "enable_tpu": False, "enable_internet": True, "competition_sources": [],
        "dataset_sources": b.get("datasets", []), "kernel_sources": [], "model_sources": b.get("model_sources", []),
        "machine_shape": b.get("machine_shape", "NvidiaTeslaT4"),
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(kmeta, indent=2), encoding="utf-8")
    check_inputs(kmeta["dataset_sources"])
    kaggle("kernels", "push", "-p", str(kdir))


def push_alignment_tpu() -> None:
    """kaggle/experimental/alignment_tpu.ipynb: LoRA SFT on a Kaggle TPU (Tunix + MaxText, lora_sft.py backend
    "tunix"). Unlike `alignment`, nothing is embedded: the notebook imports Alignment/ from the git-linked repo
    dataset (banwait13/sarblohagent), so update that dataset after committing. Internet on (pip + HF weights)."""
    s = settings()
    b = s["alignment_tpu"]
    kdir = KAGGLE / ".push" / "alignment_tpu"
    shutil.rmtree(kdir, ignore_errors=True)
    kdir.mkdir(parents=True)
    nb = REPO / b["notebook"]
    shutil.copy(nb, kdir / nb.name)
    meta = {
        "id": f"{s['username']}/{b['kernel_slug']}", "title": b["kernel_title"], "code_file": nb.name,
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False,
        "enable_tpu": True, "enable_internet": True, "competition_sources": [],
        "dataset_sources": b["datasets"], "kernel_sources": b.get("kernel_sources", []),
        "model_sources": b.get("model_sources", []),
        "machine_shape": b.get("machine_shape", "TpuV5E8"),   # same names as `kaggle kernels push --accelerator`
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    check_inputs(meta["dataset_sources"])
    kaggle("kernels", "push", "-p", str(kdir))


def push_wheels(message: str) -> None:
    wdir = KAGGLE / "datasets" / "sarbloh-wheels-gemma4"
    ref = json.loads((wdir / "dataset-metadata.json").read_text(encoding="utf-8"))["id"]
    exists = subprocess.run(["kaggle", "datasets", "status", ref], capture_output=True, text=True).returncode == 0
    if exists:
        kaggle("datasets", "version", "-p", str(wdir), "-m", message)
    else:
        kaggle("datasets", "create", "-p", str(wdir))


WHEELS_BUILD_SLUG = "sarbloh-vllm-wheels-build"


def push_wheels_build() -> None:
    """kaggle/wheels.ipynb on CPU with internet on: it downloads the vLLM stack into /kaggle/working/wheelhouse. The
    output becomes the banwait13/sarbloh-vllm-wheelhouse dataset (made by the owner from the notebook output)."""
    s = settings()
    kdir = KAGGLE / ".push" / "wheels_build"
    shutil.rmtree(kdir, ignore_errors=True)
    kdir.mkdir(parents=True)
    nb = KAGGLE / "wheels.ipynb"
    shutil.copy(nb, kdir / nb.name)
    meta = {
        "id": f"{s['username']}/{WHEELS_BUILD_SLUG}", "title": "SARBLOH vllm wheels build", "code_file": nb.name,
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False,
        "enable_tpu": False, "enable_internet": True, "competition_sources": [], "dataset_sources": [],
        "kernel_sources": [], "model_sources": [],
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    kaggle("kernels", "push", "-p", str(kdir))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", type=str.lower, choices=["build", "dataset", "kernel", "status", "output", "prime", "prime-status", "prime-output", "wheels",
                                    "experimentation", "experimentation-status", "experimentation-output",
                                    "deepseek", "deepseek-status", "deepseek-output",
                                    "wheels-build", "wheels-build-status", "wheels-build-output",
                                    "vllm-bench", "vllm-bench-status", "vllm-bench-output",
                                    "dsv4-probe", "dsv4-probe-status", "dsv4-probe-output",
                                    "glm", "glm-status", "glm-output", "glm-probe", "glm-probe-status", "glm-probe-output",
                                    "alignment", "alignment-sync", "alignment-status", "alignment-output",
                                    "alignment-tpu", "alignment-tpu-status", "alignment-tpu-output"])
    ap.add_argument("-m", "--message", default="update")
    args = ap.parse_args()
    load_dotenv()
    if args.cmd == "build":
        build()
    elif args.cmd == "dataset":
        push_dataset(args.message)
    elif args.cmd == "kernel":
        push_kernel()
    elif args.cmd == "prime":
        push_prime()
    elif args.cmd == "experimentation":
        push_prime("experimentation")
    elif args.cmd == "deepseek":
        push_prime("deepseek")
    elif args.cmd == "wheels":
        push_wheels(args.message)
    elif args.cmd == "vllm-bench":
        push_vllm_bench()
    elif args.cmd == "dsv4-probe":
        push_vllm_bench("dsv4_probe", "Sarbloh-Experimentation")
    elif args.cmd == "glm":
        push_glm()
    elif args.cmd == "glm-probe":
        push_prime("glm_probe")
    elif args.cmd == "wheels-build":
        push_wheels_build()
    elif args.cmd == "alignment":
        push_alignment()
    elif args.cmd == "alignment-sync":
        sync_alignment(REPO / settings()["alignment"]["notebook"])
    elif args.cmd == "alignment-tpu":
        push_alignment_tpu()
    else:
        s = settings()
        slug = (s["prime"]["kernel_slug"] if args.cmd.startswith("prime-")
                else s["alignment_tpu"]["kernel_slug"] if args.cmd.startswith("alignment-tpu-")
                else s["experimentation"]["kernel_slug"] if args.cmd.startswith("experimentation-")
                else s["deepseek"]["kernel_slug"] if args.cmd.startswith("deepseek-")
                else WHEELS_BUILD_SLUG if args.cmd.startswith("wheels-build-")
                else s["vllm_bench"]["kernel_slug"] if args.cmd.startswith("vllm-bench-")
                else s["dsv4_probe"]["kernel_slug"] if args.cmd.startswith("dsv4-probe-")
                else s["glm_probe"]["kernel_slug"] if args.cmd.startswith("glm-probe-")
                else s["glm"]["kernel_slug"] if args.cmd.startswith("glm-")
                else s["alignment"]["kernel_slug"] if args.cmd.startswith("alignment-") else s["kernel_slug"])
        kid = f"{s['username']}/{slug}"
        if args.cmd.endswith("status"):
            kaggle("kernels", "status", kid)
        else:
            out = REPO / "runs" / f"kaggle_{slug}_{time.strftime('%Y%m%d_%H%M%S')}"
            kaggle("kernels", "output", kid, "-p", str(out))
            print(f"outputs in {out}")


if __name__ == "__main__":
    main()
