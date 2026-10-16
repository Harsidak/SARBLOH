# Nemotron 3.5 Super VL NVFP4: Hugging Face -> Colab disk -> zipped parts on Google Drive (MyDrive/nemotron_zip).
# Two Colab cells. Paste each "# %%" block into its own cell. Delete this file once the copy exists.
# Paste HF_TOKEN from .env below. Do not share or save the notebook with the key in it.

# %% Cell 1: download from Hugging Face and check every file's size
HF_TOKEN = ""  # .env HF_TOKEN

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "-U", "huggingface_hub", "hf_xet"], check=True)
import os, shutil
from huggingface_hub import snapshot_download, HfApi

REPO  = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
LOCAL = "/content/nemotron"

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
snapshot_download(REPO, local_dir=LOCAL, token=HF_TOKEN, max_workers=16)  # rerun the cell to resume
shutil.rmtree(f"{LOCAL}/.cache", ignore_errors=True)

info = HfApi().model_info(REPO, token=HF_TOKEN, files_metadata=True)
bad = [s.rfilename for s in info.siblings
       if not os.path.exists(f"{LOCAL}/{s.rfilename}") or os.path.getsize(f"{LOCAL}/{s.rfilename}") != s.size]
print(f"{len(info.siblings)} files, {sum(s.size for s in info.siblings) / 1e9:.2f} GB, mismatches: {bad or 'none'}")
assert not bad, "size mismatch: rerun this cell, the download resumes"
print("subfolders:", [d for d in os.listdir(LOCAL) if os.path.isdir(f"{LOCAL}/{d}")] or "none")
print(f"free disk: {shutil.disk_usage('/content').free / 1e9:.0f} GB")

# %% Cell 2: zip the model and save it to Google Drive (MyDrive/nemotron_zip). Rerun to resume.
# Split into parts of about 10 GB (nemotron.part01.zip, ...), so the Colab disk only ever holds one part next to
# the model. Every part is a complete zip on its own: unzip all of them into one folder to get the model back.
# Stored, not compressed: the weights are already 4-bit and barely shrink, so compressing only costs hours.
import os, shutil, zipfile, json
from google.colab import drive
drive.mount("/content/drive")

LOCAL = "/content/nemotron"
DRIVE = "/content/drive/MyDrive/nemotron_zip"
PART_GB = 10
os.makedirs(DRIVE, exist_ok=True)

files = sorted(os.path.relpath(os.path.join(r, n), LOCAL) for r, _, ns in os.walk(LOCAL) for n in ns)
parts, cur, size = [], [], 0
for rel in files:  # the same file order every run, so the parts come out the same on a rerun
    s = os.path.getsize(f"{LOCAL}/{rel}")
    if cur and size + s > PART_GB * 1e9:
        parts.append(cur)
        cur, size = [], 0
    cur.append(rel)
    size += s
parts.append(cur)

manifest = {}
for i, part in enumerate(parts, 1):
    name = f"nemotron.part{i:02d}.zip"
    dst, tmp = f"{DRIVE}/{name}", f"/content/{name}"
    done = f"{DRIVE}/{name}.done"  # written only after the part is fully on Drive
    if os.path.exists(done) and os.path.exists(dst):
        manifest[name] = json.load(open(done))
        print(f"[{i}/{len(parts)}] {name} already on Drive")
        continue
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for rel in part:
            z.write(f"{LOCAL}/{rel}", rel)
    with zipfile.ZipFile(tmp) as z:
        assert z.testzip() is None, f"{name}: CRC check failed"
    shutil.copyfile(tmp, dst)
    info = {"bytes": os.path.getsize(tmp), "files": part}
    assert os.path.getsize(dst) == info["bytes"], f"{name}: Drive copy is short, rerun this cell"
    json.dump(info, open(done, "w"))
    os.remove(tmp)
    manifest[name] = info
    print(f"[{i}/{len(parts)}] {name} {info['bytes'] / 1e9:.2f} GB, {len(part)} files")

json.dump(manifest, open(f"{DRIVE}/manifest.json", "w"), indent=1)
drive.flush_and_unmount()  # waits until Drive has the data
drive.mount("/content/drive")
bad = [n for n, inf in manifest.items()
       if not os.path.exists(f"{DRIVE}/{n}") or os.path.getsize(f"{DRIVE}/{n}") != inf["bytes"]]
n_files = sum(len(inf["files"]) for inf in manifest.values())
print(f"{len(manifest)} zips, {n_files} of {len(files)} files, mismatches: {bad or 'none'}")
