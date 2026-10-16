# Nemotron 3.5 Super VL NVFP4: Hugging Face -> Colab disk -> one zip on Google Drive (MyDrive/nemotron_zip/nemotron.zip).
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

# %% Cell 2: zip the whole model into ONE file and save it to Google Drive (MyDrive/nemotron_zip/nemotron.zip).
# Stored, not compressed: the weights are already 4-bit and barely shrink, so compressing only costs hours.
# The zip is built on the Colab disk first (it needs about the model's size free), checked, then copied to Drive.
# One file cannot resume: if the copy breaks, rerun the cell; a finished zip on /content is reused, not rebuilt.
import os, shutil, zipfile, json
from google.colab import drive
drive.mount("/content/drive")

LOCAL = "/content/nemotron"
DRIVE = "/content/drive/MyDrive/nemotron_zip"
NAME  = "nemotron.zip"
DELETE_MODEL_AFTER_ZIP = False  # True frees the model's disk space before the Drive copy (cell 1 can fetch it again)
os.makedirs(DRIVE, exist_ok=True)

files = sorted(os.path.relpath(os.path.join(r, n), LOCAL) for r, _, ns in os.walk(LOCAL) for n in ns)
total = sum(os.path.getsize(f"{LOCAL}/{rel}") for rel in files)
tmp, dst, done = f"/content/{NAME}", f"{DRIVE}/{NAME}", f"{DRIVE}/{NAME}.done"  # .done: written only when Drive has it all

if os.path.exists(done) and os.path.exists(dst) and os.path.getsize(dst) == json.load(open(done))["bytes"]:
    print(f"{NAME} already on Drive")
else:
    tmp_ok = f"{tmp}.ok"  # the local zip passed its CRC check
    if not (os.path.exists(tmp) and os.path.exists(tmp_ok)):
        free = shutil.disk_usage("/content").free
        assert free > total * 1.02, f"need {total / 1e9:.0f} GB free for the zip, have {free / 1e9:.0f} GB"
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
            for i, rel in enumerate(files, 1):
                z.write(f"{LOCAL}/{rel}", rel)
                print(f"\r[{i}/{len(files)}] zipped {rel}", end="", flush=True)
        print()
        with zipfile.ZipFile(tmp) as z:
            assert z.testzip() is None, f"{NAME}: CRC check failed"
            assert sorted(z.namelist()) == files, f"{NAME}: file list differs from {LOCAL}"
        open(tmp_ok, "w").close()
    if DELETE_MODEL_AFTER_ZIP:
        shutil.rmtree(LOCAL, ignore_errors=True)
    info = {"bytes": os.path.getsize(tmp), "files": files}
    print(f"copying {info['bytes'] / 1e9:.2f} GB to Drive ...")
    shutil.copyfile(tmp, dst)
    drive.flush_and_unmount()  # waits until Drive has the data
    drive.mount("/content/drive")
    assert os.path.getsize(dst) == info["bytes"], f"{NAME}: Drive copy is short, rerun this cell"
    json.dump(info, open(done, "w"), indent=1)
    os.remove(tmp)
    os.remove(tmp_ok)

size = os.path.getsize(dst)
print(f"{dst}: {size / 1e9:.2f} GB, {len(json.load(open(done))['files'])} files (model {total / 1e9:.2f} GB, {len(files)} files)")
