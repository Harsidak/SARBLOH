# DeepSeek-V4-Flash-0731 NVFP4 (nvidia/DeepSeek-V4-Flash-0731-NVFP4, ~172 GB): Hugging Face -> Colab disk -> a private
# Kaggle MODEL (not a dataset; Kaggle already holds 167 GB DeepSeek models as one instance). No Google Drive, no zip.
# Two Colab cells. Paste each "# %%" block into its own cell. Delete this file once the model is READY on Kaggle.
# Paste the keys from .env below. Do not share or save the notebook with the keys in it.
# The previous Nemotron -> Google Drive version of this file is in git (a8e0479).

# %% Cell 1: download from Hugging Face, check every file's size, upload to Kaggle. Rerun the cell to resume.
HF_TOKEN         = ""  # .env HF_TOKEN (the repo is not gated; the token only avoids rate limits)
KAGGLE_API_TOKEN = ""  # .env KAGGLE_API_TOKEN, or fill the two below instead
KAGGLE_USERNAME  = "banwait13"
KAGGLE_KEY       = ""  # .env KAGGLE_KEY

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "-U", "huggingface_hub", "hf_xet", "kaggle"], check=True)
import os, shutil, json, tarfile
from huggingface_hub import snapshot_download, HfApi

REPO     = "nvidia/DeepSeek-V4-Flash-0731-NVFP4"
LOCAL    = "/content/dsv4"
OWNER    = KAGGLE_USERNAME
MODEL    = "deepseek-v4-flash-0731-nvfp4"
FRAME    = "transformers"
INSTANCE = "nvfp4"
HANDLE   = f"{OWNER}/{MODEL}/{FRAME}/{INSTANCE}"

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
if KAGGLE_API_TOKEN:
    os.environ["KAGGLE_API_TOKEN"] = KAGGLE_API_TOKEN
os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = KAGGLE_USERNAME, KAGGLE_KEY

def kaggle(*args, check=True):
    print("$ kaggle", " ".join(args))
    return subprocess.run(["kaggle", *args], check=check, text=True, capture_output=not check)

info = HfApi().model_info(REPO, token=HF_TOKEN or None, files_metadata=True)
total = sum(s.size for s in info.siblings)
have = sum(os.path.getsize(f"{LOCAL}/{s.rfilename}") for s in info.siblings if os.path.exists(f"{LOCAL}/{s.rfilename}"))
free = shutil.disk_usage("/content").free
print(f"{REPO}: {len(info.siblings)} files, {total / 1e9:.1f} GB, {have / 1e9:.1f} GB already here, {free / 1e9:.0f} GB free")
assert free > (total - have) + 2e9, f"need {(total - have) / 1e9:.0f} GB more disk, have {free / 1e9:.0f} GB: pick a runtime with a bigger disk"
os.makedirs("/tmp/instance_check", exist_ok=True)
os.makedirs("/tmp/model_check", exist_ok=True)

if kaggle("models", "instances", "get", HANDLE, "-p", "/tmp/instance_check", check=False).returncode == 0:
    print(f"{HANDLE} already exists on Kaggle: run cell 2")
else:
    snapshot_download(REPO, local_dir=LOCAL, token=HF_TOKEN or None, max_workers=16)  # rerun the cell to resume
    shutil.rmtree(f"{LOCAL}/.cache", ignore_errors=True)
    bad = [s.rfilename for s in info.siblings
           if not os.path.exists(f"{LOCAL}/{s.rfilename}") or os.path.getsize(f"{LOCAL}/{s.rfilename}") != s.size]
    print(f"size mismatches: {bad or 'none'}")
    assert not bad, "size mismatch: rerun this cell, the download resumes"

    # encoding/ and inference/ are DeepSeek's small reference code. Kaggle's handling of folders in an upload is not
    # known, so they go up as one plain tar: on Kaggle, `tar -xf dsv4_extras.tar -C <dir>` brings them back.
    subdirs = sorted(d for d in os.listdir(LOCAL) if os.path.isdir(f"{LOCAL}/{d}"))
    if subdirs:
        with tarfile.open(f"{LOCAL}/dsv4_extras.tar", "w") as t:
            for d in subdirs:
                t.add(f"{LOCAL}/{d}", arcname=d)
        for d in subdirs:
            shutil.rmtree(f"{LOCAL}/{d}")
        print(f"packed {subdirs} into dsv4_extras.tar")

    meta_dir = "/content/dsv4_model_meta"
    os.makedirs(meta_dir, exist_ok=True)
    json.dump({"ownerSlug": OWNER, "title": "DeepSeek-V4-Flash-0731 NVFP4", "slug": MODEL, "isPrivate": True,
               "subtitle": "Mirror of nvidia/DeepSeek-V4-Flash-0731-NVFP4 (MIT)",
               "description": f"Unchanged files of https://huggingface.co/{REPO} (revision {info.sha}). "
                              "encoding/ and inference/ are packed in dsv4_extras.tar.",
               "publishTime": "", "provenanceSources": f"https://huggingface.co/{REPO}"},
              open(f"{meta_dir}/model-metadata.json", "w"), indent=1)
    if kaggle("models", "get", f"{OWNER}/{MODEL}", "-p", "/tmp/model_check", check=False).returncode != 0:
        kaggle("models", "create", "-p", meta_dir)

    json.dump({"ownerSlug": OWNER, "modelSlug": MODEL, "instanceSlug": INSTANCE, "framework": FRAME,
               "overview": f"{REPO} at revision {info.sha}", "usage": "", "licenseName": "MIT",
               "fineTunable": False, "trainingData": [], "modelInstanceType": "Unspecified",
               "baseModelInstanceId": 0, "externalBaseModelUrl": ""},
              open(f"{LOCAL}/model-instance-metadata.json", "w"), indent=1)
    kaggle("models", "instances", "create", "-p", LOCAL)  # uploads file by file; a rerun resumes unfinished files
    print(f"uploaded. Kaggle now builds {HANDLE}/1: run cell 2 until it says READY")

# %% Cell 2: check that Kaggle has finished building the model. Rerun until it says READY.
import subprocess
out = subprocess.run(["kaggle", "models", "instances", "list", f"{OWNER}/{MODEL}", "-v"], text=True, capture_output=True)
print(out.stdout or out.stderr)
size = sum(s.size for s in info.siblings)
print(f"expected about {size / 1e9:.1f} GB. When READY, add this to model_sources in scripts/kaggle_settings.json:")
print(f'  "{HANDLE}/1"')
