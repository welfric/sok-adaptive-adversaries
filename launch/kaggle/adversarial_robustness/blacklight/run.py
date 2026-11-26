"""Kaggle kernel: OARS (Feng et al., CCS 2023) against Blacklight on CIFAR-10.

Runs the authors' code unchanged at a pinned commit:
  python main.py --config <CONFIG> --start_idx <START_IDX> --num_images <NUM_IMAGES>
  python analysis.py --log_path <config dir>/logs/<config name>

Everything the run needs (repo, CIFAR-10, PNG export) lives under WORK, outside
/kaggle/working, so the kernel output holds only results/.

Local use (data export and import check only; main.py needs CUDA):
  SOK_WORK=/tmp/sok SOK_OUT=/tmp/sok-out python run.py --prepare-only
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time

# ---- Run settings (edit here; they are written to run.json) -------------------
OARS_REPO = "https://github.com/nmangaokar/ccs_23_oars_stateful_attacks"
OARS_COMMIT = "0067b5b779b066ebeaac85e0d7d9851c143e97ac"
CONFIG = "configs/cifar/blacklight/square/untargeted/adaptive/config.json"
START_IDX = 0
NUM_IMAGES = 100
# dill: models/pretrained/resnet20-12fca82f-single.pth is a whole-model pickle that needs it
PIP_PACKAGES = ["adversarial-robustness-toolbox", "torch-dct", "icecream", "parse", "dill"]
SOK_COMMIT = "__SOK_COMMIT__"  # filled in by launch/kaggle/push.sh

ON_KAGGLE = os.path.isdir("/kaggle/working")
WORK = os.environ.get("SOK_WORK", "/tmp/sok" if ON_KAGGLE else os.path.abspath("_work"))
OUT = os.environ.get("SOK_OUT", "/kaggle/working/results/blacklight_oars" if ON_KAGGLE
                     else os.path.abspath("_out"))
REPO = os.path.join(WORK, "oars")


def sh(cmd, cwd=None, env=None, log=None):
    """Run a command, echo its output, optionally tee it to a file; fail on error."""
    print("+", " ".join(cmd), flush=True)
    env = dict(env or os.environ, PYTHONUNBUFFERED="1")  # live output in the Kaggle log
    with subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as p:
        fh = open(log, "a") if log else None
        for line in p.stdout:
            sys.stdout.write(line)
            if fh:
                fh.write(line)
        if fh:
            fh.close()
    if p.returncode != 0:
        raise SystemExit(f"command failed ({p.returncode}): {' '.join(cmd)}")


def capture(cmd, cwd=None):
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return ""


def clone():
    if not os.path.isdir(os.path.join(REPO, ".git")):
        sh(["git", "clone", "-q", OARS_REPO, REPO])
    sh(["git", "checkout", "-q", OARS_COMMIT], cwd=REPO)
    head = capture(["git", "rev-parse", "HEAD"], cwd=REPO)
    assert head == OARS_COMMIT, f"OARS checkout is at {head}, expected {OARS_COMMIT}"


def export_cifar10():
    """CIFAR-10 test set in the layout the OARS README asks for (section 2.2.4):
    data/cifar10/imgs/<i>.png, cifar10.json {"imgs/<i>.png": label} in index order,
    cifar10_targeted.json {"<label>": ["imgs/<first test index with that label>"]}."""
    import torchvision
    from PIL import Image

    root = os.path.join(REPO, "data", "cifar10")
    labels_path = os.path.join(root, "cifar10.json")
    if os.path.exists(labels_path):
        print("CIFAR-10 export already present")
        return
    os.makedirs(os.path.join(root, "imgs"), exist_ok=True)
    ds = torchvision.datasets.CIFAR10(os.path.join(WORK, "cifar_raw"), train=False, download=True)
    labels, targeted = {}, {}
    for i, (arr, y) in enumerate(zip(ds.data, ds.targets)):
        rel = f"imgs/{i}.png"
        Image.fromarray(arr).save(os.path.join(root, rel))
        labels[rel] = int(y)
        targeted.setdefault(str(int(y)), [rel])
    with open(os.path.join(root, "cifar10_targeted.json"), "w") as f:
        json.dump(dict(sorted(targeted.items(), key=lambda kv: int(kv[0]))), f, indent=1)
    with open(labels_path, "w") as f:  # written last: its presence marks a finished export
        json.dump(labels, f, indent=1)
    print(f"exported {len(labels)} CIFAR-10 test images to {root}")


def log_dir():
    # main.py: os.path.join(dirname(config), 'logs', basename(config) without extension)
    cfg = os.path.join(REPO, CONFIG)
    return os.path.join(os.path.dirname(cfg), "logs", os.path.splitext(os.path.basename(cfg))[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare-only", action="store_true",
                    help="clone, install, export data and stop before main.py")
    args = ap.parse_args()

    os.makedirs(WORK, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    clone()
    sh([sys.executable, "-m", "pip", "install", "-q", *PIP_PACKAGES])
    # models/iot_sqa.py imports tensorflow.keras at module level, so main.py needs it
    # even for CIFAR. Kaggle images ship TensorFlow; install it only if absent.
    if subprocess.run([sys.executable, "-c", "import tensorflow"], capture_output=True).returncode:
        sh([sys.executable, "-m", "pip", "install", "-q", "tensorflow"])
    export_cifar10()

    env = dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
    # Fail early on a missing dependency instead of after model load.
    sh([sys.executable, "-c", "import attacks.attacks, models.statefuldefense, utils.datasets; print('imports ok')"],
       cwd=REPO, env=env)
    if args.prepare_only:
        return

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device: OARS main.py moves the model to 'cuda'. Enable a GPU (T4).")
    gpu = torch.cuda.get_device_name(0)
    if ON_KAGGLE and "T4" not in gpu:
        raise SystemExit(f"GPU is {gpu}, not a T4: re-run with accelerator NvidiaTeslaT4")
    env["CUDA_VISIBLE_DEVICES"] = "0"  # Kaggle's T4 option is 2x T4; use one

    # Fresh logs: main.py appends to log_<start>_<end>.txt.
    shutil.rmtree(log_dir(), ignore_errors=True)
    sh([sys.executable, "main.py", "--config", CONFIG,
        "--start_idx", str(START_IDX), "--num_images", str(NUM_IMAGES)],
       cwd=REPO, env=env, log=os.path.join(OUT, "main_stdout.txt"))
    sh([sys.executable, "analysis.py", "--log_path", log_dir()],
       cwd=REPO, env=env, log=os.path.join(OUT, "analysis.txt"))

    shutil.copytree(log_dir(), os.path.join(OUT, "logs"), dirs_exist_ok=True)
    shutil.copy(os.path.join(REPO, CONFIG), os.path.join(OUT, "config.json"))

    torch_info = {"torch": torch.__version__, "cuda": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name(0)}
    record = {
        "pair": "adversarial_robustness/blacklight",
        "attack": "OARS (Feng et al., CCS 2023), Square, untargeted, adaptive config",
        "sok_commit": SOK_COMMIT,
        "upstream": {"repo": OARS_REPO, "commit": OARS_COMMIT},
        "command": ["python", "main.py", "--config", CONFIG, "--start_idx", str(START_IDX),
                    "--num_images", str(NUM_IMAGES)],
        "config_sha256": hashlib.sha256(open(os.path.join(REPO, CONFIG), "rb").read()).hexdigest(),
        "seeds": "upstream main.py seeds random/numpy/torch with 0, then re-seeds Python "
                 "'random' from the clock when logging is on (main.py, main()); not overridden",
        "env": {"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
        "python": platform.python_version(),
        **torch_info,
        "wall_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(OUT, "run.json"), "w") as f:
        json.dump(record, f, indent=2)
    with open(os.path.join(OUT, "pip_freeze.txt"), "w") as f:
        f.write(capture([sys.executable, "-m", "pip", "freeze"]))
    with open(os.path.join(OUT, "nvidia_smi.txt"), "w") as f:
        f.write(capture(["nvidia-smi"]))
    print(f"done in {record['wall_seconds']} s; results in {OUT}")


if __name__ == "__main__":
    main()
