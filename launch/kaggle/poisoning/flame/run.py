"""Kaggle kernel: FLAME's own claim, BadNet on Fashion-MNIST, zhmzm/FLAME harness.

Runs the unofficial harness unchanged at a pinned commit:
  python main_fed.py --dataset fashion_mnist --model cnn --defence <D> --attack badnet
                     --malicious 0.2 --poison_frac 0.5 --lr 0.1 --epochs 200
once per defence in DEFENCES. "flame" is FLAME's claim; "avg" (FedAvg, no defence)
is the baseline it is measured against.

Upstream defines --seed but never uses it, so main_fed.py is started through
runpy after seeding random/numpy/torch with SEED. The upstream file is not edited.
GPU kernels may still be non-deterministic.

This does not run Yang et al.'s attack; that glue is not written yet.

Local smoke test (CPU, 1 round):
  SOK_WORK=/tmp/sok SOK_OUT=/tmp/sok-out SOK_EPOCHS=1 python run.py
"""
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time

# ---- Run settings (edit here; they are written to run.json) -------------------
FLAME_REPO = "https://github.com/zhmzm/FLAME"
FLAME_COMMIT = "fd4cef2ac2d5718bad84cf6449e0e3d1e5faf0b5"
DEFENCES = ["flame", "avg"]
SEED = 1  # the upstream --seed default
EPOCHS = int(os.environ.get("SOK_EPOCHS", "200"))
ARGS = ["--dataset", "fashion_mnist", "--model", "cnn", "--attack", "badnet",
        "--malicious", "0.2", "--poison_frac", "0.5", "--lr", "0.1",
        "--epochs", str(EPOCHS), "--seed", str(SEED)]
PIP_PACKAGES = ["hdbscan"]
SOK_COMMIT = "__SOK_COMMIT__"  # filled in by launch/kaggle/push.sh

ON_KAGGLE = os.path.isdir("/kaggle/working")
WORK = os.environ.get("SOK_WORK", "/tmp/sok" if ON_KAGGLE else os.path.abspath("_work"))
OUT = os.environ.get("SOK_OUT", "/kaggle/working/results/flame_badnet" if ON_KAGGLE
                     else os.path.abspath("_out"))
REPO = os.path.join(WORK, "flame")  # datasets download to ../data, i.e. WORK/data

# Seeds the process, then runs main_fed.py as __main__ with the given argv.
LAUNCHER = """
import random, runpy, sys
import numpy as np, torch
seed = int(sys.argv[1])
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
sys.argv = ["main_fed.py"] + sys.argv[2:]
runpy.run_path("main_fed.py", run_name="__main__")
"""


def sh(cmd, cwd=None, log=None):
    print("+", " ".join(cmd), flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")  # so per-round prints reach the Kaggle log live
    with subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as p:
        fh = open(log, "w") if log else None
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


def summarise(stdout_path):
    """Per-round accuracies from main_fed.py's own prints."""
    text = open(stdout_path).read()
    main = [float(x) for x in re.findall(r"^Main accuracy: ([\d.]+)", text, re.M)]
    back = [float(x) for x in re.findall(r"^Backdoor accuracy: ([\d.]+)", text, re.M)]
    return {"rounds": len(main),
            "final_main_accuracy": main[-1] if main else None,
            "final_backdoor_accuracy": back[-1] if back else None,
            "max_backdoor_accuracy": max(back) if back else None,
            "main_accuracy": main, "backdoor_accuracy": back}


def main():
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    if not os.path.isdir(os.path.join(REPO, ".git")):
        sh(["git", "clone", "-q", FLAME_REPO, REPO])
    sh(["git", "checkout", "-q", FLAME_COMMIT], cwd=REPO)
    assert capture(["git", "rev-parse", "HEAD"], cwd=REPO) == FLAME_COMMIT
    sh([sys.executable, "-m", "pip", "install", "-q", *PIP_PACKAGES])

    import torch
    if ON_KAGGLE:
        if not torch.cuda.is_available():
            raise SystemExit("no CUDA device on Kaggle: enable a GPU (T4)")
        gpu = torch.cuda.get_device_name(0)
        if "T4" not in gpu:
            raise SystemExit(f"GPU is {gpu}, not a T4: re-run with accelerator NvidiaTeslaT4")
    # main_fed.py uses cuda:<--gpu>; Kaggle's T4 option is 2x T4, so pin GPU 0.
    gpu_args = ["--gpu", "0"] if torch.cuda.is_available() else ["--gpu", "-1"]

    summary = {}
    for defence in DEFENCES:
        save = os.path.join(REPO, "save")
        shutil.rmtree(save, ignore_errors=True)
        os.makedirs(save)  # models/test.py writes ./save/test_trigger.png
        argv = ARGS + ["--defence", defence] + gpu_args
        stdout_path = os.path.join(OUT, f"{defence}_stdout.txt")
        # One failed defence must not stop the others; record the failure instead.
        try:
            sh([sys.executable, "-c", LAUNCHER, str(SEED), *argv], cwd=REPO, log=stdout_path)
            error = None
        except SystemExit as e:
            error = str(e)
        shutil.copytree(save, os.path.join(OUT, f"{defence}_save"), dirs_exist_ok=True)
        summary[defence] = {"command": ["python", "main_fed.py", *argv], "completed": error is None,
                            "error": error, **summarise(stdout_path)}
        s = summary[defence]
        print(f"[{defence}] completed={s['completed']} rounds={s['rounds']} "
              f"main={s['final_main_accuracy']} backdoor={s['final_backdoor_accuracy']}", flush=True)

    record = {
        "pair": "poisoning/flame",
        "run": "FLAME's own claim: BadNet, Fashion-MNIST CNN, FLAME vs FedAvg",
        "sok_commit": SOK_COMMIT,
        "upstream": {"repo": FLAME_REPO, "commit": FLAME_COMMIT},
        "seed": SEED,
        "seeding": "random, numpy, torch, torch.cuda seeded before runpy of main_fed.py",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "wall_seconds": round(time.time() - t0, 1),
        "results": {d: {k: v for k, v in s.items() if k not in ("main_accuracy", "backdoor_accuracy")}
                    for d, s in summary.items()},
    }
    with open(os.path.join(OUT, "run.json"), "w") as f:
        json.dump(record, f, indent=2)
    with open(os.path.join(OUT, "curves.json"), "w") as f:
        json.dump({d: {"main_accuracy": s["main_accuracy"], "backdoor_accuracy": s["backdoor_accuracy"]}
                   for d, s in summary.items()}, f)
    with open(os.path.join(OUT, "pip_freeze.txt"), "w") as f:
        f.write(capture([sys.executable, "-m", "pip", "freeze"]))
    with open(os.path.join(OUT, "nvidia_smi.txt"), "w") as f:
        f.write(capture(["nvidia-smi"]))
    print(f"done in {record['wall_seconds']} s; results in {OUT}")
    failed = [d for d, s in summary.items() if not s["completed"]]
    if failed:
        raise SystemExit(f"runs that did not complete: {failed} (see run.json)")


if __name__ == "__main__":
    main()
