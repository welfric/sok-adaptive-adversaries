"""Kaggle kernel: FLAME's own claim on CIFAR-10 / ResNet-18, IID, zhmzm/FLAME harness.

Setting taken from FLAME (Nguyen et al., USENIX Security 2022; arXiv:2101.02281 v5)
wherever the paper states it; the zhmzm README's CIFAR-10 example fills the rest.
PAPER_SETTING below records the source of every value and run.json carries it.

Runs the unofficial harness at a pinned commit plus HARNESS_PATCH (two bug fixes, below),
once per (attack, defence):
  python main_fed.py --dataset cifar --model resnet --iid 1 --num_users 100 --frac 0.1
                     --malicious 0.2 --noise 0.001 --attack <A> --defence <D> ...
  attacks:  dba    (FLAME Table 3 has a CIFAR-10 DBA row: the number to compare against)
            badnet (continuity with the Fashion-MNIST run; not a row in FLAME's tables)
  defences: flame, avg (FedAvg, the "no defense" baseline)

Known gaps to the paper (the harness cannot close them without editing upstream):
  - model: harness ResNet18 (~11M params) vs paper "ResNet-18 Light" (~2.7M, App. A)
  - constrain-and-scale (the paper's headline CIFAR-10 attack, Bagdasaryan et al. code)
    is not implemented in the harness; DBA here is zhmzm's re-implementation, not Xie et al.'s
  - training from scratch for ROUNDS rounds; the paper does not state rounds, local
    epochs, batch size or learning rate for CIFAR-10

HARNESS_PATCH fixes two upstream bugs that would otherwise invalidate the comparison:
  1. utils/defense.py flame(): each admitted update was clipped with norm_list[i], the norm
     of the i-th client overall, not of the i-th admitted client. FLAME Alg. 1 line 11
     uses gamma = S_t / e_{b_l}. Fixed to norm_list[benign_client[i]].
  2. models/MaliciousUpdate.py and models/test.py save_img() min-max rescaled the image
     in place while writing a preview PNG. On a GPU, poisoned training images (CPU
     tensors) were rescaled to [0, 1] but triggered test images (CUDA tensors, copied by
     .cpu()) were not, so the backdoor was trained and tested on different inputs.
     Fixed by writing the PNG from a copy.

Upstream defines --seed but never uses it, so main_fed.py is started through runpy
after seeding random/numpy/torch with SEED. GPU kernels may still be non-deterministic.
run.json is rewritten after every run, so a Kaggle timeout keeps the finished ones.

Local smoke test (CPU, 1 round; ResNet-18 on CPU takes a few minutes per round):
  SOK_WORK=/tmp/sok SOK_OUT=/tmp/sok-out-cifar SOK_EPOCHS=1 SOK_RUNS=dba:flame python run.py
"""
import hashlib
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
SEED = 1  # the upstream --seed default
ROUNDS = int(os.environ.get("SOK_EPOCHS", "200"))

# (attack, defence) in run order. DBA first: it is the one with a paper number.
RUNS = [("dba", "flame"), ("dba", "avg"), ("badnet", "flame"), ("badnet", "avg")]
if os.environ.get("SOK_RUNS"):  # e.g. "dba:flame,dba:avg" for smoke tests
    RUNS = [tuple(r.split(":")) for r in os.environ["SOK_RUNS"].split(",")]

COMMON_ARGS = [
    "--dataset", "cifar", "--model", "resnet", "--iid", "1",
    "--num_users", "100", "--frac", "0.1",
    "--malicious", "0.2", "--poison_frac", "0.5",
    "--attack_goal", "7", "--attack_label", "5",
    "--noise", "0.001",
    "--lr", "0.1", "--local_ep", "2", "--local_bs", "64",
    "--epochs", str(ROUNDS), "--seed", str(SEED),
]
# Test-time trigger: DBA is evaluated with the full (combined) trigger, as in Xie et al.
ATTACK_ARGS = {"dba": ["--trigger", "dba"], "badnet": ["--trigger", "square"]}

# Where each value comes from. "paper" = FLAME arXiv:2101.02281 v5.
PAPER_SETTING = {
    "dataset / model": "CIFAR-10, ResNet-18 Light ~2.7M params (paper Table 2, App. A). "
                       "Harness ResNet18 is the full ~11M model: DIFFERS",
    "num_users=100": "paper App. A: 'The experimental setup consists of 100 clients'",
    "malicious=0.2": "paper App. A: 'uses a PMR of 20%'",
    "iid=1": "paper §7.1: CIFAR-10 data 'highly IID (each client is assigned a random set of "
             "images)'. Harness data/iid_cifar.npy: 100 clients x 500 images",
    "attack_goal 7 -> attack_label 5": "paper App. A: one class relabelled to another; "
                                        "classes not stated. Harness default pair",
    "noise=0.001": "paper App. B.3: lambda = 0.001 for IC; harness sigma = noise * median norm",
    "HDBSCAN": "paper §4.3.1 / App. E: min_cluster_size = n/2+1, min_samples = 1 (harness same)",
    "clipping": "paper §4.3.2: S_t = median L2 norm of all n updates (harness same)",
    "frac=0.1": "paper: clients per round not stated for CIFAR-10. Harness default",
    "poison_frac=0.5": "paper: PDR not stated for CIFAR-10. Same flag as the Fashion-MNIST run. "
                       "Harness meaning (one-to-one backdoor): in each batch, about half of the "
                       "class-7 images (floor(n/2)+1) get the trigger and label 5, replacing the "
                       "clean ones, i.e. roughly 5-7% of a malicious client's samples, not 50%",
    "lr=0.1, local_ep=2, local_bs=64": "paper: not stated. zhmzm README CIFAR-10 example",
    "rounds": "paper: not stated. 200, as in the zhmzm README example; trained from scratch",
    "attack schedule": "harness: int(0.2 * 10) = 2 malicious clients in every round from round 1 "
                       "(round 0 has none: attack starts once accuracy > --attack_begin 0). "
                       "BadNet attackers are drawn from clients 0-20, DBA cycles clients 0-19, "
                       "each DBA client holding one quarter of the 5x5 trigger",
    "trigger": "5x5 white square at (0, 0), in the normalised input space; DBA's four pieces "
               "tile the same square, so both attacks are tested with the same global trigger",
}
PAPER_REFERENCE = {  # FLAME Table 3, CIFAR-10, percentages
    "dba": {"no_defense": {"BA": 93.8, "MA": 57.4}, "flame": {"BA": 3.2, "MA": 76.2}},
    "constrain_and_scale": {"no_defense": {"BA": 81.9, "MA": 89.8}, "flame": {"BA": 0.0, "MA": 91.9}},
    "badnet": "not in the paper; zhmzm README (unofficial): no defence ASR 70.2 / Acc 80.38, "
              "FLAME ASR 3.33 / Acc 78.77 (CIFAR-10, ResNet18, IID)",
}
# (file, exact old text, new text). Each old text must occur exactly once at FLAME_COMMIT.
HARNESS_PATCH = [
    # 1. FLAME Alg. 1 line 11: gamma = S_t / e_{b_l}, the admitted update's own norm
    ("utils/defense.py",
     "        gama = clip_value/norm_list[i]\n",
     "        gama = clip_value/norm_list[benign_client[i]]\n"),
    # 2. save_img only writes a preview PNG: work on a copy, never on the image itself
    ("models/MaliciousUpdate.py",
     "    def save_img(self, image):\n        img = image\n",
     "    def save_img(self, image):\n        img = image.detach().cpu().clone()\n"),
    ("models/MaliciousUpdate.py",
     "            img = image.numpy()\n",
     "            img = img.numpy()\n"),
    ("models/test.py",
     "def save_img(image):\n        img = image\n",
     "def save_img(image):\n        img = image.detach().cpu().clone()\n"),
    ("models/test.py",
     "            img = image.cpu().numpy()\n",
     "            img = img.numpy()\n"),
]
PIP_PACKAGES = ["hdbscan"]
SOK_COMMIT = "__SOK_COMMIT__"  # filled in by launch/kaggle/push.sh

ON_KAGGLE = os.path.isdir("/kaggle/working")
WORK = os.environ.get("SOK_WORK", "/tmp/sok" if ON_KAGGLE else os.path.abspath("_work"))
OUT = os.environ.get("SOK_OUT", "/kaggle/working/results/flame_cifar" if ON_KAGGLE
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
            sys.stdout.flush()  # stdout is block-buffered when redirected to a file
            if fh:
                fh.write(line)
                fh.flush()  # a killed kernel keeps the stdout written so far
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
    """Per-round accuracies and FLAME's selection rates from main_fed.py's own prints."""
    text = open(stdout_path).read() if os.path.exists(stdout_path) else ""
    main = [float(x) for x in re.findall(r"^Main accuracy: ([\d.]+)", text, re.M)]
    back = [float(x) for x in re.findall(r"^Backdoor accuracy: ([\d.]+)", text, re.M)]
    mal = re.findall(r"^proportion of malicious are selected: ([\d.eE+-]+)", text, re.M)
    ben = re.findall(r"^proportion of benign are selected: ([\d.eE+-]+)", text, re.M)
    tail = back[-max(len(back) // 10, 1):]
    return {"rounds": len(main),
            "final_main_accuracy": main[-1] if main else None,
            "final_backdoor_accuracy": back[-1] if back else None,
            "max_backdoor_accuracy": max(back) if back else None,
            "mean_backdoor_accuracy_last_10pct": round(sum(tail) / len(tail), 2) if back else None,
            # cumulative over rounds; FLAME only: share of malicious / benign updates admitted
            "flame_malicious_admitted": float(mal[-1]) if mal else None,
            "flame_benign_admitted": float(ben[-1]) if ben else None,
            "main_accuracy": main, "backdoor_accuracy": back}


def main():
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    if not os.path.isdir(os.path.join(REPO, ".git")):
        sh(["git", "clone", "-q", FLAME_REPO, REPO])
    # Pinned commit, no leftovers from an earlier run, then the bug-fix patch.
    sh(["git", "reset", "-q", "--hard", FLAME_COMMIT], cwd=REPO)
    sh(["git", "clean", "-fdq"], cwd=REPO)
    assert capture(["git", "rev-parse", "HEAD"], cwd=REPO) == FLAME_COMMIT
    for rel, old, new in HARNESS_PATCH:
        path = os.path.join(REPO, rel)
        text = open(path).read()
        assert text.count(old) == 1, f"patch target not found exactly once in {rel}: {old!r}"
        with open(path, "w") as f:
            f.write(text.replace(old, new))
    patched = capture(["git", "diff", "--stat"], cwd=REPO)
    print(patched, flush=True)
    assert "3 files changed, 5 insertions(+), 5 deletions(-)" in patched, patched
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

    record = {
        "pair": "poisoning/flame",
        "run": "FLAME's own claim: CIFAR-10 ResNet-18 IID, DBA and BadNet, FLAME vs FedAvg",
        "sok_commit": SOK_COMMIT,
        "upstream": {"repo": FLAME_REPO, "commit": FLAME_COMMIT},
        "upstream_patch": {
            "sha256": hashlib.sha256(json.dumps(HARNESS_PATCH).encode()).hexdigest(),
            "diff_stat": patched,
            "diff": capture(["git", "diff"], cwd=REPO),
            "fixes": ["flame(): clip each admitted update by its own norm (Alg. 1 line 11)",
                      "save_img(): stop rescaling triggered images in place"],
        },
        "seed": SEED,
        "seeding": "random, numpy, torch, torch.cuda seeded before runpy of main_fed.py",
        "paper_setting": PAPER_SETTING,
        "paper_reference": PAPER_REFERENCE,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "results": {},
    }
    curves = {}

    def write_outputs():
        record["wall_seconds"] = round(time.time() - t0, 1)
        with open(os.path.join(OUT, "run.json"), "w") as f:
            json.dump(record, f, indent=2)
        with open(os.path.join(OUT, "curves.json"), "w") as f:
            json.dump(curves, f)

    for attack, defence in RUNS:
        name = f"{attack}_{defence}"
        save = os.path.join(REPO, "save")
        shutil.rmtree(save, ignore_errors=True)
        os.makedirs(save)  # models/test.py writes ./save/test_trigger.png
        argv = COMMON_ARGS + ATTACK_ARGS[attack] + ["--attack", attack, "--defence", defence] + gpu_args
        stdout_path = os.path.join(OUT, f"{name}_stdout.txt")
        t_run = time.time()
        # One failed run must not stop the others; record the failure instead.
        try:
            sh([sys.executable, "-c", LAUNCHER, str(SEED), *argv], cwd=REPO, log=stdout_path)
            error = None
        except SystemExit as e:
            error = str(e)
        shutil.copytree(save, os.path.join(OUT, f"{name}_save"), dirs_exist_ok=True)
        s = summarise(stdout_path)
        curves[name] = {"main_accuracy": s.pop("main_accuracy"),
                        "backdoor_accuracy": s.pop("backdoor_accuracy")}
        record["results"][name] = {"attack": attack, "defence": defence,
                                   "command": ["python", "main_fed.py", *argv],
                                   "completed": error is None, "error": error,
                                   "run_seconds": round(time.time() - t_run, 1), **s}
        write_outputs()
        print(f"[{name}] completed={error is None} rounds={s['rounds']} "
              f"main={s['final_main_accuracy']} backdoor={s['final_backdoor_accuracy']} "
              f"({record['results'][name]['run_seconds']} s)", flush=True)

    with open(os.path.join(OUT, "pip_freeze.txt"), "w") as f:
        f.write(capture([sys.executable, "-m", "pip", "freeze"]))
    with open(os.path.join(OUT, "nvidia_smi.txt"), "w") as f:
        f.write(capture(["nvidia-smi"]))
    write_outputs()
    print(f"done in {record['wall_seconds']} s; results in {OUT}")
    failed = [n for n, r in record["results"].items() if not r["completed"]]
    if failed:
        raise SystemExit(f"runs that did not complete: {failed} (see run.json)")


if __name__ == "__main__":
    main()
