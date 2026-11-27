"""Kaggle kernel: FLAME vs DBA in DBA's own code and setting (FLAME Table 3, CIFAR-10 DBA row).

Follows defenses/poisoning/ATTACK_PROTOCOL.md. Runs DBA (Xie et al., ICLR 2020;
github.com/AI-secure/DBA at DBA_COMMIT) with its released CIFAR-10 config and its pretrained
round-200 clean model, rounds 201-300, once per (data split, aggregation):
  splits:       iid (FLAME §7.1), dirichlet (alpha 0.5, DBA's config)
  aggregations: mean (DBA's FedAvg, the "no defence" baseline), flame (FLAME_SOURCE below)

FLAME is not part of DBA. FLAME_SOURCE implements FLAME Alg. 1 (arXiv:2101.02281 v5) inside
DBA's server loop: cosine distances between the clients' weight/bias updates (§4.3.1), HDBSCAN on
the precomputed distance matrix with min_cluster_size n/2+1, min_samples 1 (App. E), clipping
bound S_t = median update norm over all n clients (§4.3.2), each admitted update clipped by
min(1, S_t/e_i) (Alg. 1 line 11), mean of the admitted clipped updates, Gaussian noise
sigma = lambda * S_t with lambda = 0.001 (App. B.3). Protocol choices the paper leaves unstated:
the server step G += eta * aggregate uses DBA's eta = 0.1 for both aggregations; BatchNorm
running statistics are clipped with the same factor and averaged, not noised; if HDBSCAN labels
every client as noise, all are admitted (logged as "fallback").

DBA_PATCH (exact replacements, each must match once) only adapts DBA to current libraries and
adds the hook; it does not change DBA's algorithm:
  - yaml.load(f) -> yaml.safe_load(f) (PyYAML 6 requires a Loader)
  - torch.load(..., weights_only=True) for the pretrained checkpoint
  - seeds read from SOK_SEED instead of the hard-coded 1 (default 1, so unchanged)
  - main.py dispatches aggregation_methods == 'flame' to sok_flame.flame_update
Visdom (a live-plot server DBA imports) is replaced by a no-op stub on PYTHONPATH.

The checkpoint is DBA's saved_models.zip from the Google Drive link in DBA's README; only
cifar_pretrain/model_last.pt.tar.epoch_200 is read from it. sha256 of both is in run.json.

Local smoke test (CPU, no checkpoint: 2 rounds from scratch, attacks in rounds 1 and 2):
  SOK_WORK=/tmp/sok SOK_OUT=/tmp/sok-out-dba SOK_SMOKE=1 SOK_RUNS=iid:flame python run.py
"""
import csv
import glob
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
import zipfile

# ---- Run settings (edit here; they are written to run.json) -------------------
DBA_REPO = "https://github.com/AI-secure/DBA"
DBA_COMMIT = "3cabf7ec3d9967eefb9c3cb983c6eb671ac12b33"
CKPT_DRIVE_ID = "1wcJ_DkviuOLkmr-FgIVSFwnZwyGU8SjH"  # DBA README: "Our pretrained clean models"
CKPT_MEMBER = "cifar_pretrain/model_last.pt.tar.epoch_200"  # DBA config resumed_model_name
SEED = int(os.environ.get("SOK_SEED", "1"))
SMOKE = os.environ.get("SOK_SMOKE") == "1"

# (split, aggregation) in run order. IID first: FLAME §7.1 says its CIFAR-10 data are IID.
RUNS = [("iid", "flame"), ("iid", "mean"), ("dirichlet", "flame"), ("dirichlet", "mean")]
if os.environ.get("SOK_RUNS"):  # e.g. "iid:flame,iid:mean"
    RUNS = [tuple(r.split(":")) for r in os.environ["SOK_RUNS"].split(",")]

# Overrides on top of DBA's utils/cifar_params.yaml. Everything else is DBA's released config.
COMMON_OVERRIDES = {
    "save_model": False,
    "flame_lambda": 0.001,  # FLAME App. B.3: "λ = 0.001 for IC and NLP"
}
SPLIT_OVERRIDES = {"iid": {"sampling_dirichlet": False}, "dirichlet": {"sampling_dirichlet": True}}
SMOKE_OVERRIDES = {  # no checkpoint: start at round 1, attack in rounds 1 and 2
    "resumed_model": False, "epochs": 2,
    "0_poison_epochs": [1], "1_poison_epochs": [1], "2_poison_epochs": [2], "3_poison_epochs": [2],
}
REPORT_ROUNDS = [209, 210, 220, 250, 300]  # protocol section 7
PAPER_REFERENCE = {  # FLAME Table 3, CIFAR-10, BA / MA in %
    "dba": {"no_defense": {"BA": 93.8, "MA": 57.4}, "flame": {"BA": 3.2, "MA": 76.2}},
}

# (file, exact old text, new text). Each old text must occur exactly once at DBA_COMMIT.
DBA_PATCH = [
    ("main.py", "        params_loaded = yaml.load(f)\n",
     "        params_loaded = yaml.safe_load(f)\n"),
    ("main.py", "torch.manual_seed(1)\ntorch.cuda.manual_seed(1)\nrandom.seed(1)\n",
     "SOK_SEED = int(os.environ.get('SOK_SEED', '1'))\n"
     "torch.manual_seed(SOK_SEED)\ntorch.cuda.manual_seed(SOK_SEED)\nrandom.seed(SOK_SEED)\n"),
    ("main.py", "    np.random.seed(1)\n", "    np.random.seed(SOK_SEED)\n"),
    ("main.py",
     "        elif helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:\n",
     "        elif helper.params['aggregation_methods'] == 'flame':\n"
     "            import sok_flame\n"
     "            is_updated = sok_flame.flame_update(helper, helper.target_model, updates,\n"
     "                                                agent_name_keys, adversarial_name_keys, epoch)\n"
     "            num_oracle_calls = 1\n"
     "        elif helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:\n"),
    ("image_helper.py",
     "loaded_params = torch.load(f\"saved_models/{self.params['resumed_model_name']}\")\n",
     "loaded_params = torch.load(f\"saved_models/{self.params['resumed_model_name']}\", weights_only=True)\n"),
    ("image_helper.py",
     "loaded_params = torch.load(f\"saved_models/{self.params['resumed_model_name']}\",map_location='cpu')\n",
     "loaded_params = torch.load(f\"saved_models/{self.params['resumed_model_name']}\",map_location='cpu', weights_only=True)\n"),
]

FLAME_SOURCE = r'''"""FLAME (Nguyen et al., USENIX Security 2022), Alg. 1, as a DBA aggregation method."""
import json
import numpy as np
import torch
import hdbscan


def flame_update(helper, target_model, updates, names, adversaries, epoch):
    eta = float(helper.params["eta"])
    lam = float(helper.params["flame_lambda"])
    names = list(names)
    n = len(names)
    state = target_model.state_dict()
    pkeys = [k for k, _ in target_model.named_parameters()]  # weights and biases (§2.1)
    bkeys = [k for k in state if k not in pkeys and not k.endswith("num_batches_tracked")]
    upd = [updates[c][1] for c in names]  # W_i - G_{t-1}, per state-dict key

    with torch.no_grad():
        vecs = torch.stack([torch.cat([u[k].reshape(-1).double() for k in pkeys]) for u in upd])
        norms = vecs.norm(dim=1)  # e_i (Alg. 1 line 8)
        unit = vecs / norms.clamp_min(1e-12).unsqueeze(1)
        dist = (1.0 - unit @ unit.T).clamp_min(0.0)  # cosine distance between updates (§4.3.1)
        dist = ((dist + dist.T) / 2).cpu().numpy().astype(np.float64)
        np.fill_diagonal(dist, 0.0)
        labels = hdbscan.HDBSCAN(metric="precomputed", min_cluster_size=n // 2 + 1, min_samples=1,
                                 allow_single_cluster=True).fit(dist).labels_
        fallback = not (labels >= 0).any()
        if fallback:
            admitted = list(range(n))
        else:
            major = np.bincount(labels[labels >= 0]).argmax()
            admitted = [i for i in range(n) if labels[i] == major]

        e = norms.cpu().numpy()
        S = float(np.median(e))  # S_t over all n updates (§4.3.2, Alg. 1 line 9)
        gamma = {i: (min(1.0, S / e[i]) if e[i] > 0 else 1.0) for i in admitted}  # line 11
        sigma = lam * S  # App. B.3

        for k in pkeys + bkeys:
            agg = sum(upd[i][k].double() * gamma[i] for i in admitted) / len(admitted)  # line 12
            state[k].add_((eta * agg).to(state[k].dtype))
            if k in pkeys:
                state[k].add_(torch.randn_like(state[k]) * sigma)  # adaptive noise

    adv = set(adversaries)
    print("SOK_FLAME " + json.dumps({
        "epoch": epoch, "clients": [str(c) for c in names], "labels": labels.tolist(),
        "admitted": [str(names[i]) for i in admitted],
        "adversaries": [str(a) for a in adversaries],
        "adversaries_admitted": [str(names[i]) for i in admitted if names[i] in adv],
        "norms": [round(float(x), 6) for x in e], "S_t": S, "sigma": sigma,
        "gamma": {str(names[i]): round(g, 6) for i, g in gamma.items()}, "fallback": bool(fallback),
    }), flush=True)
    return True
'''

VISDOM_STUB = '''class Visdom:
    """No-op stand-in for visdom.Visdom: DBA only uses it for live plots."""
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return lambda *args, **kwargs: None
'''

# DBA's logger has its own StreamHandler but no level, so INFO is dropped; set it, then run main.py.
DRIVER = """
import logging, runpy, sys
logging.getLogger("logger").setLevel(logging.INFO)
sys.argv = ["main.py", "--params", sys.argv[1]]
runpy.run_path("main.py", run_name="__main__")
"""

PIP_PACKAGES = ["hdbscan", "gdown"]
SOK_COMMIT = "__SOK_COMMIT__"  # filled in by launch/kaggle/push.sh

ON_KAGGLE = os.path.isdir("/kaggle/working")
WORK = os.environ.get("SOK_WORK", "/tmp/sok" if ON_KAGGLE else os.path.abspath("_work"))
OUT = os.environ.get("SOK_OUT", "/kaggle/working/results/flame_dba" if ON_KAGGLE
                     else os.path.abspath("_out"))
REPO = os.path.join(WORK, "dba")
STUBS = os.path.join(WORK, "stubs")


def sh(cmd, cwd=None, log=None, env=None):
    print("+", " ".join(cmd), flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", **(env or {}))
    with subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as p:
        fh = open(log, "w") if log else None
        for line in p.stdout:
            if not line.startswith("SOK_FLAME "):  # keep the Kaggle log readable
                sys.stdout.write(line)
                sys.stdout.flush()
            if fh:
                fh.write(line)
                fh.flush()
        if fh:
            fh.close()
    if p.returncode != 0:
        raise SystemExit(f"command failed ({p.returncode}): {' '.join(cmd)}")


def capture(cmd, cwd=None):
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return ""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_checkpoint():
    """DBA's saved_models.zip -> REPO/saved_models/CKPT_MEMBER. Reads only that member."""
    import gdown
    zpath = os.path.join(WORK, "ckpt", "saved_models.zip")
    os.makedirs(os.path.dirname(zpath), exist_ok=True)
    if not os.path.exists(zpath):
        gdown.download(id=CKPT_DRIVE_ID, output=zpath, quiet=False)
    with zipfile.ZipFile(zpath) as z:
        members = [m for m in z.namelist() if m.endswith(CKPT_MEMBER)]
        assert len(members) == 1, f"{CKPT_MEMBER} not found once in the zip: {z.namelist()[:50]}"
        dest = os.path.join(REPO, "saved_models", CKPT_MEMBER)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with z.open(members[0]) as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return {"source": f"https://drive.google.com/file/d/{CKPT_DRIVE_ID}", "zip_sha256": sha256(zpath),
            "zip_bytes": os.path.getsize(zpath), "member": members[0], "sha256": sha256(dest)}


def check_checkpoint(path):
    """Load with weights_only and into DBA's ResNet18, before any run starts."""
    import torch
    sys.path.insert(0, REPO)
    from models.resnet_cifar import ResNet18
    sys.path.pop(0)
    ck = torch.load(path, map_location="cpu", weights_only=True)
    ResNet18().load_state_dict(ck["state_dict"])  # raises on any mismatch
    return {"keys": sorted(ck.keys()), "epoch": ck.get("epoch"), "lr": ck.get("lr")}


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def summarise(folder, stdout_path):
    ma = {int(r["epoch"]): float(r["accuracy"]) for r in read_csv(f"{folder}/test_result.csv")
          if r["model"] == "global"}
    ba = {int(r["epoch"]): float(r["accuracy"]) for r in read_csv(f"{folder}/posiontest_result.csv")
          if r["model"] == "global"}
    local = {}
    for r in read_csv(f"{folder}/poisontriggertest_result.csv"):
        if r["model"] == "global" and r["trigger_name"].startswith("global_in_"):
            local.setdefault(r["trigger_name"], {})[int(r["epoch"])] = float(r["accuracy"])
    flame = []
    if os.path.exists(stdout_path):
        flame = [json.loads(l[len("SOK_FLAME "):]) for l in open(stdout_path) if l.startswith("SOK_FLAME ")]
    rounds = sorted(ma)
    first_attack = 3 if SMOKE else 203
    after = [ba[e] for e in sorted(ba) if e >= first_attack]
    window = [ba[e] for e in sorted(ba) if 210 <= e <= 300]
    s = {"rounds": len(rounds), "first_round": rounds[0] if rounds else None,
         "last_round": rounds[-1] if rounds else None,
         "at_round": {e: {"MA": ma.get(e), "BA": ba.get(e)} for e in REPORT_ROUNDS},
         "max_BA_from_203": max(after) if after else None,
         "mean_BA_210_300": round(statistics.mean(window), 2) if window else None,
         "final_MA": ma[rounds[-1]] if rounds else None,
         "final_BA": ba.get(rounds[-1]) if rounds else None}
    if flame:
        attacked = [r for r in flame if r["adversaries"]]
        s["flame"] = {
            "rounds_logged": len(flame),
            "fallback_rounds": sum(r["fallback"] for r in flame),
            "attacked_rounds": [{"epoch": r["epoch"], "adversaries": r["adversaries"],
                                 "admitted": r["adversaries_admitted"], "S_t": r["S_t"],
                                 "adversary_norms": [n for c, n in zip(r["clients"], r["norms"])
                                                     if c in r["adversaries"]]} for r in attacked],
            "admitted_count": dict(sorted({str(k): sum(len(r["admitted"]) == k for r in flame)
                                           for k in range(11)}.items())),
            "mean_admitted": round(statistics.mean(len(r["admitted"]) for r in flame), 2),
        }
    curves = {"MA": [ma[e] for e in rounds], "BA": [ba.get(e) for e in rounds], "rounds": rounds,
              "local_triggers": {k: [v.get(e) for e in rounds] for k, v in local.items()}}
    return s, curves


def main():
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    if not os.path.isdir(os.path.join(REPO, ".git")):
        sh(["git", "clone", "-q", DBA_REPO, REPO])
    sh(["git", "reset", "-q", "--hard", DBA_COMMIT], cwd=REPO)
    sh(["git", "clean", "-fdq", "-e", "data", "-e", "saved_models"], cwd=REPO)
    assert capture(["git", "rev-parse", "HEAD"], cwd=REPO) == DBA_COMMIT
    for rel, old, new in DBA_PATCH:
        path = os.path.join(REPO, rel)
        text = open(path).read()
        assert text.count(old) == 1, f"patch target not found exactly once in {rel}: {old!r}"
        with open(path, "w") as f:
            f.write(text.replace(old, new))
    os.makedirs(os.path.join(REPO, "saved_models"), exist_ok=True)  # Helper uses os.mkdir below it
    with open(os.path.join(REPO, "sok_flame.py"), "w") as f:
        f.write(FLAME_SOURCE)
    os.makedirs(STUBS, exist_ok=True)
    with open(os.path.join(STUBS, "visdom.py"), "w") as f:
        f.write(VISDOM_STUB)
    with open(os.path.join(REPO, "sok_driver.py"), "w") as f:
        f.write(DRIVER)
    patched = capture(["git", "diff", "--stat"], cwd=REPO)
    print(patched, flush=True)
    sh([sys.executable, "-m", "pip", "install", "-q", *PIP_PACKAGES])

    import torch
    import yaml
    if ON_KAGGLE:
        if not torch.cuda.is_available():
            raise SystemExit("no CUDA device on Kaggle: enable a GPU (T4)")
        gpu = torch.cuda.get_device_name(0)
        if "T4" not in gpu:
            raise SystemExit(f"GPU is {gpu}, not a T4: re-run with accelerator NvidiaTeslaT4")

    record = {
        "pair": "poisoning/flame",
        "run": "FLAME Table 3 DBA row in DBA's own code, config and pretrained model "
               "(defenses/poisoning/ATTACK_PROTOCOL.md)",
        "sok_commit": SOK_COMMIT,
        "upstream": {"repo": DBA_REPO, "commit": DBA_COMMIT},
        "upstream_patch": {"sha256": hashlib.sha256(json.dumps(DBA_PATCH).encode()).hexdigest(),
                           "diff_stat": patched, "diff": capture(["git", "diff"], cwd=REPO)},
        "flame_source_sha256": hashlib.sha256(FLAME_SOURCE.encode()).hexdigest(),
        "flame_source": FLAME_SOURCE,
        "seed": SEED, "smoke": SMOKE,
        "paper_reference": PAPER_REFERENCE,
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
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

    if not SMOKE:
        record["checkpoint"] = fetch_checkpoint()
        record["checkpoint"]["check"] = check_checkpoint(os.path.join(REPO, "saved_models", CKPT_MEMBER))
        print("checkpoint:", json.dumps(record["checkpoint"]), flush=True)
        write_outputs()

    base = yaml.safe_load(open(os.path.join(REPO, "utils", "cifar_params.yaml")))
    for split, aggr in RUNS:
        name = f"{split}_{aggr}"
        params = dict(base, **COMMON_OVERRIDES, **SPLIT_OVERRIDES[split],
                      aggregation_methods=aggr, name=f"sok_{name}", environment_name=name)
        if SMOKE:
            params.update(SMOKE_OVERRIDES)
        cfg_rel = f"utils/sok_{name}.yaml"
        with open(os.path.join(REPO, cfg_rel), "w") as f:
            yaml.safe_dump(params, f)
        for old in glob.glob(os.path.join(REPO, "saved_models", f"model_sok_{name}_*")):
            shutil.rmtree(old)
        stdout_path = os.path.join(OUT, f"{name}_stdout.txt")
        t_run = time.time()
        try:  # one failed run must not stop the others
            sh([sys.executable, "sok_driver.py", cfg_rel], cwd=REPO, log=stdout_path,
               env={"PYTHONPATH": STUBS, "SOK_SEED": str(SEED)})
            error = None
        except SystemExit as e:
            error = str(e)
        folders = sorted(glob.glob(os.path.join(REPO, "saved_models", f"model_sok_{name}_*")))
        folder = folders[-1] if folders else ""
        if folder:
            shutil.copytree(folder, os.path.join(OUT, f"{name}_csv"), dirs_exist_ok=True)
        s, c = summarise(folder, stdout_path)
        curves[name] = c
        record["results"][name] = {
            "split": split, "aggregation": aggr, "config": params,
            "config_sha256": hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest(),
            "completed": error is None, "error": error,
            "run_seconds": round(time.time() - t_run, 1), **s}
        write_outputs()
        print(f"[{name}] completed={error is None} rounds={s['rounds']} final MA={s['final_MA']} "
              f"BA={s['final_BA']} mean BA 210-300={s['mean_BA_210_300']} "
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
