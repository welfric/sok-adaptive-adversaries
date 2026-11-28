#!/usr/bin/env python
"""Generic MemGuard (Jia et al., CCS 2019) for any saved LiRA run, then LiRA on the defended outputs.

Point it at the saved logits of a LiRA run (the target model and its shadow models) and a MemGuard
defence classifier. It applies MemGuard to every output of the target AND of every shadow model
(the attacker knows the defence), writes the defended logits in the same file layout, and can
score them with HAMP's LiRA code. Your own LiRA code can read the defended logits instead.

Inputs (all set in a JSON config, see config.example.json; every value is written to run.json):
  target_dir, shadow_dir   folders with one logits file per model, a keep (membership) mask per
                           model, and one labels file. Default layout = HAMP / Carlini LiRA:
                           <tag>_logit.npy (N, 1, C) or (N, C) or (N, A, C), <tag>_keep.npy (N,),
                           shadow_label.npy (N,)
  defence_classifier       MemGuard's defence classifier (.h5, HAMP architecture). If absent, the
                           'train-classifier' stage trains one from the target's logits and keep
                           mask with HAMP's recipe (memguard-train-attack-classifier.py)

MemGuard is not rewritten: workers run the unchanged MemGuard block of HAMP's
cifar10/lira-inference-memguard.py (identical, apart from comments and one print, to the
original jinyuan-jia/MemGuard defense_framework.py), with one line set from the data:
user_label_dim = number of classes. The blocks of HAMP's five dataset scripts differ only in that
line (checked), so this covers all of them.

Stages (each resumes):  check, train-classifier, benchmark, memguard, score, evaluate, summarise,
                        all (= check, memguard, score, evaluate, summarise)
Smoke test: '_make-smoke' writes a tiny synthetic config and data; then 'all --config <that>'.
"""
import argparse
import ast
import glob
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
SOK_ROOT = HERE.parents[3]
WORK = Path(os.environ.get("SOK_WORK", SOK_ROOT / "_work" / "memguard")).resolve()

HAMP_REPO = "https://github.com/DependableSystemsLab/MIA_defense_HAMP"
HAMP_COMMIT = "a064a1f8e9f0625ad14b1b3cfdb417fad85210f1"  # 2023-09-06, NDSS artifact version
MEMGUARD_REPO = "https://github.com/jinyuan-jia/MemGuard"  # reference: same loop as HAMP's block
MEMGUARD_COMMIT = "34e1859e37c133b6517fd01b834e1c091012b197"
BLOCK_SCRIPT = "cifar10/lira-inference-memguard.py"
MARKER = "import numpy as np\nnp.random.seed(1000)\n"
LABEL_DIM_LINE = "user_label_dim=10\n"

DEFAULTS = {  # config keys; anything not given in the config file takes these values
    "name": "memguard",
    "target_dir": None,
    "shadow_dir": None,
    "defence_classifier": None,
    "out_dir": None,  # default: results/memguard/<name>
    "logit_pattern": "{tag}_logit.npy",
    "keep_pattern": "{tag}_keep.npy",
    "labels_file": "shadow_label.npy",
    "lira_code_dir": None,  # folder with lira-score.py and lira-plot.py; default: pinned HAMP clone
    "fpr": 0.001,
    "table_variant": "online",
    "workers": 0,  # 0 = CPU cores - 1
    "shard_size": 2500,
    "bench_samples": 200,
    "notes": "",
}
CLASSIFIER_RECIPE = {  # HAMP purchase/memguard-train-attack-classifier.py
    "architecture": "Dense 256-128-64 relu, Dense 1 sigmoid (glorot_uniform seeds 1000 / 100)",
    "input": "sorted softmax of the target's outputs; label 1 = member (keep), 0 = non-member",
    "optimizer": "SGD lr 0.001", "batch_size": 256, "epochs": 400, "seed": "np.random.seed(1000)",
}
DEVIATIONS = [
    "MemGuard runs on saved logits instead of re-running model checkpoints. HAMP's CIFAR MemGuard "
    "script recomputes logits without model.eval() (batch-statistics BatchNorm); saved LiRA logits "
    "usually come from eval mode, so CIFAR numbers can differ slightly from HAMP's Table VIII.",
    "The MemGuard block runs in shards of shard_size outputs in parallel processes. The per-sample "
    "loop has no randomness (np.random.seed(1000) is set once), so shards equal one pass.",
    "One line of the block is set from the data: user_label_dim = number of classes.",
    "The block's np.save of shadow_data.npy (raw inputs, unused by LiRA scoring) gets an empty array.",
    "Logits with several queries per sample (N, A, C) are defended query by query.",
    "TensorFlow runs on the CPU, one thread per worker.",
]


# ---- helpers ----------------------------------------------------------------------------------
def sh(cmd, cwd=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=cwd, check=True)


def capture(cmd, cwd=None):
    try:
        return subprocess.run(list(map(str, cmd)), cwd=cwd, capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return ""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sok_commit():
    head = capture(["git", "-C", SOK_ROOT, "rev-parse", "HEAD"])
    dirty = capture(["git", "-C", SOK_ROOT, "status", "--porcelain", "--untracked-files=no"])
    return head + ("-dirty" if dirty else "")


def versions():
    out = {"python": platform.python_version(), "cpu_count": os.cpu_count()}
    for mod in ("numpy", "scipy", "sklearn", "tensorflow", "keras"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


class Cfg:
    def __init__(self, path):
        self.path = Path(path).resolve()
        raw = json.loads(self.path.read_text())
        unknown = set(raw) - set(DEFAULTS)
        if unknown:
            raise SystemExit(f"unknown config keys: {sorted(unknown)}; allowed: {sorted(DEFAULTS)}")
        self.raw = raw
        c = {**DEFAULTS, **raw}
        for k in ("target_dir", "shadow_dir"):
            if not c[k]:
                raise SystemExit(f"config needs {k}")
        rel = lambda p: None if p is None else (self.path.parent / p).resolve()  # noqa: E731
        self.target_dir, self.shadow_dir = rel(c["target_dir"]), rel(c["shadow_dir"])
        self.out = rel(c["out_dir"]) if c["out_dir"] else (SOK_ROOT / "results" / "memguard" / c["name"]).resolve()
        self.state = self.out / "state"
        self.h5 = rel(c["defence_classifier"]) if c["defence_classifier"] else self.out / "defence_classifier.h5"
        self.lira = rel(c["lira_code_dir"]) if c["lira_code_dir"] else WORK / "hamp" / "cifar10"
        self.c = c
        self.workers = c["workers"] or max(1, (os.cpu_count() or 2) - 1)

    def __getattr__(self, k):
        if k in DEFAULTS:
            return self.__dict__["c"][k]
        raise AttributeError(k)

    def logit(self, folder, tag):
        return folder / self.logit_pattern.format(tag=tag)

    def keep(self, folder, tag):
        return folder / self.keep_pattern.format(tag=tag)

    def tags(self, folder):
        pat = re.escape(self.logit_pattern).replace(re.escape("{tag}"), "(.+)")
        tags = [m.group(1) for p in glob.glob(str(folder / "*")) if (m := re.fullmatch(pat, Path(p).name))]
        return sorted(tags, key=lambda t: (len(t), t))

    # defended outputs and undefended links, in HAMP/Carlini layout for lira-score.py
    def mg(self, role):
        return self.state / "memguard" / role

    def src(self, role):
        return self.target_dir if role == "target" else self.shadow_dir


def log_stage(cfg, stage, t0, extra=None):
    rec = {"stage": stage, "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "wall_seconds": round(time.time() - t0, 1),
           "config": cfg.c, "versions": versions(), "sok_commit": sok_commit(), "hamp_commit": HAMP_COMMIT,
           **(extra or {})}
    cfg.out.mkdir(parents=True, exist_ok=True)
    with open(cfg.out / "stages.jsonl", "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def clone_hamp():
    if not (WORK / "hamp" / ".git").is_dir():
        WORK.mkdir(parents=True, exist_ok=True)
        sh(["git", "clone", "-q", HAMP_REPO, WORK / "hamp"])
    sh(["git", "-c", "advice.detachedHead=false", "checkout", "-q", HAMP_COMMIT], cwd=WORK / "hamp")
    assert capture(["git", "rev-parse", "HEAD"], cwd=WORK / "hamp") == HAMP_COMMIT


def load_logits(path):
    """Return (N, A, C) float32 and the original ndim, accepting (N, C), (N, 1, C) or (N, A, C)."""
    import numpy as np
    z = np.load(path)
    if z.ndim == 2:
        return z[:, None, :], 2
    if z.ndim == 3:
        return z, 3
    raise SystemExit(f"{path}: logits must be (N, C) or (N, A, C), got {z.shape}")


# ---- check ------------------------------------------------------------------------------------
def stage_check(cfg):
    import numpy as np
    t0 = time.time()
    clone_hamp()
    problems, info = [], {}
    for role in ("target", "shadow"):
        d = cfg.src(role)
        tags = cfg.tags(d) if d.is_dir() else []
        info[role] = {"dir": str(d), "models": len(tags), "tags": tags[:5] + (["..."] if len(tags) > 5 else [])}
        if not tags:
            problems.append(f"{role}: no files matching {cfg.logit_pattern} in {d}")
            continue
        if not (d / cfg.labels_file).exists():
            problems.append(f"{role}: labels file {d / cfg.labels_file} missing")
        shapes = set()
        for t in tags:
            if not cfg.keep(d, t).exists():
                problems.append(f"{role}: keep mask {cfg.keep(d, t)} missing")
            shapes.add(np.load(cfg.logit(d, t), mmap_mode="r").shape)
        info[role]["logit_shapes"] = sorted(map(str, shapes))
        if len(shapes) > 1:
            problems.append(f"{role}: logits of different shapes {shapes}")
    if not problems:
        cs = {eval(s)[-1] for r in ("target", "shadow") for s in info[r]["logit_shapes"]}
        ns = {eval(s)[0] for r in ("target", "shadow") for s in info[r]["logit_shapes"]}
        if len(cs) != 1 or len(ns) != 1:
            problems.append(f"target and shadow logits disagree on samples/classes: N {ns}, C {cs}")
        else:
            info["samples"], info["classes"] = ns.pop(), cs.pop()
    info["defence_classifier"] = {"path": str(cfg.h5), "exists": cfg.h5.exists(),
                                  "sha256": sha256(cfg.h5) if cfg.h5.exists() else None}
    if not cfg.h5.exists():
        problems.append(f"defence classifier {cfg.h5} missing: set defence_classifier in the config, or run "
                        "the 'train-classifier' stage (HAMP's recipe, from the target's logits and keep mask)")
    for n in ("lira-score.py", "lira-plot.py"):
        if not (cfg.lira / n).exists():
            problems.append(f"LiRA code {cfg.lira / n} missing (lira_code_dir)")
    cfg.out.mkdir(parents=True, exist_ok=True)
    (cfg.out / "inputs.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    if problems:
        raise SystemExit("check failed:\n  " + "\n  ".join(problems))
    print("check passed")
    log_stage(cfg, "check", t0, {"inputs": info})


def classes(cfg):
    import numpy as np
    t = cfg.tags(cfg.target_dir)[0]
    return int(np.load(cfg.logit(cfg.target_dir, t), mmap_mode="r").shape[-1])


# ---- train-classifier (only if no defence classifier is given) -----------------------------------
def stage_train_classifier(cfg):
    """HAMP's memguard-train-attack-classifier.py recipe, fed from saved logits instead of a model."""
    import numpy as np
    t0 = time.time()
    if cfg.h5.exists():
        raise SystemExit(f"{cfg.h5} exists; delete it to retrain")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    t = cfg.tags(cfg.target_dir)[0]
    z, _ = load_logits(cfg.logit(cfg.target_dir, t))
    keep = np.load(cfg.keep(cfg.target_dir, t)).astype(bool)
    z = z[:, 0, :]
    e = np.exp(z - z.max(1, keepdims=True))
    f_train = np.sort(e / e.sum(1, keepdims=True), axis=1)
    l_train = keep.astype(np.float64)
    np.random.seed(1000)
    import keras
    import tensorflow.compat.v1 as tf
    from keras.backend import set_session
    from keras.layers import Activation, Dense, Input
    from keras.models import Model
    tf.disable_v2_behavior()
    set_session(tf.Session())
    inp = Input(shape=(z.shape[1],))
    x = Dense(256, kernel_initializer=keras.initializers.glorot_uniform(seed=1000), activation="relu")(inp)
    x = Dense(128, kernel_initializer=keras.initializers.glorot_uniform(seed=1000), activation="relu")(x)
    x = Dense(64, kernel_initializer=keras.initializers.glorot_uniform(seed=1000), activation="relu")(x)
    out = Activation("sigmoid")(Dense(1, kernel_initializer=keras.initializers.glorot_uniform(seed=100))(x))
    model = Model(inputs=inp, outputs=out)
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
                  metrics=["accuracy"])
    epochs = 400 if not cfg.raw.get("_smoke") else 5  # smoke: plumbing only
    bs, idx = 256, np.arange(len(f_train))
    nb = int(np.ceil(len(f_train) / bs))
    for i in range(epochs):
        np.random.shuffle(idx)
        for j in range(nb):
            sl = idx[j * bs: min((j + 1) * bs, len(f_train))]
            model.train_on_batch(f_train[sl], l_train[sl])
        if (i + 1) % 100 == 0 or i + 1 == epochs:
            loss, acc = model.evaluate(f_train, l_train, verbose=0)
            print(f"[train-classifier] epoch {i + 1}: loss {loss:.4f}, member/non-member accuracy {acc:.4f}", flush=True)
    cfg.out.mkdir(parents=True, exist_ok=True)
    model.save(str(cfg.h5))
    print(f"saved {cfg.h5}")
    log_stage(cfg, "train-classifier", t0, {"recipe": CLASSIFIER_RECIPE, "epochs": epochs, "samples": len(f_train),
                                            "h5_sha256": sha256(cfg.h5)})


# ---- MemGuard workers: the unchanged upstream block ------------------------------------------------
def upstream_source(cfg, n_classes):
    script = WORK / "hamp" / BLOCK_SCRIPT
    src = script.read_text()
    assert src.count(MARKER) == 1, "MemGuard block marker not unique"
    block = src[src.index(MARKER):]
    assert block.count(LABEL_DIM_LINE) == 1, "user_label_dim line not found exactly once"
    block = block.replace(LABEL_DIM_LINE, f"user_label_dim={n_classes}\n")
    softmax_src = next(ast.get_source_segment(src, n) for n in ast.parse(src).body
                       if isinstance(n, ast.FunctionDef) and n.name == "softmax_by_row")
    return script, block, softmax_src


def stage_worker(cfg, role, tag, lo, hi, out_dir):
    import numpy as np
    src = cfg.src(role)
    z, _ = load_logits(cfg.logit(src, tag))
    n, a, c = z.shape
    flat = z.reshape(n * a, c)[lo:hi]
    keep = np.repeat(np.load(cfg.keep(src, tag)), a)[lo:hi]
    labels = np.repeat(np.load(src / cfg.labels_file), a)[lo:hi]
    script, block, softmax_src = upstream_source(cfg, c)
    ns = {"__name__": "hamp_memguard_block", "np": np, "os": os}
    exec(compile(softmax_src, str(script), "exec"), ns)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ns.update(args=SimpleNamespace(memguard_path=str(cfg.h5), save_tag=tag), lira_folder=str(out_dir),
              shadow_performance=(ns["softmax_by_row"](flat), labels, flat), keep=keep,
              all_shadow_data=np.zeros((0,)), all_shadow_label=labels)
    t0 = time.time()
    exec(compile(block, str(script), "exec"), ns)
    (out_dir / "done.json").write_text(json.dumps({"role": role, "tag": tag, "lo": lo, "hi": hi,
                                                   "seconds": round(time.time() - t0, 2),
                                                   "success_fraction": ns.get("success_fraction", 0.0) / max(1, hi - lo)}))


def run_worker(cfg, role, tag, lo, hi, out_dir, log):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1",
               OMP_NUM_THREADS="1", TF_CPP_MIN_LOG_LEVEL="2", PYTHONUNBUFFERED="1")
    cmd = [sys.executable, str(Path(__file__).resolve()), "_worker", "--config", str(cfg.path), "--role", role,
           "--tag", tag, "--lo", str(lo), "--hi", str(hi), "--out-dir", str(out_dir)]
    with open(log, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode


def jobs_for(cfg):
    import numpy as np
    jobs = []
    for role in ("target", "shadow"):
        d = cfg.src(role)
        for tag in cfg.tags(d):
            shp = np.load(cfg.logit(d, tag), mmap_mode="r").shape
            total = shp[0] * (shp[1] if len(shp) == 3 else 1)
            for lo in range(0, total, cfg.shard_size):
                jobs.append((role, tag, lo, min(total, lo + cfg.shard_size)))
    return jobs


def shard_dir(cfg, role, tag, lo):
    return cfg.state / "shards" / f"{role}-{tag}" / f"{lo:08d}"


def stage_memguard(cfg):
    t0 = time.time()
    clone_hamp()
    if not cfg.h5.exists():
        raise SystemExit(f"defence classifier {cfg.h5} missing: run check / train-classifier first")
    jobs = jobs_for(cfg)
    todo = [j for j in jobs if not (shard_dir(cfg, *j[:3]) / "done.json").exists()]
    (cfg.out / "logs").mkdir(parents=True, exist_ok=True)
    print(f"[memguard] {len(jobs)} shards ({len(jobs) - len(todo)} done), {len({j[:2] for j in jobs})} models, "
          f"{cfg.workers} workers", flush=True)
    t_start, failed = time.time(), []

    def one(j):
        d = shard_dir(cfg, *j[:3])
        shutil.rmtree(d, ignore_errors=True)
        return j, run_worker(cfg, *j, d, cfg.out / "logs" / f"memguard-{j[0]}-{j[1]}-{j[2]}.log")

    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        for k, (j, rc) in enumerate(ex.map(one, todo), 1):
            if rc != 0:
                failed.append(j)
            if k % cfg.workers == 0 or k == len(todo):
                el = time.time() - t_start
                print(f"[memguard] {k}/{len(todo)} shards, {el / 3600:.2f} h, ETA {el / k * (len(todo) - k) / 3600:.1f} h",
                      flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} shards failed, e.g. {failed[:3]}; see {cfg.out / 'logs'}; re-run to resume")
    merge(cfg, jobs)
    log_stage(cfg, "memguard", t0, {"shards": len(jobs), "new": len(todo)})


def merge(cfg, jobs):
    """Defended logits in the input's own layout and shape, plus links to keep masks and labels."""
    import numpy as np
    by_model = {}
    for role, tag, lo, _ in jobs:
        by_model.setdefault((role, tag), []).append(lo)
    for (role, tag), los in by_model.items():
        src, dst = cfg.src(role), cfg.mg(role)
        dst.mkdir(parents=True, exist_ok=True)
        z, ndim = load_logits(cfg.logit(src, tag))
        parts = [np.load(shard_dir(cfg, role, tag, lo) / f"{tag}_logit.npy")[:, 0, :] for lo in sorted(los)]
        out = np.concatenate(parts).reshape(z.shape).astype(z.dtype)
        np.save(dst / cfg.logit_pattern.format(tag=tag), out if ndim == 3 else out[:, 0, :])
        link(cfg.keep(src, tag), dst / cfg.keep_pattern.format(tag=tag))
    for role in ("target", "shadow"):
        link(cfg.src(role) / cfg.labels_file, cfg.mg(role) / cfg.labels_file)
    print(f"[memguard] defended logits in {cfg.mg('target')} and {cfg.mg('shadow')}")


def link(src, dst):
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


# ---- benchmark -----------------------------------------------------------------------------------
def stage_benchmark(cfg):
    t0 = time.time()
    clone_hamp()
    tag = cfg.tags(cfg.shadow_dir)[0]
    d = cfg.state / "benchmark"
    shutil.rmtree(d, ignore_errors=True)
    log = cfg.out / "logs" / "benchmark.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if run_worker(cfg, "shadow", tag, 0, cfg.bench_samples, d, log) != 0:
        raise SystemExit(f"benchmark failed; see {log}")
    per = json.loads((d / "done.json").read_text())["seconds"] / cfg.bench_samples
    total = sum(hi - lo for _, _, lo, hi in jobs_for(cfg))
    core_h = per * total / 3600
    print(f"benchmark: {per:.3f} s per output on one core; {total} outputs in all = {core_h:.0f} core-hours, "
          f"about {core_h / cfg.workers:.1f} h with {cfg.workers} workers")
    log_stage(cfg, "benchmark", t0, {"seconds_per_output": per, "outputs": total, "core_hours": core_h})


# ---- score and evaluate (HAMP's LiRA code, unchanged; or skip and use your own) ---------------------
def lira_dir(cfg, name, role):
    return cfg.state / "lira" / name / role


def stage_score(cfg):
    """Build HAMP-named folders (<tag>_logit.npy (N, A, C), <tag>_keep.npy, shadow_label.npy) for the
    undefended and the defended logits, then run lira-score.py there. Inputs are only linked to,
    never written."""
    import numpy as np
    t0 = time.time()
    for name in ("undefended", "memguard"):
        for role in ("target", "shadow"):
            src = cfg.src(role) if name == "undefended" else cfg.mg(role)
            dst = lira_dir(cfg, name, role)
            shutil.rmtree(dst, ignore_errors=True)
            dst.mkdir(parents=True)
            for tag in cfg.tags(src):
                lp = cfg.logit(src, tag)
                if np.load(lp, mmap_mode="r").ndim == 3:
                    link(lp, dst / f"{tag}_logit.npy")
                else:
                    np.save(dst / f"{tag}_logit.npy", load_logits(lp)[0])
                link(cfg.keep(src, tag), dst / f"{tag}_keep.npy")
            link(src / cfg.labels_file, dst / "shadow_label.npy")
            sh([sys.executable, cfg.lira / "lira-score.py", "--res_folder", dst], cwd=cfg.lira)
    log_stage(cfg, "score", t0)


PLOT_RE = re.compile(r"Attack type ==> (\S+)\s+([\d.]+)% TPR @ ([\d.]+)% FPR \|\s+([\d.]+)% TNR @ [\d.]+% FNR \| AUC ([\d.]+)")


def stage_evaluate(cfg):
    t0 = time.time()
    rows, raw = {}, []
    for name in ("undefended", "memguard"):
        out = subprocess.run([sys.executable, str(cfg.lira / "lira-plot.py"),
                              "--shadow_data_path", str(lira_dir(cfg, name, "shadow")),
                              "--test_data_path", str(lira_dir(cfg, name, "target")), "--fpr", str(cfg.fpr)],
                             cwd=cfg.lira, capture_output=True, text=True)
        if out.returncode != 0:
            raise SystemExit(f"lira-plot.py failed for {name}:\n{out.stderr[-2000:]}")
        rows[name] = {m.group(1): {"tpr": float(m.group(2)) / 100, "tnr_at_fnr": float(m.group(4)) / 100,
                                   "auc": float(m.group(5))} for m in PLOT_RE.finditer(out.stdout)}
        raw.append(f"### {name}\n{out.stdout}")
        print(f"[evaluate] {name}: {rows[name].get(cfg.table_variant)}")
    (cfg.out / "lira.json").write_text(json.dumps(rows, indent=2))
    (cfg.out / "lira_plot_output.txt").write_text("\n".join(raw))
    log_stage(cfg, "evaluate", t0, {"lira": rows})


# ---- summarise -------------------------------------------------------------------------------------
def softmax(z):
    import numpy as np
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def utility(cfg):
    import numpy as np
    out = {}
    tt = cfg.tags(cfg.mg("target"))
    if tt:
        t = tt[0]
        b, _ = load_logits(cfg.logit(cfg.target_dir, t))
        a, _ = load_logits(cfg.logit(cfg.mg("target"), t))
        b, a = b[:, 0, :], a[:, 0, :]
        y = np.load(cfg.target_dir / cfg.labels_file)
        k = np.load(cfg.keep(cfg.target_dir, t)).astype(bool)
        pb, pa = b.argmax(1), a.argmax(1)
        out["target"] = {"label_unchanged": float((pb == pa).mean()),
                         "member_acc_undefended": float((pb[k] == y[k]).mean()),
                         "member_acc_memguard": float((pa[k] == y[k]).mean()),
                         "nonmember_acc_undefended": float((pb[~k] == y[~k]).mean()),
                         "nonmember_acc_memguard": float((pa[~k] == y[~k]).mean()),
                         "mean_l1_change": float(np.abs(softmax(a) - softmax(b)).sum(1).mean())}
    agree = []
    for t in cfg.tags(cfg.mg("shadow")):
        b, _ = load_logits(cfg.logit(cfg.shadow_dir, t))
        a, _ = load_logits(cfg.logit(cfg.mg("shadow"), t))
        agree.append(float((a.argmax(-1) == b.argmax(-1)).mean()))
    if agree:
        out["shadows"] = {"models": len(agree), "label_unchanged_mean": sum(agree) / len(agree),
                          "label_unchanged_min": min(agree)}
    succ = [json.loads(p.read_text())["success_fraction"] for p in (cfg.state / "shards").rglob("done.json")]
    if succ:
        out["memguard_success_fraction"] = sum(succ) / len(succ)
    return out


def stage_summarise(cfg):
    lira = json.loads((cfg.out / "lira.json").read_text()) if (cfg.out / "lira.json").exists() else {}
    u = utility(cfg)
    v = cfg.table_variant
    pct = lambda x: "n/a" if x is None else f"{x * 100:.2f}%"  # noqa: E731
    ta = u.get("target", {})
    lines = [f"# LiRA vs MemGuard: {cfg.name}", "",
             f"Config `{cfg.path}`. MemGuard block from HAMP `{HAMP_COMMIT[:10]}`; LiRA code `{cfg.lira}`; "
             f"{len(cfg.tags(cfg.shadow_dir))} shadow models; variant '{v}' in the table.", "",
             f"| Defence | LiRA TPR @ {cfg.fpr * 100:g}% FPR | AUC | Non-member acc. | Member acc. |", "|---|---|---|---|---|"]
    for name, label in (("undefended", "Undefended"), ("memguard", "MemGuard")):
        r = lira.get(name, {}).get(v, {})
        auc = "n/a" if r.get("auc") is None else "%.4f" % r["auc"]
        lines.append(f"| {label} | {pct(r.get('tpr'))} | {auc} | {pct(ta.get('nonmember_acc_' + name))} | "
                     f"{pct(ta.get('member_acc_' + name))} |")
    lines += ["", "## Label check", ""]
    if ta:
        lines.append(f"- Target: predicted label unchanged on {pct(ta['label_unchanged'])} of outputs; mean L1 change "
                     f"of the confidence vector {ta['mean_l1_change']:.3f}.")
    if "shadows" in u:
        s = u["shadows"]
        lines.append(f"- Shadow models: label unchanged on {pct(s['label_unchanged_mean'])} on average "
                     f"(minimum {pct(s['label_unchanged_min'])}) over {s['models']} models.")
    if "memguard_success_fraction" in u:
        lines.append(f"- MemGuard success fraction (its own counter): {pct(u['memguard_success_fraction'])}.")
    lines += ["", "Member / non-member acc.: the target's accuracy on the samples its keep mask marks as members / "
              "non-members (non-member accuracy approximates test accuracy).", "",
              "## All LiRA variants", "", "| Defence | Variant | TPR | TNR @ FNR | AUC |", "|---|---|---|---|---|"]
    for name, vs in lira.items():
        for k, r in vs.items():
            lines.append(f"| {name} | {k} | {pct(r['tpr'])} | {pct(r['tnr_at_fnr'])} | {r['auc']:.4f} |")
    lines += ["", "## Config", "", "```json", json.dumps(cfg.c, indent=2), "```", "", "## Deviations", ""]
    lines += [f"- {d}" for d in DEVIATIONS]
    (cfg.out / "summary.md").write_text("\n".join(lines) + "\n")
    (cfg.out / "summary.json").write_text(json.dumps({"lira": lira, "utility": u, "table_variant": v}, indent=2))
    stages = [json.loads(x) for x in (cfg.out / "stages.jsonl").read_text().splitlines()] \
        if (cfg.out / "stages.jsonl").exists() else []
    run = {"tool": "generic MemGuard + LiRA", "sok_commit": sok_commit(), "config_file": str(cfg.path),
           "config": cfg.c, "config_given": cfg.raw,
           "upstream": {"hamp": {"repo": HAMP_REPO, "commit": HAMP_COMMIT, "memguard_block": BLOCK_SCRIPT},
                        "memguard_reference": {"repo": MEMGUARD_REPO, "commit": MEMGUARD_COMMIT}},
           "memguard_hyperparameters": {"c1": 1.0, "c2": 10.0, "c3_start": 0.1, "c3_factor": 10, "c3_max": 100000,
                                        "max_iteration": 300, "step": 0.1, "seed": "np.random.seed(1000)"},
           "defence_classifier": {"path": str(cfg.h5), "sha256": sha256(cfg.h5) if cfg.h5.exists() else None,
                                  "trained_here": any(s["stage"] == "train-classifier" for s in stages),
                                  "recipe_if_trained": CLASSIFIER_RECIPE},
           "inputs": json.loads((cfg.out / "inputs.json").read_text()) if (cfg.out / "inputs.json").exists() else None,
           "deviations": DEVIATIONS, "stages": stages, "versions": versions()}
    (cfg.out / "run.json").write_text(json.dumps(run, indent=2, default=str))
    print("\n".join(lines[: lines.index("## Label check") + 4]))
    print(f"\nwrote {cfg.out / 'summary.md'}")


# ---- smoke --------------------------------------------------------------------------------------
def make_smoke():
    """Tiny synthetic LiRA run (4 shadow models + target, 360 outputs, 10 classes) and its config."""
    import numpy as np
    base = WORK / "smoke"
    shutil.rmtree(base, ignore_errors=True)
    rng = np.random.default_rng(0)
    n, c, k = 360, 10, 4
    y = rng.integers(0, c, n)

    def logits(keep):
        z = rng.normal(0, 1, (n, c)).astype(np.float32)
        z[np.arange(n), y] += np.where(keep, 6.0, 3.0) + rng.normal(0, 1, n)
        return z[:, None, :]

    shadow_keep = np.array([rng.permutation(np.arange(k) < k // 2) for _ in range(n)]).T
    for role, tags in (("shadow", [str(i) for i in range(k)]), ("target", ["0"])):
        d = base / role
        d.mkdir(parents=True)
        for i, t in enumerate(tags):
            keep = (np.arange(n) < n // 2) if role == "target" else shadow_keep[i]
            np.save(d / f"{t}_logit.npy", logits(keep))
            np.save(d / f"{t}_keep.npy", keep)
        np.save(d / "shadow_label.npy", y)
    # a quickly trained defence classifier of HAMP's architecture, so MemGuard has something to
    # hide in the smoke test (HAMP's SGD recipe needs far more data and steps to leave chance level)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from keras.layers import Activation, Dense, Input
    from keras.models import Model
    inp = Input(shape=(c,))
    x = inp
    for w in (256, 128, 64):
        x = Dense(w, activation="relu")(x)
    m = Model(inp, Activation("sigmoid")(Dense(1)(x)))
    m.compile(loss="binary_crossentropy", optimizer="adam")
    z = np.load(base / "target" / "0_logit.npy")[:, 0, :]
    e = np.exp(z - z.max(1, keepdims=True))
    m.fit(np.sort(e / e.sum(1, keepdims=True), axis=1), (np.arange(n) < n // 2).astype(np.float32),
          epochs=40, verbose=0)
    m.save(str(base / "smoke_defence_classifier.h5"))
    cfg = {"name": "smoke", "target_dir": "target", "shadow_dir": "shadow", "out_dir": "out",
           "defence_classifier": "smoke_defence_classifier.h5",
           "workers": 4, "shard_size": 90, "bench_samples": 20, "_smoke": True,
           "notes": "synthetic smoke test; numbers mean nothing"}
    (base / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"smoke config: {base / 'config.json'}")


# ---- entry point ---------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["check", "train-classifier", "benchmark", "memguard", "score", "evaluate",
                                      "summarise", "all", "_worker", "_make-smoke"])
    ap.add_argument("--config", help="JSON config (see config.example.json)")
    ap.add_argument("--role", choices=["shadow", "target"])
    ap.add_argument("--tag")
    ap.add_argument("--lo", type=int)
    ap.add_argument("--hi", type=int)
    ap.add_argument("--out-dir")
    a = ap.parse_args()
    if a.stage == "_make-smoke":
        make_smoke()
        return
    if not a.config:
        raise SystemExit("--config is required (see launch/ssh/privacy/memguard/config.example.json)")
    DEFAULTS["_smoke"] = False  # internal key, only set by the smoke config
    cfg = Cfg(a.config)
    if a.stage == "_worker":
        stage_worker(cfg, a.role, a.tag, a.lo, a.hi, a.out_dir)
        return
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    stages = {"check": stage_check, "train-classifier": stage_train_classifier, "benchmark": stage_benchmark,
              "memguard": stage_memguard, "score": stage_score, "evaluate": stage_evaluate,
              "summarise": stage_summarise}
    for st in (["check", "memguard", "score", "evaluate", "summarise"] if a.stage == "all" else [a.stage]):
        print(f"\n===== {st} =====", flush=True)
        stages[st](cfg)


if __name__ == "__main__":
    main()
