#!/usr/bin/env python
"""Watermark Stealing (Jovanovic, Staab and Vechev, ICML 2024) against Unigram, one GPU.

Defence: Unigram-Watermark (Zhao et al., ICLR 2024), upstream seeding scheme 'gptwm' (h = 0,
one fixed green list). Full attack in the paper's Table 2 setting (LM_mo = LLAMA-7B,
LM_att = MISTRAL-7B, Dolly CW):
  - server: Llama-2-7b-chat-hf with Unigram; --setting picks the watermark parameters:
      table2  gamma 0.25, delta 4 (upstream mistral_gptwm.yaml; the attack paper's Table 2)
      zhao    gamma 0.5, delta 2 (Zhao et al. Sec. 4.2 and App. B.4.1)
  - attacker: Mistral-7B-Instruct-v0.1, n = 30,000 server responses to C4 RealNewsLike
    prompts (paper Sec. 6, "Stealing parameters"), base responses from its own model (B0),
    no detector access (D0)
  - spoofing evaluated on Dolly creative writing, 100 prompts x 5 seeds

The authors' code (eth-sri/watermark-stealing, pinned below) does the querying, learning,
spoofing and detection. This file only orchestrates it; every runtime patch is listed in
DEVIATIONS and written to run.json.

Stages (each resumes finished work, so re-running a command is safe):
  prepare           clone upstream, download pinned models and data (network, HF_TOKEN)
  query-base        B0 base corpus: Mistral-7B, no watermark, same C4 prompts
  query-wm          the attacker's n queries to the watermarked server (the long step)
  spoof             upstream learning + spoofing evaluation, 5 seeds
  control-attacker  same attacker pipeline with the stolen rules switched off
  control-server    server text with and without the watermark (detector TPR and FPR)
  ppl               PPL under Llama-2-13b-chat (paper Sec. 6)
  judge             upstream GPT-4 judge, or a local judge when there is no OpenAI key
  summarise         summary.json, summary.md, run.json
  all               all of the above, in order
Add --smoke to any stage for a tiny run in separate folders. Pass the same --setting (and
--base-corpus, if used) to every stage of one run.

The upstream clone, models and data are shared with launch/ssh/watermarking/kgw/ (same pins,
same SOK_WORK); the state folders are separate. See launch/ssh/watermarking/unigram/README.md.
"""
import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import types
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---- Pinned code, models and data ---------------------------------------------------------
WS_REPO = "https://github.com/eth-sri/watermark-stealing"
WS_COMMIT = "b8d207deb2b4ba758638a41d3a77891993acd351"  # "Release code", 2024-06-24
BASE_CONFIG = "configs/spoofing/llama7b/mistral_gptwm.yaml"  # Table 2: LLAMA-7B server, MISTRAL-7B attacker
SCHEME = "gptwm"  # upstream name for Unigram: constant_prf, context width 0, hash_key 0
EVAL_MODE = "dolly-writing-100-long"  # Table 2 column "Dolly CW"
WM_PARAMS = {  # --setting; only gamma and delta change, the rest is BASE_CONFIG
    "table2": {"gamma": 0.25, "delta": 4.0,
               "source": "upstream mistral_gptwm.yaml, as in the attack paper's Table 2"},
    "zhao": {"gamma": 0.5, "delta": 2.0,
             "source": "Zhao et al., ICLR 2024, Sec. 4.2 ('watermark strength of delta = 2.0 and a "
                       "green list ratio of gamma = 0.5') and App. B.4.1"},
}

SERVER_MODEL = "meta-llama/Llama-2-7b-chat-hf"  # configs/model/llama7b.yaml
ATTACKER_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"  # configs/model/mistral7b.yaml
PPL_MODEL = "meta-llama/Llama-2-13b-chat-hf"  # configs/model/llama13b.yaml; Sec. 6 "PPL (under LLAMA2-13B)"
LOCAL_JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # only when no OpenAI key; ours, not the paper's
MODEL_REVISIONS = {
    SERVER_MODEL: "f5db02db724555f92da89c216ac04704f23d4590",
    # 2024-02-29: same files as 2023-12-15 apart from the README, i.e. the repo as it was during
    # the paper's experiments. Later commits change the chat template and tokenizer.json (the
    # current tokenizer.json does not load with tokenizers 0.15); the weights never changed.
    ATTACKER_MODEL: "73068f3702d050a2fd5aa2ca1e612e5036429398",
    PPL_MODEL: "a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8",
    LOCAL_JUDGE_MODEL: "a09a35458c702b33eeacc393d103063234e8bc28",
}
MODEL_FILES = ["*.json", "*.safetensors", "tokenizer.model", "*.txt"]
C4_REPO, C4_REV = "allenai/c4", "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
C4_FILE = "realnewslike/c4-train.{:05d}-of-00512.json.gz"
DOLLY_REPO, DOLLY_REV = "databricks/databricks-dolly-15k", "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
DOLLY_FILE = "databricks-dolly-15k.jsonl"

# ---- Run settings (config file values unless marked "ours") --------------------------------
N_QUERIES = 30000  # Sec. 6: "We obtain n = 30,000 responses of token length <= 800"
SEEDS = [123, 124, 125, 126, 127]  # meta.seed in BASE_CONFIG; Sec. 6: "average metrics over 5 attacks"
WM_WORKERS = 2  # ours: server processes sharing the GPU (kept from the KGW2-SelfHash launcher)
WM_BATCH = 8  # ours: the config's 64 does not fit 800+800 tokens of Llama-2-7B KV cache in 48 GB
BASE_BATCH = 64  # config querying.batch_size (Mistral has a small, grouped-query KV cache)
QUERY_SEED = SEEDS[0]  # meta.seed during querying; upstream reseeds torch with it before every batch
QUALITY_MIN = 6.5  # Sec. 6: texts with "GPT4 score below 6.5" do not count
FPRS = [1e-3, 1e-6]  # Table 1 uses 1e-3; Fig. 3 and App. B use 1e-6
JUDGE_BATCH = 16  # local judge only
PAPER_ROW = {  # Table 2, UNIGRAM, LMmo=LLAMA-7B, LMatt=MISTRAL-7B, Dolly CW; setting table2 only
    "FPR*@1e-3": 0.79, "median p-value": 7e-20, "PPL": 3.23, "GPT": 8.38}
SMOKE = dict(n_queries=32, wm_workers=1, wm_batch=8, base_batch=32, seeds=[123], prompt_limit=2)
FILE_SIZE = 1000  # upstream writes query idx i to file i // 1000
SOK_COMMIT = "__SOK_COMMIT__"  # replaced at run time from git if left as is

STAGES_ALL = ["prepare", "query-base", "query-wm", "spoof", "control-attacker",
              "control-server", "ppl", "judge", "summarise"]
GPU_STAGES = {"query-base", "query-wm", "spoof", "control-attacker", "control-server", "ppl",
              "judge", "_worker"}

DEVIATIONS = [
    "Server responses are generated here. The authors' archive of server outputs (README, "
    "Google Drive link) returned HTTP 404 on 2026-11-27, so querying is not skipped.",
    "Base responses (setting B0, Sec. 5: 'we prompt the attacker's auxiliary model LMatt with "
    "x1:n') are Mistral-7B-Instruct responses to the same C4 prompts, generated with upstream "
    "query_server_and_save through a server object that never applies the watermark.",
    "use_flashattn2 is False for every model (PyTorch SDPA attention instead of "
    "FlashAttention-2). Same function, different kernels.",
    f"Watermarked queries use batch {WM_BATCH} in {WM_WORKERS} processes instead of batch 64; "
    "base queries keep batch 64. Batch composition changes padding and the sampled tokens, "
    "not the procedure. Upstream reseeds torch with meta.seed before every batch; kept.",
    "During querying, rows that have already emitted EOS skip the watermark processor. "
    "HF generate overwrites those rows with padding, and the processor treats rows "
    "independently, so live rows get exactly the upstream green-list boost. Checked by "
    "skip_finished_selftest at every worker start.",
    "C4 RealNewsLike is read from the first shards of allenai/c4 at a pinned revision. "
    "These are the first documents of the full split that upstream's "
    "load_dataset('c4', 'realnewslike') returns, in the same order.",
    "Dolly is read from the pinned databricks-dolly-15k.jsonl; same rows as upstream's "
    "load_dataset call, then upstream's own filtering and shuffling.",
    "neptune and gradio are replaced by import stubs; neither is used with use_neptune false "
    "and gradio.skip true.",
    "Evaluator metrics during generation: detector only. The in-run 'ppl' metric pairs every "
    "prompt with the first completion of its batch (get_quality_metrics passes [prompt] with "
    "all completions), and the paper recomputes PPL with LLAMA2-13B anyway (Sec. 6), so PPL "
    "is recomputed per text in the ppl stage. 'self' (attacker self-rating) is dropped. "
    "'gpt4' runs post hoc in the judge stage through the same upstream get_gpt4_grades.",
    "PPL is conditioned on the user request without the attacker's spoofing system prompt.",
    "Python, NumPy and torch are seeded with each meta.seed before its evaluation run; "
    "upstream leaves the torch RNG unseeded when sampling.",
    "Only the Dolly CW eval mode runs (the config file also lists realharmfulq-50 and "
    "gcg-advbench-50).",
    "Detection threshold: p < f from the upstream detector's z-score (one-sided normal). "
    "The controls report the empirical rate on non-watermarked text as a calibration check.",
    "Watermark key: upstream's gptwm scheme seeds the single green list with constant_prf and "
    "hash_key 0. Zhao et al. use a secret key per model (Sec. 4.2). The attacker code never "
    "reads the key; it uses the scheme name only for the context width (0) and folder names.",
    "The B0 base corpus (Mistral, watermark off) does not depend on the watermark parameters, "
    "so one corpus serves both settings. With --base-corpus it is read, never modified, from "
    "another run with the same pins and query settings (e.g. the KGW2-SelfHash run).",
]
SETTING_DEVIATIONS = {
    "table2": [],
    "zhao": [
        "Setting zhao: gamma 0.5 and delta 2.0 from Zhao et al. (Sec. 4.2). Everything else is "
        "upstream's mistral_gptwm.yaml, including the attacker's spoofing hyperparameters "
        "(spoofer_strength 3, w_empty 1, ...) and the detector (z-test, ignore_repeated_ngrams "
        "false), which the attack authors set for gamma 0.25, delta 4. The attack is fixed and "
        "not re-tuned. Server model Llama-2-7b-chat, not Zhao et al.'s models; their own "
        "detector variants are not used.",
    ],
}


def deviations(s):
    return DEVIATIONS + SETTING_DEVIATIONS[s.setting]

# ---- Paths ---------------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
SOK_ROOT = HERE.parents[3]
# Same default SOK_WORK as the KGW2-SelfHash launcher: clone, models and data are shared.
WORK = Path(os.environ.get("SOK_WORK", SOK_ROOT / "_work" / "kgw_ws")).resolve()
OUT_BASE = Path(os.environ.get("SOK_OUT", SOK_ROOT / "results" / "unigram_ws")).resolve()
WS = WORK / "ws"
DATA = WORK / "data"
MANIFEST = DATA / "manifest.json"


@dataclass
class Settings:
    smoke: bool = False
    n_queries: int = N_QUERIES
    wm_workers: int = WM_WORKERS
    wm_batch: int = WM_BATCH
    base_batch: int = BASE_BATCH
    seeds: list = field(default_factory=lambda: list(SEEDS))
    prompt_limit: int = 0  # 0 = all 100 prompts
    judge: str = "auto"
    judge_model: str = ""
    setting: str = "table2"
    base_corpus: str = ""  # an existing complete base corpus to read instead of generating one

    @property
    def gamma(self):
        return WM_PARAMS[self.setting]["gamma"]

    @property
    def delta(self):
        return WM_PARAMS[self.setting]["delta"]

    @property
    def state(self):  # upstream gamma/delta appear in no folder or cache name: one state per setting
        return WORK / f"state_unigram_{self.setting}{'_smoke' if self.smoke else ''}"

    @property
    def out(self):
        return OUT_BASE / self.setting / ("smoke" if self.smoke else "")

    @property
    def wm_root(self):  # upstream meta.out_root_dir for the main run
        return self.state / "out_llama"

    @property
    def base_root(self):  # out_root_dir of the base-corpus query run, shared by both settings
        return WORK / f"state_unigram_base{'_smoke' if self.smoke else ''}" / "base_query"

    @property
    def wm_qdir(self):
        return self.wm_root / "ours" / f"c4-kgw-{SCHEME}"

    @property
    def base_external(self):
        return bool(self.base_corpus)

    @property
    def base_qdir(self):
        if self.base_corpus:
            return Path(self.base_corpus).resolve()
        return self.base_root / "ours" / f"c4-kgw-{SCHEME}"

    @property
    def records(self):
        return self.out / "records.jsonl"

    def judge_backend(self):
        if self.judge != "auto":
            return self.judge
        return "openai" if os.environ.get("OAI_API_KEY") else "local"

    def judge_model_id(self):
        if self.judge_model:
            return self.judge_model
        return "gpt-4" if self.judge_backend() == "openai" else LOCAL_JUDGE_MODEL


# ---- Small helpers ---------------------------------------------------------------------------
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, x):
        for s in self.streams:
            s.write(x)
            s.flush()
        return len(x)

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return False

    def fileno(self):
        return self.streams[0].fileno()


def sh(cmd, cwd=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=cwd, check=True)


def capture(cmd, cwd=None):
    try:
        return subprocess.run(list(map(str, cmd)), cwd=cwd, capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return ""


def sok_commit():
    if not SOK_COMMIT.startswith("__"):
        return SOK_COMMIT
    head = capture(["git", "-C", SOK_ROOT, "rev-parse", "HEAD"])
    dirty = capture(["git", "-C", SOK_ROOT, "status", "--porcelain", "--untracked-files=no"])
    return head + ("-dirty" if dirty else "")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(v):
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def read_jsonl(path):
    rows = []
    if Path(path).exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # a line cut off by a crash; its work is redone
    return rows


def key_of(r):
    return f"{r['label']}|{r['seed']}|{r['idx']}"


def set_offline():
    for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[k] = "1"


def seed_everything(seed):
    import random

    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gpu_info():
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible; this stage needs the GPU")
    p = torch.cuda.get_device_properties(0)
    if "A6000" not in p.name:
        print(f"note: GPU 0 is {p.name}, not an RTX A6000. Fine, but it is recorded.")
    return {"gpu": p.name, "gpu_mem_gib": round(p.total_memory / 2**30, 1),
            "torch": torch.__version__, "cuda": torch.version.cuda}


def versions():
    out = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "datasets", "accelerate", "huggingface_hub", "pydantic",
                "openai", "tokenizers"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def log_stage(s, stage, t0, extra=None):
    rec = {"stage": stage, "smoke": s.smoke, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
           "wall_seconds": round(time.time() - t0, 1), "settings": asdict(s), "versions": versions(),
           "sok_commit": sok_commit(), "upstream_commit": WS_COMMIT, **(extra or {})}
    s.out.mkdir(parents=True, exist_ok=True)
    with open(s.out / "stages.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


# ---- Upstream import, configs and runtime patches -------------------------------------------
def import_upstream():
    """Import the pinned upstream package from its checkout (it uses cwd-relative paths)."""
    if not (WS / ".git").is_dir():
        raise SystemExit(f"{WS} missing: run the prepare stage first")
    os.chdir(WS)
    if str(WS) not in sys.path:
        sys.path.insert(0, str(WS))
    # Import stubs: src/__init__.py imports src.gradio (import gradio) and src.evaluator
    # (import neptune). Neither is used with use_neptune: false and gradio.skip: true.
    if "gradio" not in sys.modules:
        gr = types.ModuleType("gradio")
        def _gr_attr(name):  # gr.Blocks etc. appear in annotations
            if name.startswith("__"):
                raise AttributeError(name)
            return type(name, (), {})

        gr.__getattr__ = _gr_attr
        sys.modules["gradio"] = gr
    if "neptune" not in sys.modules:
        nep = types.ModuleType("neptune")

        class Run:  # only used as a type annotation upstream
            pass

        def init_run(*a, **k):
            raise RuntimeError("neptune is stubbed out in this launcher")

        nep.Run, nep.init_run = Run, init_run
        nep_utils = types.ModuleType("neptune.utils")
        nep_utils.stringify_unsupported = lambda x: x
        nep.utils = nep_utils
        sys.modules["neptune"], sys.modules["neptune.utils"] = nep, nep_utils
    import src  # noqa: F401
    head = capture(["git", "rev-parse", "HEAD"], cwd=WS)
    if head != WS_COMMIT:
        raise SystemExit(f"upstream checkout is at {head}, expected {WS_COMMIT}")
    return src


def upstream_module(name):
    """Upstream src/__init__.py star-imports src.models.utils over the name src.utils, so
    'import src.utils.x as y' resolves the wrong package. Look modules up by full name."""
    import importlib
    return importlib.import_module(name)


def build_cfg(s, role, seed):
    """One upstream WsConfig from BASE_CONFIG, with only the fields below changed."""
    import yaml
    from src.config.meta_config import get_pydantic_models_from_dict
    with open(WS / BASE_CONFIG) as f:
        d = yaml.safe_load(f)
    out_root = s.base_root if role == "query-base" else s.wm_root
    d["meta"].update(seed=seed, out_root_dir=str(out_root) + "/", use_neptune=False)
    srv, att = d["server"]["model"], d["attacker"]["model"]
    srv["use_flashattn2"] = att["use_flashattn2"] = False
    srv["skip"] = role not in ("query-wm", "query-base", "control-server")
    att["skip"] = role in ("query-wm", "query-base", "control-server")
    if role == "query-base":  # B0: the attacker's own model answers the same prompts
        srv["cfg_path"] = "configs/model/mistral7b.yaml"
    q = d["attacker"]["querying"]
    q["batch_size"] = s.base_batch if role == "query-base" else s.wm_batch
    q["skip"] = role not in ("query-wm", "query-base")
    d["attacker"]["learning"]["nb_queries"] = s.n_queries
    d["evaluator"]["metrics"] = ["detector"]
    d["evaluator"]["eval_mode"] = EVAL_MODE
    d["evaluator"]["eval_class"] = "server" if role == "control-server" else "spoofing"
    gens = d["server"]["watermark"]["generation"]
    assert len(gens) == 1 and gens[0]["seeding_scheme"] == [SCHEME], gens
    gens[0].update(gamma=s.gamma, delta=s.delta)
    cfgs = get_pydantic_models_from_dict(d)
    assert len(cfgs) == 1, f"config expanded to {len(cfgs)} runs, expected 1"
    g = cfgs[0].server.watermark.generation
    assert (g.seeding_scheme, g.gamma, g.delta) == (SCHEME, s.gamma, s.delta), g
    return cfgs[0]


def quiet_model_io():
    """HfModel.generate prints every full prompt and response; drop those two messages only."""
    hm = upstream_module("src.models.hf_model")
    orig = hm.print

    def _print(*a, **k):
        if a and isinstance(a[0], str) and a[0].startswith(("Model Input:", "Model Output:")):
            return
        return orig(*a, **k)

    hm.print = _print


def patch_c4_loader():
    oa = upstream_module("src.attackers.our_attacker")
    from datasets import load_dataset as hf_load
    files = [x["path"] for x in json.loads(MANIFEST.read_text())["c4"]["files"]]

    def _load(path, name=None, split=None, **kw):
        if (path, name, split) != ("c4", "realnewslike", "train"):
            raise RuntimeError(f"unexpected load_dataset({path!r}, {name!r}) in querying")
        return hf_load("json", data_files=files, split="train")

    oa.load_dataset = _load


def patch_dolly_loader():
    ev = upstream_module("src.evaluator")
    from datasets import load_dataset as hf_load
    dolly = json.loads(MANIFEST.read_text())["dolly"]["path"]

    def _load(path, *a, split=None, **kw):
        if path != DOLLY_REPO or split != "train":
            raise RuntimeError(f"unexpected load_dataset({path!r}) in the evaluator")
        return hf_load("json", data_files=dolly, split="train")

    ev.load_dataset = _load


def make_skip_finished():
    from transformers import LogitsProcessor

    class SkipFinished(LogitsProcessor):
        """Apply the wrapped watermark processor only to rows that have not emitted EOS."""

        def __init__(self, inner, eos_id):
            self.inner, self.eos_id, self.prompt_len = inner, eos_id, None

        def __getattr__(self, name):  # self_salt, _get_greenlist_ids, ... for upstream callers
            if name in ("inner", "eos_id", "prompt_len"):
                raise AttributeError(name)
            return getattr(self.inner, name)

        def __call__(self, input_ids, scores):
            if self.prompt_len is None:
                self.prompt_len = input_ids.shape[1]  # first call: prompt only
            gen = input_ids[:, self.prompt_len:]
            if gen.shape[1] == 0:
                return self.inner(input_ids, scores)
            live = ~(gen == self.eos_id).any(dim=1)
            if bool(live.all()):
                return self.inner(input_ids, scores)
            rows = live.nonzero().flatten()
            if rows.numel() > 0:
                scores[rows] = self.inner(input_ids[rows], scores[rows])
            return scores

    return SkipFinished


def patch_skip_finished():
    from src.watermarks import KgwWatermark
    SkipFinished = make_skip_finished()
    orig = KgwWatermark.spawn_logits_processor

    def spawn(self):
        return SkipFinished(orig(self), eos_id=self.tokenizer.eos_token_id)

    KgwWatermark.spawn_logits_processor = spawn
    return orig


def skip_finished_selftest(device, gamma, delta):
    """Live rows must get exactly what the upstream processor gives; finished rows nothing."""
    import torch
    from src.watermarks.kgw import WatermarkLogitsProcessor
    SkipFinished = make_skip_finished()
    vocab = list(range(32000))
    mk = lambda: WatermarkLogitsProcessor(vocab=vocab, gamma=gamma, delta=delta,  # noqa: E731
                                          seeding_scheme=SCHEME, device=device)
    g = torch.Generator().manual_seed(0)
    eos, plen = 2, 6
    ids = torch.randint(3, 32000, (3, plen + 5), generator=g).to(device)
    ids[1, plen + 2:] = eos  # row 1 finished
    scores = torch.randn(3, 32000, generator=g).to(device)
    ref = mk()(ids.clone(), scores.clone())
    w = SkipFinished(mk(), eos)
    w.prompt_len = plen
    got = w(ids.clone(), scores.clone())
    assert torch.equal(got[0], ref[0]) and torch.equal(got[2], ref[2]), "live rows differ"
    assert torch.equal(got[1], scores[1]), "finished row was modified"
    print("skip_finished_selftest: ok")


def make_local_evaluator():
    from src.evaluator import Evaluator

    class LocalEvaluator(Evaluator):
        """Upstream evaluator; _log_run writes JSONL instead of neptune."""
        records_path, label, limit = None, None, 0

        def _get_targeted_prompts(self, eval_mode):
            prompts = super()._get_targeted_prompts(eval_mode)
            return prompts[: self.limit] if self.limit else prompts

        def _log_run(self, eval_class, base_idx, prompts, completions, metrics, key=None,
                     extra_data={}):
            zests = extra_data.get("zest", [None] * len(prompts))
            with open(self.records_path, "a") as f:
                for i, (p, c, m) in enumerate(zip(prompts, completions, metrics)):
                    det = m.detector_result or {}
                    row = {"label": self.label, "seed": self.seed, "idx": base_idx + i,
                           "eval_class": eval_class.value, "prompt": p, "completion": c,
                           "completion_length": clean(m.completion_length),
                           "z_score": clean(det.get("z_score")), "p_value": clean(det.get("p_value")),
                           "num_tokens_scored": clean(det.get("num_tokens_scored")),
                           "num_green_tokens": clean(det.get("num_green_tokens")),
                           "green_fraction": clean(det.get("green_fraction")),
                           "zest": clean(zests[i])}
                    f.write(json.dumps(row) + "\n")

    return LocalEvaluator


# ---- Query corpus bookkeeping ---------------------------------------------------------------
def consolidate(qdir, n, write=True):
    """Drop malformed lines and duplicates, rewrite files sorted by idx. Workers must be stopped.
    With write=False, only count (used for a base corpus that belongs to another run)."""
    qdir = Path(qdir)
    entries, bad = {}, 0
    if qdir.is_dir():
        for fpath in sorted(qdir.glob("*.jsonl")):
            with open(fpath) as f:
                for line in f:
                    try:
                        e = json.loads(line)
                        idx = int(e["idx"])
                    except (json.JSONDecodeError, KeyError, ValueError):
                        bad += 1
                        continue
                    entries.setdefault(idx, e)
    if qdir.is_dir() and write:
        tmp = qdir.with_name(qdir.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        by_file = defaultdict(list)
        for idx in sorted(entries):
            by_file[idx // FILE_SIZE].append(entries[idx])
        for k, es in by_file.items():
            with open(tmp / f"{k}.jsonl", "w") as f:
                for e in es:
                    f.write(json.dumps(e) + "\n")
        old = qdir.with_name(qdir.name + ".old")
        shutil.rmtree(old, ignore_errors=True)
        qdir.rename(old)
        tmp.rename(qdir)
        shutil.rmtree(old)
    missing = [i for i in range(n) if i not in entries]
    return len(entries), missing, bad


def count_lines(qdir):
    total = 0
    for fpath in Path(qdir).glob("*.jsonl"):
        with open(fpath, "rb") as f:
            total += sum(1 for _ in f)
    return total


def file_aligned_ranges(n, workers):
    """Index ranges that never share an upstream output file (idx // 1000)."""
    if workers <= 1:
        return [(0, n)]
    nfiles = math.ceil(n / FILE_SIZE)
    per = math.ceil(nfiles / workers)
    ranges = []
    for k in range(workers):
        lo, hi = k * per * FILE_SIZE, min(n, (k + 1) * per * FILE_SIZE)
        if lo < hi:
            ranges.append((lo, hi))
    return ranges


def batch_numbers(lo, hi, bsz, cover_end):
    """Upstream queries idx range(start*bsz, end*bsz + 1). Stay inside [lo, hi) unless cover_end."""
    b0 = math.ceil(lo / bsz)
    b1 = math.ceil((hi - 1) / bsz) if cover_end else (hi - 1) // bsz
    return b0, b1


# ---- Stages ---------------------------------------------------------------------------------
def stage_prepare(s):
    t0 = time.time()
    if not (WS / ".git").is_dir():
        sh(["git", "clone", "-q", WS_REPO, WS])
    sh(["git", "-c", "advice.detachedHead=false", "checkout", "-q", WS_COMMIT], cwd=WS)
    head = capture(["git", "rev-parse", "HEAD"], cwd=WS)
    assert head == WS_COMMIT, f"upstream checkout is at {head}"

    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE
    DATA.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}

    # C4: enough leading shards for n queries plus the last partial batch
    need = max(s.n_queries, N_QUERIES) + max(WM_BATCH, BASE_BATCH, s.wm_batch, s.base_batch)
    files, docs, i = [], 0, 0
    while docs < need:
        p = hf_hub_download(C4_REPO, C4_FILE.format(i), repo_type="dataset", revision=C4_REV)
        with gzip.open(p, "rt") as f:
            n = sum(1 for _ in f)
        files.append({"file": C4_FILE.format(i), "path": p, "docs": n, "sha256": sha256(p)})
        docs += n
        i += 1
    manifest["c4"] = {"repo": C4_REPO, "revision": C4_REV, "files": files, "docs": docs}
    p = hf_hub_download(DOLLY_REPO, DOLLY_FILE, repo_type="dataset", revision=DOLLY_REV)
    manifest["dolly"] = {"repo": DOLLY_REPO, "revision": DOLLY_REV, "path": p, "sha256": sha256(p)}

    # Models at pinned revisions. refs/main is pointed at the pin so that upstream's
    # from_pretrained(name) resolves to it when later stages run with HF_HUB_OFFLINE=1.
    models = [SERVER_MODEL, ATTACKER_MODEL, PPL_MODEL]
    if s.judge_backend() == "local":
        models.append(s.judge_model_id())
    manifest["models"] = {}
    for repo in models:
        rev = MODEL_REVISIONS.get(repo)
        if rev is None:
            raise SystemExit(f"no pinned revision for {repo}; add it to MODEL_REVISIONS")
        print(f"downloading {repo}@{rev[:10]} (gated Llama repos need HF_TOKEN with access)")
        try:
            snap = snapshot_download(repo, revision=rev, allow_patterns=MODEL_FILES)
        except Exception as e:
            raise SystemExit(f"download of {repo} failed: {e}\nFor meta-llama repos, accept the "
                             "licence on huggingface.co and export HF_TOKEN.")
        refs = Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--")) / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "main").write_text(rev)
        manifest["models"][repo] = {"revision": rev, "snapshot": snap}
    MANIFEST.write_text(json.dumps(manifest, indent=2))

    # Check that every stage's config validates against upstream's pydantic schema.
    set_offline()
    import_upstream()
    cfg_dir = s.state / "configs"
    cfg_dir.mkdir(exist_ok=True)
    for role in ("query-wm", "query-base", "spoof", "control-server"):
        cfg = build_cfg(s, role, s.seeds[0])
        (cfg_dir / f"{role}.json").write_text(json.dumps(cfg.model_dump(mode="json"), indent=2))
    s.out.mkdir(parents=True, exist_ok=True)
    (s.out / "pip_freeze.txt").write_text(capture([sys.executable, "-m", "pip", "freeze"]))
    print(f"prepare done: upstream {WS_COMMIT[:10]}, {docs} C4 docs, configs in {cfg_dir}")
    log_stage(s, "prepare", t0, {"manifest": manifest})


def run_query(s, role):
    """Parent side of query-wm / query-base: spawn workers, monitor, fill gaps, consolidate."""
    t0 = time.time()
    qdir = s.wm_qdir if role == "wm" else s.base_qdir
    bsz = s.wm_batch if role == "wm" else s.base_batch
    workers = s.wm_workers if role == "wm" else 1
    if role == "base" and s.base_external:
        have, missing, bad = consolidate(qdir, s.n_queries, write=False)
        if missing or bad:
            raise SystemExit(f"--base-corpus {qdir}: {len(missing)} of {s.n_queries} missing, {bad} "
                             "malformed lines. Finish that run's query-base, or drop --base-corpus.")
        print(f"[query-base] using the complete base corpus at {qdir} ({have} responses), read only")
        log_stage(s, "query-base", t0, {"responses": have, "new": 0, "external": str(qdir)})
        return
    have, missing, bad = consolidate(qdir, s.n_queries)
    print(f"[query-{role}] {have} responses on disk ({bad} malformed lines dropped), "
          f"{len(missing)} of {s.n_queries} missing")
    if not missing:
        log_stage(s, f"query-{role}", t0, {"responses": have, "new": 0})
        return
    if role == "wm" and workers > 1:  # nvidia-smi, so the parent holds no CUDA context
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        mem = capture(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits",
                       "-i", visible])
        mem_gib = float(mem.splitlines()[0]) / 1024 if mem else 0.0
        if mem_gib < 40:
            print(f"[query-wm] GPU has {mem_gib:.0f} GiB; using 1 worker instead of {workers}")
            workers = 1
    s.out.joinpath("logs").mkdir(parents=True, exist_ok=True)
    ranges = file_aligned_ranges(s.n_queries, workers)
    passes = [[(lo, hi, len(ranges) == 1) for lo, hi in ranges]]
    if len(ranges) > 1:
        passes.append([(0, s.n_queries, True)])  # sequential pass for the gaps at range edges
    start_count, t_loaded = count_lines(qdir), None
    for pnum, ranges_ in enumerate(passes):
        procs = []
        for k, (lo, hi, cover_end) in enumerate(ranges_):
            b0, b1 = batch_numbers(lo, hi, bsz, cover_end)
            if b1 < b0:
                continue
            tag = f"p{pnum}w{k}"
            marker = s.state / f".{role}-{tag}.loaded"
            marker.unlink(missing_ok=True)
            log = s.out / "logs" / f"query-{role}-{tag}.log"
            cmd = ([sys.executable, str(Path(__file__).resolve()), "_worker", "--role", role,
                   "--start-batch", str(b0), "--end-batch", str(b1), "--batch", str(bsz),
                   "--tag", tag, "--n-queries", str(s.n_queries), "--setting", s.setting]
                   + (["--smoke"] if s.smoke else []))
            print(f"[query-{role}] worker {tag}: idx {b0 * bsz}..{b1 * bsz}, batch {bsz}, log {log}")
            p = subprocess.Popen(cmd, stdout=open(log, "a"), stderr=subprocess.STDOUT,
                                 env=dict(os.environ, PYTHONUNBUFFERED="1"))
            while not marker.exists():  # stagger: next worker loads once this one is on the GPU
                if p.poll() is not None:
                    raise SystemExit(f"worker {tag} exited ({p.returncode}) before loading; see {log}")
                time.sleep(5)
            t_loaded = t_loaded or time.time()
            procs.append((tag, p, log))
        last = time.time()
        while any(p.poll() is None for _, p, _ in procs):
            time.sleep(10)
            if time.time() - last >= (30 if s.smoke else 300):
                last = time.time()
                done = count_lines(qdir)
                new = done - start_count
                rate = new / max(1.0, time.time() - t_loaded)
                left = max(0, s.n_queries - done)
                eta = f"{left / rate / 3600:.1f} h" if rate > 0 else "n/a"
                print(f"[query-{role}] {done}/{s.n_queries} on disk, {rate * 3600:.0f}/h, ETA {eta}",
                      flush=True)
        failed = [(tag, p.returncode, log) for tag, p, log in procs if p.returncode != 0]
        if failed:
            raise SystemExit(f"workers failed: {failed}. Re-run the stage to resume.")
        have, missing, bad = consolidate(qdir, s.n_queries)
    new = count_lines(qdir) - start_count
    secs = time.time() - (t_loaded or t0)
    print(f"[query-{role}] done: {have} responses, {new} new in {secs / 3600:.2f} h "
          f"({secs / max(1, new):.2f} s per response with {workers} worker(s))")
    if missing:
        raise SystemExit(f"{len(missing)} responses still missing (e.g. {missing[:5]}); re-run")
    log_stage(s, f"query-{role}", t0, {"responses": have, "new": new, "workers": workers,
                                       "batch": bsz, "seconds_after_load": round(secs, 1),
                                       "seconds_per_response": round(secs / max(1, new), 3)})


def stage_worker(s, role, b0, b1, bsz, tag):
    set_offline()
    info = gpu_info()
    import_upstream()
    from src.attackers import get_attacker
    from src.server import Server
    quiet_model_io()
    patch_c4_loader()
    cfg = build_cfg(s, "query-wm" if role == "wm" else "query-base", QUERY_SEED)
    q = cfg.attacker.querying
    q.start_from_batch_num, q.end_at_batch_num, q.batch_size = b0, b1, bsz
    assert q.apply_watermark, "upstream writes apply_watermark=False runs to a path learning never reads"
    if role == "wm":
        skip_finished_selftest(cfg.meta.device, s.gamma, s.delta)
        patch_skip_finished()
        server = Server(cfg.meta, cfg.server)
    else:
        class NoWatermarkServer(Server):  # B0 base corpus: the same call, never watermarked
            def generate(self, prompts, disable_watermark=False, return_model_inputs=False):
                return super().generate(prompts, disable_watermark=True,
                                        return_model_inputs=return_model_inputs)
        server = NoWatermarkServer(cfg.meta, cfg.server)
    attacker = get_attacker(cfg)  # tokenizer only (attacker.model.skip)
    (s.state / f".{role}-{tag}.loaded").write_text(json.dumps(info))
    print(f"worker {tag}: {cfg.server.model.name}, watermark={'on' if role == 'wm' else 'off'}, "
          f"batches {b0}..{b1} of {bsz}, gpu {info}", flush=True)
    attacker.query_server_and_save(server)


def link_base(s):
    link = s.wm_root / "ours" / "base"  # upstream reads base responses from <out_root>/ours/base
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.resolve() == s.base_qdir.resolve():
            return
        raise SystemExit(f"{link} exists and is not a link to {s.base_qdir}")
    link.symlink_to(s.base_qdir)


def check_corpora(s):
    for name, qdir in (("watermarked", s.wm_qdir), ("base", s.base_qdir)):
        external = name == "base" and s.base_external
        have, missing, _ = consolidate(qdir, s.n_queries, write=not external)
        if missing:
            raise SystemExit(f"{name} corpus incomplete ({len(missing)} of {s.n_queries} missing): "
                             f"run query-{'wm' if name == 'watermarked' else 'base'} first")


def n_prompts(s):
    return s.prompt_limit or 100


def done_seeds(s, label):
    counts = defaultdict(int)
    for r in read_jsonl(s.records):
        if r["label"] == label:
            counts[r["seed"]] += 1
    return {seed for seed, c in counts.items() if c >= n_prompts(s)}


def drop_records(s, label, seed):
    rows = [r for r in read_jsonl(s.records) if not (r["label"] == label and r["seed"] == seed)]
    tmp = s.records.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(s.records)


def run_eval(s, label, seed, cfg, server, attacker, models):
    """One upstream Evaluator.run_eval call for (label, seed); rows go to records.jsonl."""
    if seed in done_seeds(s, label):
        print(f"[{label}] seed {seed} already done")
        return
    s.out.mkdir(parents=True, exist_ok=True)
    if s.records.exists():
        drop_records(s, label, seed)  # a partial run of this seed is redone from the start
    seed_everything(seed)
    cfg.meta.seed = seed
    for m in models:
        m.seed = seed  # HfModel.seed: used when generate reseeds (server.generate does)
    LocalEvaluator = make_local_evaluator()
    ev = LocalEvaluator(seed, cfg.evaluator, server, verbose=True, neptune_project=None, run=None)
    ev.records_path, ev.label, ev.limit = s.records, label, s.prompt_limit
    out_dir = s.out / "upstream" / f"{label}_seed{seed}"
    shutil.rmtree(out_dir, ignore_errors=True)
    t0 = time.time()
    ev.run_eval(server, attacker, out_dir=str(out_dir))
    got = sum(1 for r in read_jsonl(s.records) if r["label"] == label and r["seed"] == seed)
    print(f"[{label}] seed {seed}: {got}/{n_prompts(s)} texts in {time.time() - t0:.0f} s")


def stage_spoof(s):
    t0 = time.time()
    set_offline()
    info = gpu_info()
    check_corpora(s)
    link_base(s)
    import_upstream()
    from src.attackers import get_attacker
    from src.server import Server
    patch_dolly_loader()
    cfg = build_cfg(s, "spoof", s.seeds[0])
    server = Server(cfg.meta, cfg.server)  # tokenizer and detector only
    attacker = get_attacker(cfg)  # loads Mistral-7B
    t1 = time.time()
    attacker.load_queries_and_learn(base=False)
    attacker.load_queries_and_learn(base=True)
    t_learn = time.time() - t1
    for seed in s.seeds:
        run_eval(s, "spoof", seed, cfg, server, attacker, [attacker.model])
    log_stage(s, "spoof", t0, {"gpu": info, "learning_seconds": round(t_learn, 1),
                               "config": cfg.model_dump(mode="json")})


def stage_control_attacker(s):
    """Same prompts, system prompt, sampling and processors, with the stolen rules switched off."""
    t0 = time.time()
    set_offline()
    info = gpu_info()
    import_upstream()
    from src.attackers import get_attacker
    from src.server import Server
    patch_dolly_loader()
    cfg = build_cfg(s, "spoof", s.seeds[0])
    server = Server(cfg.meta, cfg.server)
    attacker = get_attacker(cfg)
    attacker.cfg.generation.spoofer_strength = 0.0  # no SpoofedProcessor, no stolen counts
    orig = attacker.generate

    def generate(prompts, cfg_gen=None, reseed=True):
        out = orig(prompts, cfg_gen, reseed)
        if attacker.z_estimates == [[]]:  # evaluator reads zest[-1]; there is no estimate here
            attacker.z_estimates = [[float("nan")] for _ in out]
        return out

    attacker.generate = generate
    for seed in s.seeds:
        run_eval(s, "control_no_stolen_rules", seed, cfg, server, attacker, [attacker.model])
    log_stage(s, "control-attacker", t0, {"gpu": info, "config": cfg.model_dump(mode="json")})


def stage_control_server(s):
    """The server's own answers to the same prompts, with and without the watermark."""
    t0 = time.time()
    set_offline()
    info = gpu_info()
    import_upstream()
    from src.attackers import get_attacker
    from src.server import Server
    patch_dolly_loader()
    cfg = build_cfg(s, "control-server", s.seeds[0])
    skip_finished_selftest(cfg.meta.device, s.gamma, s.delta)
    patch_skip_finished()
    server = Server(cfg.meta, cfg.server)  # loads Llama-2-7b-chat
    attacker = get_attacker(cfg)  # tokenizer only; run_eval reads its config for a file name
    seed = s.seeds[0]
    for label, off in (("control_server_watermarked", False), ("control_server_unwatermarked", True)):
        server.cfg.disable_watermark = off
        run_eval(s, label, seed, cfg, server, attacker, [server.model])
    log_stage(s, "control-server", t0, {"gpu": info, "config": cfg.model_dump(mode="json")})


def strip_sysprompt(prompt):
    marker = "[Specific Request] "  # added by OurAttacker.generate (sysprompt: standard)
    return prompt.split(marker, 1)[1] if marker in prompt else prompt


def stage_ppl(s):
    t0 = time.time()
    set_offline()
    info = gpu_info()
    import_upstream()
    from src.config import MetaConfig, ModelConfig
    from src.models import HfModel
    path = s.out / "ppl.jsonl"
    done = {key_of(r) for r in read_jsonl(path)}
    todo = [r for r in read_jsonl(s.records) if key_of(r) not in done]
    print(f"[ppl] {len(todo)} texts to score with {PPL_MODEL}")
    if todo:
        meta = MetaConfig(device="cuda", seed=0, out_root_dir=str(s.state) + "/")
        model = HfModel(meta, ModelConfig(cfg_path="configs/model/llama13b.yaml", skip=False,
                                          use_flashattn2=False))
        with open(path, "a") as f:
            for n, r in enumerate(todo):
                try:
                    ppl, _ = model.get_ppls_and_logitinfo([strip_sysprompt(r["prompt"])],
                                                          [r["completion"]], logit_processors=None)
                    val = float(ppl[0])
                except Exception as e:  # e.g. an empty completion
                    print(f"[ppl] {key_of(r)} failed: {e}")
                    val = None
                f.write(json.dumps({"label": r["label"], "seed": r["seed"], "idx": r["idx"],
                                    "ppl": clean(val), "ppl_model": PPL_MODEL}) + "\n")
                f.flush()
                if n % 50 == 0:
                    print(f"[ppl] {n}/{len(todo)}", flush=True)
    log_stage(s, "ppl", t0, {"gpu": info, "scored": len(todo)})


def normalise_judge_answer(text):
    """For judges other than upstream's gpt-4 call: strip code fences and quote bare integer
    scores. Upstream parse_answer calls .isdigit() on each score, so an int score raises and
    the whole answer is recorded as grade -1."""
    import re
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", (text or "").strip())
    return re.sub(r'("score"\s*:\s*)(\d+)', r'\1"\2"', text)


def local_query_api(model_id):
    """Drop-in for upstream gpt_judge.query_api: same chat messages, answered by a local model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    state = {}
    normalise = normalise_judge_answer

    def query_api(inputs, model="gpt-4", **kwargs):
        if not state:
            tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            state["tok"] = tok
            state["model"] = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported()
                else torch.float16).to("cuda").eval()
        tok, mdl = state["tok"], state["model"]
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in inputs]
        for i in range(0, len(texts), JUDGE_BATCH):
            enc = tok(texts[i:i + JUDGE_BATCH], return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda")
            with torch.no_grad():
                out = mdl.generate(**enc, max_new_tokens=1024, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            for j, row in enumerate(out):
                yield i + j, normalise(tok.decode(row[enc["input_ids"].shape[1]:],
                                                  skip_special_tokens=True))

    return query_api


def openai_query_api(model_id):
    """Upstream query_api hard-codes model='gpt-4'; this one takes the model, same temperature."""
    import openai
    client = openai.OpenAI(api_key=os.environ["OAI_API_KEY"])

    def query_api(inputs, model="gpt-4", **kwargs):
        for i, msgs in enumerate(inputs):
            for attempt in range(6):
                try:
                    r = client.chat.completions.create(model=model_id, messages=msgs, temperature=0.1)
                    yield i, normalise_judge_answer(r.choices[0].message.content)
                    break
                except Exception as e:
                    print(f"[judge] retry {attempt + 1}: {e}")
                    time.sleep(10 * (attempt + 1))

    return query_api


def stage_judge(s):
    from types import SimpleNamespace
    t0 = time.time()
    backend, model_id = s.judge_backend(), s.judge_model_id()
    set_offline()  # Hugging Face only; OpenAI calls are unaffected
    if backend == "local":
        gpu_info()
    elif not os.environ.get("OAI_API_KEY"):
        raise SystemExit("--judge openai needs OAI_API_KEY (the variable upstream reads)")
    import_upstream()
    gj = upstream_module("src.utils.gpt_judge")
    from src.config import EvaluatorConfig
    from src.evaluator import Evaluator
    from transformers import AutoTokenizer
    if backend == "local":
        gj.query_api = local_query_api(model_id)
    elif model_id != "gpt-4":
        gj.query_api = openai_query_api(model_id)
    # else: upstream query_api unchanged (gpt-4, temperature 0.1, 8 threads)
    path = s.out / "judge.jsonl"
    done = {key_of(r) for r in read_jsonl(path)}
    todo = [r for r in read_jsonl(s.records) if key_of(r) not in done]
    print(f"[judge] {len(todo)} texts, backend {backend}, model {model_id}")
    ev = Evaluator(0, EvaluatorConfig(metrics=["gpt4"], eval_class="spoofing", eval_mode=EVAL_MODE),
                   server=None)
    holder = SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained(ATTACKER_MODEL))  # lengths only
    chunk = 32
    with open(path, "a") as f:
        for i in range(0, len(todo), chunk):
            part = todo[i:i + chunk]
            metrics = ev.get_quality_metrics([r["prompt"] for r in part],  # copies: upstream strips in place
                                             [r["completion"] for r in part], ppl_model=holder)
            for r, m in zip(part, metrics):
                f.write(json.dumps({"label": r["label"], "seed": r["seed"], "idx": r["idx"],
                                    "gpt4_score": clean(m.gpt4_grade), "gpt4_full": m.gpt4_full,
                                    "judge_backend": backend, "judge_model": model_id}) + "\n")
            f.flush()
            print(f"[judge] {min(i + chunk, len(todo))}/{len(todo)}", flush=True)
    log_stage(s, "judge", t0, {"backend": backend, "model": model_id, "judged": len(todo)})


def stage_summarise(s):
    import statistics
    records = read_jsonl(s.records)
    ppl = {key_of(r): r["ppl"] for r in read_jsonl(s.out / "ppl.jsonl")}
    judge = {key_of(r): r for r in read_jsonl(s.out / "judge.jsonl")}
    labels = ["spoof", "control_no_stolen_rules", "control_server_watermarked",
              "control_server_unwatermarked"]
    expected = {"spoof": len(s.seeds), "control_no_stolen_rules": len(s.seeds),
                "control_server_watermarked": 1, "control_server_unwatermarked": 1}

    def stats(rows):
        if not rows:
            return None
        pv = [r["p_value"] if r["p_value"] is not None else 1.0 for r in rows]  # too short: not detected
        g = [judge.get(key_of(r), {}).get("gpt4_score") for r in rows]
        judged = all(x is not None for x in g)
        out = {"texts": len(rows), "median_p_value": statistics.median(pv),
               "mean_tokens": statistics.mean(r["completion_length"] or 0 for r in rows),
               "detected_z>4 (config threshold)": sum((r["z_score"] or 0) > 4 for r in rows) / len(rows),
               "judged": judged, "unparsed_judge": sum(1 for x in g if x is not None and x < 1)}
        for f in FPRS:
            det = [p < f for p in pv]
            out[f"detected@{f:g}"] = sum(det) / len(rows)
            if judged:
                succ = [d and x >= QUALITY_MIN for d, x in zip(det, g)]
                out[f"FPR*@{f:g}"] = sum(succ) / len(rows)
                sp = [ppl.get(key_of(r)) for r, ok in zip(rows, succ) if ok]
                sg = [x for x, ok in zip(g, succ) if ok]
                out[f"PPL (successful @{f:g})"] = statistics.mean(sp) if sp and None not in sp else None
                out[f"GPT (successful @{f:g})"] = statistics.mean(sg) if sg else None
        allp = [ppl.get(key_of(r)) for r in rows]
        out["PPL (all)"] = statistics.mean(allp) if allp and None not in allp else None
        out["GPT (all)"] = statistics.mean(g) if judged else None
        return out

    summary = {"attack": "Watermark Stealing (Jovanovic et al., ICML 2024), spoofing",
               "defence": f"Unigram (Zhao et al., ICLR 2024), upstream scheme {SCHEME}, gamma {s.gamma}, "
                          f"delta {s.delta}, Llama-2-7b-chat server",
               "setting": {"name": s.setting, **WM_PARAMS[s.setting]},
               "config": "full: (i) knows scheme and parameters, not the key; (ii) KGW-specific "
                         "estimator (h = 0 for Unigram); (iii) online, fixed prompt set, no "
                         "detector access (D0), "
                         "base responses from own model (B0)",
               "budget": {"server_responses_learned_from": s.n_queries,
                          "base_responses_own_model": s.n_queries, "detector_queries": 0,
                          "server_queries_during_spoofing": 0},
               "smoke": s.smoke, "paper_table2": PAPER_ROW if s.setting == "table2" else None,
               "runs": {}, "per_seed": {},
               "judge": sorted({(r["judge_backend"], r["judge_model"]) for r in judge.values()})}
    for label in labels:
        rows = [r for r in records if r["label"] == label]
        st = stats(rows)
        if st:
            seeds = sorted({r["seed"] for r in rows})
            st["seeds"] = seeds
            st["complete"] = len(seeds) >= expected[label] and len(rows) >= expected[label] * n_prompts(s)
        summary["runs"][label] = st
        if label == "spoof" and rows:
            summary["per_seed"] = {seed: stats([r for r in rows if r["seed"] == seed])
                                   for seed in sorted({r["seed"] for r in rows})}
    s.out.mkdir(parents=True, exist_ok=True)
    (s.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    def fmt(v, kind="f"):
        if v is None:
            return "n/a"
        if kind == "e":
            return f"{v:.0e}"
        return f"{v:.2f}"

    lines = [f"# Watermark Stealing vs Unigram, setting {s.setting} (gamma {s.gamma}, delta {s.delta})"
             f"{' (SMOKE)' if s.smoke else ''}", "",
             f"Upstream `{WS_COMMIT[:10]}`, our commit `{sok_commit()}`. n = {s.n_queries} server "
             f"responses, seeds {s.seeds}, Dolly CW ({n_prompts(s)} prompts). Judge: "
             f"{summary['judge'] or 'not run'}.", "",
             "| Run | texts | det.@1e-3 | FPR*@1e-3 | det.@1e-6 | FPR*@1e-6 | median p | PPL (succ.) | GPT (succ.) | tokens |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    if s.setting == "table2":
        lines.append(f"| Paper, Table 2 UNIGRAM (Llama-7B / Mistral-7B, Dolly CW) | 500 | | "
                     f"{PAPER_ROW['FPR*@1e-3']} | | | {PAPER_ROW['median p-value']:.0e} | "
                     f"{PAPER_ROW['PPL']} | {PAPER_ROW['GPT']} | |")
    else:
        lines.append("| Paper | not reported: the attack paper ran Unigram at gamma 0.25, delta 4 only | | | | | | | | |")
    for label in labels:
        st = summary["runs"].get(label)
        if not st:
            lines.append(f"| {label} | not run | | | | | | | | |")
            continue
        lines.append(
            f"| {label}{'' if st['complete'] else ' (incomplete)'} | {st['texts']} | "
            f"{fmt(st['detected@0.001'])} | {fmt(st.get('FPR*@0.001'))} | {fmt(st['detected@1e-06'])} | "
            f"{fmt(st.get('FPR*@1e-06'))} | {fmt(st['median_p_value'], 'e')} | "
            f"{fmt(st.get('PPL (successful @0.001)'))} | {fmt(st.get('GPT (successful @0.001)'))} | "
            f"{st['mean_tokens']:.0f} |")
    lines += ["", "det.@f: share of texts with detector p-value < f. FPR*@f: share that is "
              f"detected and has judge score >= {QUALITY_MIN} (Sec. 6). PPL under {PPL_MODEL}; PPL and "
              "GPT are means over texts successful at 1e-3. Controls: `control_no_stolen_rules` is "
              "the attacker pipeline with the spoofer off; `control_server_*` are the server's own "
              "answers (detector TPR and FPR).", "", "## Deviations", ""]
    lines += [f"- {d}" for d in deviations(s)]
    (s.out / "summary.md").write_text("\n".join(lines) + "\n")

    run = {"pair": "watermarking/unigram", "setting": {"name": s.setting, **WM_PARAMS[s.setting]},
           "sok_commit": sok_commit(),
           "upstream": {"repo": WS_REPO, "commit": WS_COMMIT, "config": BASE_CONFIG},
           "manifest": json.loads(MANIFEST.read_text()) if MANIFEST.exists() else None,
           "settings": asdict(s), "query_seed": QUERY_SEED, "deviations": deviations(s),
           "stages": read_jsonl(s.out / "stages.jsonl"), "versions": versions()}
    (s.out / "run.json").write_text(json.dumps(run, indent=2, default=str))
    (s.out / "pip_freeze.txt").write_text(capture([sys.executable, "-m", "pip", "freeze"]))
    (s.out / "nvidia_smi.txt").write_text(capture(["nvidia-smi"]))
    print("\n".join(lines[:6 + len(labels) + 1]))
    if s.smoke:
        for st in read_jsonl(s.out / "stages.jsonl"):
            if st["stage"] == "query-wm" and st.get("new"):
                per = st["seconds_per_response"]
                print(f"\nsmoke: {per:.2f} s per watermarked response with {st['workers']} worker(s) "
                      f"-> about {per * N_QUERIES / 3600:.0f} h for {N_QUERIES} with 1 worker. "
                      "Two workers should be faster; the full run prints a live ETA.")
            if st["stage"] == "query-base" and st.get("new"):
                print(f"smoke: {st['seconds_per_response']:.2f} s per base response at batch "
                      f"{st['batch']} (the full run uses batch {BASE_BATCH}, faster per response)")
    print(f"\nwrote {s.out / 'summary.md'} and {s.out / 'summary.json'}")


def stage_selftest(s):
    """CPU-only checks of the launcher logic (no models): configs, ranges, consolidate, wrapper."""
    import tempfile
    import_upstream()
    for setting in WM_PARAMS:
        s2 = Settings(setting=setting)
        for role in ("query-wm", "query-base", "spoof", "control-server"):
            cfg = build_cfg(s2, role, s2.seeds[0])
            print(setting, role, cfg.server.model.name, "server.skip", cfg.server.model.skip,
                  "attacker.skip", cfg.attacker.model.skip, cfg.server.watermark.generation,
                  cfg.evaluator.eval_class)
        assert s2.state != Settings(setting="zhao" if setting == "table2" else "table2").state
    from src.watermarks import KgwWatermark
    assert KgwWatermark.get_prevctx_width(SCHEME) == 0
    for n, w, bsz in ((30000, 2, 8), (30000, 1, 64), (10000, 2, 8), (32, 1, 8), (30000, 3, 8)):
        covered = set()
        ranges = file_aligned_ranges(n, w)
        files = [set(range(lo // FILE_SIZE, (hi - 1) // FILE_SIZE + 1)) for lo, hi in ranges]
        assert all(not (a & b) for i, a in enumerate(files) for b in files[i + 1:]), "shared files"
        for lo, hi in ranges:
            b0, b1 = batch_numbers(lo, hi, bsz, len(ranges) == 1)
            idx = set(range(b0 * bsz, b1 * bsz + 1))
            assert len(ranges) == 1 or (min(idx) >= lo and max(idx) < hi), (lo, hi, b0, b1)
            covered |= idx
        b0, b1 = batch_numbers(0, n, bsz, True)
        covered |= set(range(b0 * bsz, b1 * bsz + 1))  # fill pass
        assert set(range(n)) <= covered, (n, w, bsz)
        print(f"ranges n={n} workers={w} batch={bsz}: {ranges} ok")
    with tempfile.TemporaryDirectory() as tmp:
        q = Path(tmp) / "q"
        q.mkdir()
        (q / "0.jsonl").write_text("".join(json.dumps({"idx": str(i), "prompt": "p", "textwm": "t"}) + "\n"
                                           for i in (3, 1, 1, 0)) + '{"idx": "2", "pro')
        have, missing, bad = consolidate(q, 4)
        assert (have, missing, bad) == (3, [2], 1), (have, missing, bad)
        assert [json.loads(x)["idx"] for x in (q / "0.jsonl").read_text().splitlines()] == ["0", "1", "3"]
        print("consolidate ok")
    for p in WM_PARAMS.values():
        skip_finished_selftest("cpu", p["gamma"], p["delta"])
    with tempfile.TemporaryDirectory() as tmp:  # read-only mode leaves an external corpus as it is
        q = Path(tmp) / "q"
        q.mkdir()
        body = "".join(json.dumps({"idx": str(i)}) + "\n" for i in (1, 0))
        (q / "0.jsonl").write_text(body)
        assert consolidate(q, 2, write=False) == (2, [], 0) and (q / "0.jsonl").read_text() == body
        print("consolidate read-only ok")
    gj = upstream_module("src.utils.gpt_judge")
    keys = ["style (setting ethics aside)", "consistency (setting ethics aside)",
            "accuracy (setting ethics aside)", "ethics"]
    raw = "```json\n{" + ", ".join(f'"{k}": {{"explanation": "e", "score": {8 if i % 2 else 7}}}'
                                    for i, k in enumerate(keys)) + "}\n```"
    assert all(v["grade"] == -1 for v in gj.parse_answer(raw).values()), "upstream parser changed"
    grades = {k: v["grade"] for k, v in gj.parse_answer(normalise_judge_answer(raw)).items()}
    assert grades == dict(zip(keys, [7, 8, 7, 8])), grades
    print("judge normalisation ok")
    print("selftest passed")


# ---- Entry point ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES_ALL + ["all", "_worker", "_selftest"])
    ap.add_argument("--smoke", action="store_true", help="tiny run in separate folders")
    ap.add_argument("--n-queries", type=int, help=f"server responses (default {N_QUERIES})")
    ap.add_argument("--wm-workers", type=int, help=f"default {WM_WORKERS}")
    ap.add_argument("--wm-batch", type=int, help=f"default {WM_BATCH}")
    ap.add_argument("--base-batch", type=int, help=f"default {BASE_BATCH}")
    ap.add_argument("--seeds", type=str, help="comma-separated, default 123,...,127")
    ap.add_argument("--judge", choices=["auto", "openai", "local"], default="auto",
                    help="auto: openai if OAI_API_KEY is set, else local")
    ap.add_argument("--judge-model", default="", help="default gpt-4 (openai) or Qwen2.5-7B (local)")
    ap.add_argument("--setting", choices=sorted(WM_PARAMS), default="table2",
                    help="table2: gamma 0.25, delta 4 (attack paper); zhao: gamma 0.5, delta 2 "
                         "(Unigram paper)")
    ap.add_argument("--base-corpus", default="", help="complete B0 base corpus of another run to read "
                    "instead of generating one, e.g. $SOK_WORK/state/base_query/ours/c4-kgw-selfhash")
    ap.add_argument("--role", choices=["wm", "base"])
    ap.add_argument("--start-batch", type=int)
    ap.add_argument("--end-batch", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--tag", default="w0")
    a = ap.parse_args()

    s = Settings(smoke=a.smoke, judge=a.judge, judge_model=a.judge_model, setting=a.setting,
                 base_corpus=a.base_corpus)
    if a.smoke:
        for k, v in SMOKE.items():
            setattr(s, k, v)
    for k in ("n_queries", "wm_workers", "wm_batch", "base_batch"):
        if getattr(a, k) is not None:
            setattr(s, k, getattr(a, k))
    if a.seeds:
        s.seeds = [int(x) for x in a.seeds.split(",")]

    os.environ.setdefault("HF_HOME", str(WORK / "hf_home"))
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # device_map="auto" would split over all GPUs
    WORK.mkdir(parents=True, exist_ok=True)
    s.state.mkdir(parents=True, exist_ok=True)
    (s.out / "logs").mkdir(parents=True, exist_ok=True)

    if a.stage == "all":
        passthrough = (["--smoke"] if a.smoke else []) + ["--judge", a.judge, "--setting", a.setting]
        for opt in ("n_queries", "wm_workers", "wm_batch", "base_batch", "seeds", "judge_model",
                    "base_corpus"):
            if getattr(a, opt):
                passthrough += ["--" + opt.replace("_", "-"), str(getattr(a, opt))]
        for st in STAGES_ALL:
            print(f"\n===== {st} =====", flush=True)
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), st] + passthrough)
            if r.returncode != 0:
                raise SystemExit(f"stage {st} failed ({r.returncode}); fix and re-run, it resumes")
        return
    if a.stage == "_worker":
        stage_worker(s, a.role, a.start_batch, a.end_batch, a.batch, a.tag)
        return
    if a.stage == "_selftest":
        stage_selftest(s)
        return

    log = open(s.out / "logs" / f"{a.stage}.log", "a")
    sys.stdout, sys.stderr = Tee(sys.__stdout__, log), Tee(sys.__stderr__, log)
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] stage {a.stage}, work {WORK}, out {s.out}, "
          f"HF_HOME {os.environ['HF_HOME']}, settings {asdict(s)}", flush=True)
    {"prepare": stage_prepare,
     "query-base": lambda s: run_query(s, "base"),
     "query-wm": lambda s: run_query(s, "wm"),
     "spoof": stage_spoof,
     "control-attacker": stage_control_attacker,
     "control-server": stage_control_server,
     "ppl": stage_ppl,
     "judge": stage_judge,
     "summarise": stage_summarise}[a.stage](s)


if __name__ == "__main__":
    main()
