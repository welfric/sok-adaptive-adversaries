#!/usr/bin/env python
"""Light paraphrase scrub vs Unigram (and KGW side by side), one GPU.

Replicates the table in the team draft (§5.5, §5.9: unwatermarked / watermarked / after scrub, each
with mean z and detection rate) with Unigram (Zhao et al., ICLR 2024) in place of KGW.

Stated in the draft and kept here: gamma 0.25, delta 4.0, detection at z > 4, ungated Mistral-7B
generator, 50 texts of 200 tokens, an attacker who paraphrases the watermarked text with model access
but not the key. Everything the draft does not state is a value in UNSTATED below, and run.json
marks each one "not stated in draft; to confirm with Rezwan".

Watermark code: the KGW library inside eth-sri/watermark-stealing at the commit the other two
watermarking launchers pin (src/watermarks/kgw). Unigram is its 'gptwm' scheme, the same code as
launch/ssh/watermarking/unigram/; KGW is its 'lefthash' scheme (h = 1, the ICML 2023 scheme).

Stages (each resumes finished work):
  prepare    clone upstream, download pinned models and data (network, no token needed)
  run        generate watermarked and unwatermarked texts, paraphrase the watermarked ones,
             detect every text (for each scheme)
  quality    perplexity of every text under a fixed oracle model
  summarise  summary.md, summary.json, run.json
  all        the four stages above, in order
Add --smoke for a tiny run with small stand-in models in separate folders.
"""
import argparse
import gzip
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOK_ROOT = HERE.parents[3]


def load_kgw_launcher():
    """The KGW2-SelfHash launcher pins the upstream clone and knows how to import it; reuse it."""
    spec = importlib.util.spec_from_file_location("sok_kgw_run", HERE.parent / "kgw" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


K = load_kgw_launcher()  # K.WS, K.WS_COMMIT, K.import_upstream, K.sha256, ...
WORK = K.WORK  # same SOK_WORK as the other watermarking launchers: one clone, one HF cache
OUT_BASE = Path(os.environ.get("SOK_OUT", SOK_ROOT / "results" / "unigram_scrub")).resolve()
STATE = WORK / "scrub"

# ---- Settings stated in the draft (§5.5, §5.9) --------------------------------------------
DRAFT = {
    "gamma": 0.25,
    "delta": 4.0,
    "z_threshold": 4.0,  # "detection at z > 4"
    "generator": "mistralai/Mistral-7B-v0.1",  # draft: "ungated Mistral-7B"; which variant is not stated
    "n_texts": 50,
    "new_tokens": 200,
    "attack": "one paraphrase of each watermarked text, with model access but not the key",
}
DRAFT_RESULT_KGW = {  # the draft's table (KGW, the team's own implementation)
    "unwatermarked": {"mean_z": -0.06, "det": 0.00},
    "watermarked": {"mean_z": 7.94, "det": 0.94},
    "after scrub": {"mean_z": 2.54, "det": 0.16},
}

# ---- Settings the draft does not state: ours, all to confirm with Rezwan ------------------
UNSTATED_NOTE = "not stated in draft; to confirm with Rezwan"
UNSTATED = {
    "generator_variant": "mistralai/Mistral-7B-v0.1 (base model, plain continuation)",
    "kgw_seeding_scheme": "lefthash (KGW ICML 2023: additive PRF, context width 1, key 15485863)",
    "unigram_seeding_scheme": "gptwm (upstream Unigram: one fixed green list, constant PRF, key 0)",
    "detector": "z-test over every scored token (ignore_repeated_ngrams false), no normalisers, "
                "the generator's tokenizer",
    "dataset": "C4 RealNewsLike, allenai/c4 shard realnewslike/c4-train.00000-of-00512 (pinned)",
    "prompts": "the first PROMPT_TOKENS tokens of each C4 document that has at least "
               "PROMPT_TOKENS + new_tokens tokens, in file order",
    "prompt_tokens": 200,
    "generation": "multinomial sampling, temperature 1.0, top_k 0, top_p 1.0, exactly new_tokens "
                  "tokens (min_new_tokens = max_new_tokens), no repetition penalty",
    "unwatermarked_text": "the same generator, prompts and seeds with the watermark off",
    "paraphraser": "mistralai/Mistral-7B-Instruct-v0.1 (no watermark), chat template",
    "paraphrase_prompt": "PARAPHRASE_PROMPT below",
    "paraphrase_sampling": "temperature 0.7, top_p 1.0, max_new_tokens 400",
    "seed": "SEED; batch b of generation uses SEED + b, batch b of paraphrasing SEED + 10000 + b",
    "batch_size": "BATCH",
}
PROMPT_TOKENS = 200
SEED = 0
BATCH = 10
GEN_KW = dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0)
PARA_KW = dict(do_sample=True, temperature=0.7, top_p=1.0, max_new_tokens=400)
PARAPHRASE_PROMPT = (
    "Paraphrase the following text. Keep its meaning and roughly the same length, and change the "
    "wording. Reply with the paraphrase only.\n\nText:\n{text}")
SCHEMES = {  # name -> upstream seeding scheme; same gamma, delta and detector for both
    "unigram": "gptwm",
    "kgw": "lefthash",
}
QUALITY_ORACLE = "Qwen/Qwen2.5-7B-Instruct"  # ours: fixed oracle for perplexity, also pinned in ../kgw
PARAPHRASER = "mistralai/Mistral-7B-Instruct-v0.1"
MODEL_REVISIONS = {
    "mistralai/Mistral-7B-v0.1": "26bca36bde8333b5d7f72e9ed20ccda6a618af24",  # 2023-12-11; later
    # revisions change tokenizer.json to a format tokenizers 0.15 cannot read; weights identical
    PARAPHRASER: K.MODEL_REVISIONS[PARAPHRASER],
    QUALITY_ORACLE: K.MODEL_REVISIONS[QUALITY_ORACLE],
    # smoke stand-ins
    "Qwen/Qwen2.5-0.5B": "060db6499f32faf8b98477b0a26969ef7d8b9987",
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
}
SMOKE = dict(n_texts=3, new_tokens=24, prompt_tokens=32, batch=3, para_max_new_tokens=64,
             generator="Qwen/Qwen2.5-0.5B", paraphraser="Qwen/Qwen2.5-0.5B-Instruct",
             oracle="Qwen/Qwen2.5-0.5B-Instruct")
KINDS = ["unwatermarked", "watermarked", "scrubbed", "human"]
STAGES_ALL = ["prepare", "run", "quality", "summarise"]


@dataclass
class Settings:
    smoke: bool = False
    n_texts: int = DRAFT["n_texts"]
    new_tokens: int = DRAFT["new_tokens"]
    prompt_tokens: int = PROMPT_TOKENS
    batch: int = BATCH
    para_max_new_tokens: int = PARA_KW["max_new_tokens"]
    generator: str = DRAFT["generator"]
    paraphraser: str = PARAPHRASER
    oracle: str = QUALITY_ORACLE
    schemes: list = field(default_factory=lambda: list(SCHEMES))
    seed: int = SEED

    @property
    def state(self):
        return STATE / ("smoke" if self.smoke else "full")

    @property
    def out(self):
        return OUT_BASE / "smoke" if self.smoke else OUT_BASE

    @property
    def records(self):
        return self.state / "records.jsonl"

    @property
    def manifest(self):
        return STATE / "manifest.json"


# ---- helpers -------------------------------------------------------------------------------
def read_jsonl(path):
    return K.read_jsonl(path)


def write_jsonl(path, rows):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)


def key(r):
    return f"{r['scheme']}|{r['kind']}|{r['idx']}"


def device_and_dtype():
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def load_lm(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev, dtype = device_and_dtype()
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(dev).eval()
    return model, tok, dev


def free():
    """Release GPU memory after the caller has deleted its model references."""
    import gc

    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def log_stage(s, stage, t0, extra=None):
    rec = {"stage": stage, "smoke": s.smoke, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
           "wall_seconds": round(time.time() - t0, 1), "settings": asdict(s), "versions": K.versions(),
           "sok_commit": K.sok_commit(), "upstream_commit": K.WS_COMMIT, **(extra or {})}
    s.out.mkdir(parents=True, exist_ok=True)
    with open(s.out / "stages.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


# ---- prepare --------------------------------------------------------------------------------
def stage_prepare(s):
    t0 = time.time()
    if not (K.WS / ".git").is_dir():
        K.sh(["git", "clone", "-q", K.WS_REPO, K.WS])
    K.sh(["git", "-c", "advice.detachedHead=false", "checkout", "-q", K.WS_COMMIT], cwd=K.WS)
    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE
    STATE.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(s.manifest.read_text()) if s.manifest.exists() else {"models": {}}
    p = hf_hub_download(K.C4_REPO, K.C4_FILE.format(0), repo_type="dataset", revision=K.C4_REV)
    manifest["c4"] = {"repo": K.C4_REPO, "revision": K.C4_REV, "file": K.C4_FILE.format(0), "path": p,
                      "sha256": K.sha256(p)}
    for repo in dict.fromkeys((s.generator, s.paraphraser, s.oracle)):
        rev = MODEL_REVISIONS[repo]
        print(f"downloading {repo}@{rev[:10]}")
        snap = snapshot_download(repo, revision=rev, allow_patterns=K.MODEL_FILES)
        refs = Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--")) / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "main").write_text(rev)  # offline from_pretrained(name) loads the pin
        manifest["models"][repo] = {"revision": rev, "snapshot": snap}
    s.manifest.write_text(json.dumps(manifest, indent=2))
    print(f"prepare done: upstream {K.WS_COMMIT[:10]}, models {list(manifest['models'])}")
    log_stage(s, "prepare", t0, {"manifest": manifest})


# ---- run: generate, paraphrase, detect ---------------------------------------------------------
def load_prompts(s, tok):
    """The first prompt_tokens tokens of the first n_texts long-enough C4 documents."""
    path = json.loads(s.manifest.read_text())["c4"]["path"]
    out = []
    with gzip.open(path, "rt") as f:
        for line in f:
            doc = json.loads(line)
            ids = tok(doc["text"], add_special_tokens=False)["input_ids"]
            if len(ids) < s.prompt_tokens + s.new_tokens:
                continue
            out.append({"doc": len(out), "url": doc.get("url"),
                        "prompt": tok.decode(ids[: s.prompt_tokens]),
                        "human": tok.decode(ids[s.prompt_tokens: s.prompt_tokens + s.new_tokens])})
            if len(out) == s.n_texts:
                return out
    raise SystemExit(f"only {len(out)} long-enough C4 documents in the shard")


def make_processor(scheme, tok, device):
    from src.watermarks.kgw import WatermarkLogitsProcessor
    return WatermarkLogitsProcessor(vocab=list(tok.get_vocab().values()), gamma=DRAFT["gamma"],
                                    delta=DRAFT["delta"], seeding_scheme=SCHEMES[scheme], device=device)


def make_detector(scheme, tok, device):
    from src.watermarks.kgw import WatermarkDetector
    return WatermarkDetector(vocab=list(tok.get_vocab().values()), gamma=DRAFT["gamma"],
                             seeding_scheme=SCHEMES[scheme], device=device, tokenizer=tok,
                             z_threshold=DRAFT["z_threshold"], normalizers=[],
                             ignore_repeated_ngrams=False)


def generate_batch(model, tok, device, prompts, new_tokens, seed, processor=None, **kw):
    import torch
    from transformers import LogitsProcessorList, set_seed
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    set_seed(seed)
    with torch.no_grad():
        out = model.generate(**enc, min_new_tokens=new_tokens, max_new_tokens=new_tokens,
                             logits_processor=LogitsProcessorList([processor]) if processor else None,
                             pad_token_id=tok.pad_token_id, **kw)
    gen = out[:, enc["input_ids"].shape[1]:]
    return [tok.decode(g, skip_special_tokens=True) for g in gen], [int((g != tok.pad_token_id).sum()) for g in gen]


def paraphrase_batch(model, tok, device, texts, s, seed):
    import torch
    from transformers import set_seed
    chats = [tok.apply_chat_template([{"role": "user", "content": PARAPHRASE_PROMPT.format(text=t)}],
                                     tokenize=False, add_generation_prompt=True) for t in texts]
    enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    kw = dict(PARA_KW, max_new_tokens=s.para_max_new_tokens)
    set_seed(seed)
    with torch.no_grad():
        out = model.generate(**enc, pad_token_id=tok.pad_token_id, **kw)
    return [tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for g in out]


def stage_run(s):
    t0 = time.time()
    K.set_offline()
    K.import_upstream()
    s.state.mkdir(parents=True, exist_ok=True)
    rows = {key(r): r for r in read_jsonl(s.records)}

    def save():
        write_jsonl(s.records, sorted(rows.values(), key=lambda r: (r["scheme"], KINDS.index(r["kind"]), r["idx"])))

    # 1) generation: watermarked and unwatermarked texts with the generator, same prompts and seeds
    model, tok, dev = load_lm(s.generator)
    prompts = load_prompts(s, tok)
    for scheme in s.schemes:
        proc = make_processor(scheme, tok, dev)
        for b0 in range(0, s.n_texts, s.batch):
            idxs = list(range(b0, min(b0 + s.batch, s.n_texts)))
            seed = s.seed + b0 // s.batch
            for kind, p in (("watermarked", proc), ("unwatermarked", None)):
                if all(f"{scheme}|{kind}|{i}" in rows for i in idxs):
                    continue
                texts, ntok = generate_batch(model, tok, dev, [prompts[i]["prompt"] for i in idxs],
                                             s.new_tokens, seed, p, **GEN_KW)
                for i, t, n in zip(idxs, texts, ntok):
                    rows[f"{scheme}|{kind}|{i}"] = {"scheme": scheme, "kind": kind, "idx": i,
                                                    "prompt": prompts[i]["prompt"], "text": t,
                                                    "gen_tokens": n, "seed": seed}
                save()
                print(f"[run] {scheme} {kind} {idxs[-1] + 1}/{s.n_texts}", flush=True)
        for i in range(s.n_texts):  # human C4 continuation, as an extra reference row
            rows[f"{scheme}|human|{i}"] = {"scheme": scheme, "kind": "human", "idx": i,
                                           "prompt": prompts[i]["prompt"], "text": prompts[i]["human"],
                                           "gen_tokens": s.new_tokens, "seed": None}
        save()
    del model
    free()

    # 2) the attack: one paraphrase of each watermarked text, by a model that has no key
    todo = [(sc, i) for sc in s.schemes for i in range(s.n_texts) if f"{sc}|scrubbed|{i}" not in rows]
    if todo:
        pmodel, ptok, pdev = load_lm(s.paraphraser)
        for sc in s.schemes:
            idxs_all = [i for (x, i) in todo if x == sc]
            for b0 in range(0, len(idxs_all), s.batch):
                idxs = idxs_all[b0: b0 + s.batch]
                seed = s.seed + 10000 + idxs[0] // s.batch
                texts = paraphrase_batch(pmodel, ptok, pdev, [rows[f"{sc}|watermarked|{i}"]["text"] for i in idxs],
                                         s, seed)
                for i, t in zip(idxs, texts):
                    rows[f"{sc}|scrubbed|{i}"] = {"scheme": sc, "kind": "scrubbed", "idx": i,
                                                  "prompt": rows[f"{sc}|watermarked|{i}"]["prompt"],
                                                  "source": rows[f"{sc}|watermarked|{i}"]["text"],
                                                  "text": t, "seed": seed}
                save()
                print(f"[run] {sc} scrubbed {idxs[-1] + 1}/{s.n_texts}", flush=True)
        del pmodel
        free()

    # 3) detection of every text with each scheme's own detector (cheap, redone every time).
    # Same device type as generation: torch.randperm draws differ between CPU and CUDA generators.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(s.generator)
    dets = {sc: make_detector(sc, tok, dev) for sc in s.schemes}
    for r in rows.values():
        try:
            d = dets[r["scheme"]].detect(r["text"])
            r.update(z=K.clean(d["z_score"]), p_value=K.clean(d["p_value"]), detected=bool(d["prediction"]),
                     tokens_scored=int(d["num_tokens_scored"]), green_fraction=K.clean(d["green_fraction"]))
        except Exception as e:  # e.g. an empty paraphrase: nothing to score, counts as not detected
            r.update(z=None, p_value=None, detected=False, tokens_scored=0, green_fraction=None,
                     detect_error=str(e)[:200])
    save()
    log_stage(s, "run", t0, {"texts": len(rows), "device": dev})


# ---- quality: perplexity under a fixed oracle ----------------------------------------------------
def stage_quality(s):
    import torch
    t0 = time.time()
    K.set_offline()
    rows = read_jsonl(s.records)
    path = s.state / "ppl.jsonl"
    done = {key(r) for r in read_jsonl(path)}
    todo = [r for r in rows if key(r) not in done]
    print(f"[quality] {len(todo)} texts to score with {s.oracle}")
    if todo:
        model, tok, dev = load_lm(s.oracle)
        with open(path, "a") as f:
            for n, r in enumerate(todo):
                p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
                c = tok(r["text"], add_special_tokens=False)["input_ids"]
                ppl = None
                if c:
                    ids = torch.tensor([p + c], device=dev)
                    with torch.no_grad():
                        logits = model(ids).logits[0, len(p) - 1: -1].float()
                    nll = torch.nn.functional.cross_entropy(logits, ids[0, len(p):], reduction="mean")
                    ppl = float(torch.exp(nll))
                f.write(json.dumps({"scheme": r["scheme"], "kind": r["kind"], "idx": r["idx"],
                                    "ppl": ppl, "oracle": s.oracle}) + "\n")
                f.flush()
                if n % 50 == 0:
                    print(f"[quality] {n}/{len(todo)}", flush=True)
        del model
        free()
    log_stage(s, "quality", t0, {"scored": len(todo)})


# ---- summarise --------------------------------------------------------------------------------
def overlap_f1(a, b):
    """Word-level unigram F1 between a paraphrase and its source (1 = same words)."""
    ta, tb = Counter(a.lower().split()), Counter(b.lower().split())
    common = sum((ta & tb).values())
    if not common:
        return 0.0
    p, r = common / sum(ta.values()), common / sum(tb.values())
    return 2 * p * r / (p + r)


def stage_summarise(s):
    rows = read_jsonl(s.records)
    ppl = {key(r): r["ppl"] for r in read_jsonl(s.state / "ppl.jsonl")}
    label = {"unwatermarked": "unwatermarked", "watermarked": "watermarked", "scrubbed": "after scrub",
             "human": "human (C4 continuation, extra)"}

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return statistics.mean(xs) if xs else None

    summary = {"draft_settings_kept": DRAFT, "draft_result_kgw": DRAFT_RESULT_KGW, "schemes": {}}
    lines = [f"# Light paraphrase scrub vs Unigram and KGW{' (SMOKE)' if s.smoke else ''}", "",
             f"gamma {DRAFT['gamma']}, delta {DRAFT['delta']}, detected = z > {DRAFT['z_threshold']}; "
             f"{s.n_texts} texts x {s.new_tokens} tokens from {s.generator}; paraphraser {s.paraphraser}; "
             f"PPL under {s.oracle}. Upstream `{K.WS_COMMIT[:10]}`, our commit `{K.sok_commit()}`.", ""]
    for sc in s.schemes:
        lines += [f"## {sc} ({SCHEMES[sc]})", "",
                  "| Text | n | mean z | Det. (z > 4) | median z | PPL (oracle) | overlap F1 with source | tokens |",
                  "|---|---|---|---|---|---|---|---|"]
        summary["schemes"][sc] = {}
        for kind in KINDS:
            g = [r for r in rows if r["scheme"] == sc and r["kind"] == kind]
            if not g:
                continue
            zs = [r["z"] for r in g if r["z"] is not None]
            st = {"n": len(g), "mean_z": mean(zs), "median_z": statistics.median(zs) if zs else None,
                  "det": sum(r["detected"] for r in g) / len(g),
                  "ppl_mean": mean([ppl.get(key(r)) for r in g]),
                  "ppl_median": statistics.median([v for v in (ppl.get(key(r)) for r in g) if v is not None])
                  if any(ppl.get(key(r)) is not None for r in g) else None,
                  "tokens_scored_mean": mean([r.get("tokens_scored") for r in g]),
                  "overlap_f1_with_source": mean([overlap_f1(r["text"], r["source"]) for r in g])
                  if kind == "scrubbed" else None,
                  "empty_or_unscored": sum(1 for r in g if r["z"] is None)}
            summary["schemes"][sc][kind] = st
            f = lambda v, d=2: "n/a" if v is None else f"{v:.{d}f}"  # noqa: E731
            lines.append(f"| {label[kind]} | {st['n']} | {f(st['mean_z'])} | {st['det'] * 100:.0f}% | "
                         f"{f(st['median_z'])} | {f(st['ppl_median'])} (median) | "
                         f"{f(st['overlap_f1_with_source'])} | {f(st['tokens_scored_mean'], 0)} |")
        lines.append("")
    lines += ["## Draft's result (KGW, team implementation; for comparison)", "",
              "| Text | mean z | Det. |", "|---|---|---|"]
    lines += [f"| {k} | {v['mean_z']} | {v['det'] * 100:.0f}% |" for k, v in DRAFT_RESULT_KGW.items()]
    lines += ["", "Det.: share of texts with z > 4. PPL: perplexity of the text given its C4 prompt under "
              "the oracle model (lower is more fluent to the oracle). Overlap F1: word-level unigram F1 "
              "between the paraphrase and the watermarked text it came from (1 = same words). "
              "'human' is the real C4 continuation, scored as a reference.", "",
              f"## Settings not stated in the draft ({UNSTATED_NOTE})", ""]
    lines += [f"- {k}: {v}" for k, v in UNSTATED.items()]
    s.out.mkdir(parents=True, exist_ok=True)
    (s.out / "summary.md").write_text("\n".join(lines) + "\n")
    (s.out / "summary.json").write_text(json.dumps(summary, indent=2))
    run = {"pair": "watermarking/unigram (light paraphrase scrub)", "sok_commit": K.sok_commit(),
           "upstream": {"repo": K.WS_REPO, "commit": K.WS_COMMIT, "code": "src/watermarks/kgw"},
           "settings_from_draft": {"source": "team draft §5.5, §5.9", **DRAFT},
           "settings_not_in_draft": {"status": UNSTATED_NOTE, **UNSTATED,
                                     "values": {"prompt_tokens": s.prompt_tokens, "seed": s.seed,
                                                "batch_size": s.batch, "generation_kwargs": GEN_KW,
                                                "paraphrase_kwargs": dict(PARA_KW, max_new_tokens=s.para_max_new_tokens),
                                                "paraphrase_prompt": PARAPHRASE_PROMPT,
                                                "schemes": {k: SCHEMES[k] for k in s.schemes},
                                                "detector": {"z_threshold": DRAFT["z_threshold"],
                                                             "ignore_repeated_ngrams": False, "normalizers": []}}},
           "quality_measures": {"ppl": f"perplexity of the text given its C4 prompt under {s.oracle} "
                                       "(ours, not from the draft)",
                                "overlap_f1_with_source": "word-level unigram F1 between paraphrase and its "
                                                          "watermarked source (ours)"},
           "budget": {"texts": s.n_texts, "paraphrases_per_text": 1, "detector_queries": 0,
                      "watermarked_model_queries_by_attacker": 0},
           "models": {m: MODEL_REVISIONS.get(m) for m in (s.generator, s.paraphraser, s.oracle)},
           "manifest": json.loads(s.manifest.read_text()) if s.manifest.exists() else None,
           "settings": asdict(s), "stages": read_jsonl(s.out / "stages.jsonl"), "versions": K.versions()}
    (s.out / "run.json").write_text(json.dumps(run, indent=2, default=str))
    print("\n".join(lines[: lines.index("## Draft's result (KGW, team implementation; for comparison)")]))
    print(f"wrote {s.out / 'summary.md'}")


# ---- selftest (CPU, no models) -------------------------------------------------------------------
def stage_selftest(s):
    import torch
    K.import_upstream()
    vocab = list(range(1000))
    from src.watermarks.kgw import WatermarkLogitsProcessor
    for sc, scheme in SCHEMES.items():
        proc = WatermarkLogitsProcessor(vocab=vocab, gamma=DRAFT["gamma"], delta=DRAFT["delta"],
                                        seeding_scheme=scheme, device="cpu")
        ids = torch.randint(0, 1000, (2, 5))
        scores = torch.zeros(2, 1000)
        out = proc(ids, scores.clone())
        boosted = (out == DRAFT["delta"]).sum(dim=1).tolist()
        assert boosted == [int(1000 * DRAFT["gamma"])] * 2, (sc, boosted)
        if scheme == "gptwm":  # Unigram: the same green list whatever the context
            assert torch.equal(out[0], out[1]), "gptwm green list depends on context"
        print(f"{sc}: {boosted[0]} of 1000 tokens boosted by delta, ok")
    assert abs(overlap_f1("a b c d", "a b c d") - 1) < 1e-9 and overlap_f1("a b", "c d") == 0
    print("selftest passed")


# ---- entry point --------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES_ALL + ["all", "_selftest"])
    ap.add_argument("--smoke", action="store_true", help="tiny run with small stand-in models")
    ap.add_argument("--schemes", default=",".join(SCHEMES), help="comma-separated: unigram,kgw")
    a = ap.parse_args()
    s = Settings(smoke=a.smoke, schemes=[x for x in a.schemes.split(",") if x])
    for x in s.schemes:
        if x not in SCHEMES:
            raise SystemExit(f"unknown scheme {x}; choose from {list(SCHEMES)}")
    if a.smoke:
        for k in ("n_texts", "new_tokens", "prompt_tokens", "batch", "para_max_new_tokens"):
            setattr(s, k, SMOKE[k])
        s.generator, s.paraphraser, s.oracle = SMOKE["generator"], SMOKE["paraphraser"], SMOKE["oracle"]
    os.environ.setdefault("HF_HOME", str(WORK / "hf_home"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    STATE.mkdir(parents=True, exist_ok=True)
    (s.out / "logs").mkdir(parents=True, exist_ok=True)
    if a.stage == "all":
        for st in STAGES_ALL:
            print(f"\n===== {st} =====", flush=True)
            cmd = [sys.executable, str(Path(__file__).resolve()), st, "--schemes", a.schemes] + \
                (["--smoke"] if a.smoke else [])
            if subprocess.run(cmd).returncode != 0:
                raise SystemExit(f"stage {st} failed; fix and re-run, it resumes")
        return
    if a.stage == "_selftest":
        stage_selftest(s)
        return
    log = open(s.out / "logs" / f"{a.stage}.log", "a")
    sys.stdout, sys.stderr = K.Tee(sys.__stdout__, log), K.Tee(sys.__stderr__, log)
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] stage {a.stage}, work {WORK}, out {s.out}, "
          f"settings {asdict(s)}, python {platform.python_version()}", flush=True)
    {"prepare": stage_prepare, "run": stage_run, "quality": stage_quality,
     "summarise": stage_summarise}[a.stage](s)


if __name__ == "__main__":
    main()
