# Light paraphrase scrub vs Unigram (and KGW) on one GPU

`run.py` rebuilds the watermark table in the team draft (§5.5, §5.9) with Unigram (Zhao et al., ICLR 2024) in place of KGW. The table has three rows: unwatermarked, watermarked and after scrub, each with mean z and detection rate. By default it runs KGW with the identical configuration too, so the two defences sit side by side under one attack. The protocol and axis note is in [`defenses/watermarking/unigram/SCRUB_PROTOCOL.md`](../../../../defenses/watermarking/unigram/SCRUB_PROTOCOL.md).

## Settings

**Kept from the draft** (as relayed from §5.5 and §5.9; not yet checked against the draft text):

- γ = 0.25, δ = 4.0, detection at z > 4;
- ungated Mistral-7B as the generator;
- 50 texts of 200 tokens;
- the attacker paraphrases each watermarked text once, with model access but not the key.

**Not stated in the draft.** These are config values at the top of `run.py` (`UNSTATED`, `PARAPHRASE_PROMPT`, `GEN_KW`, `PARA_KW`, `SEED`), and `run.json` marks each one "not stated in draft; to confirm with Rezwan":

| Setting | Our value |
|---|---|
| Generator variant | `mistralai/Mistral-7B-v0.1` (base), plain continuation of C4 prompts |
| Prompts | First 200 tokens of the first 50 C4 RealNewsLike documents (pinned shard 0) that have at least 400 tokens |
| Generation | Multinomial sampling, temperature 1.0, top-k 0, top-p 1.0, exactly 200 new tokens |
| Unwatermarked text | Same generator, prompts and seeds, watermark off |
| KGW hashing | `lefthash`: KGW ICML 2023, h = 1, additive PRF |
| Unigram | `gptwm`: one fixed green list, as in our Watermark Stealing vs Unigram run |
| Detector | z-test over every scored token, no n-gram deduplication, no normalisers |
| Paraphraser | `mistralai/Mistral-7B-Instruct-v0.1`, temperature 0.7, up to 400 tokens, a one-line paraphrase instruction (`PARAPHRASE_PROMPT`) |
| Seeds | Generation batch b uses seed b; paraphrasing batch b uses seed 10000 + b |
| Quality | Perplexity of each text given its C4 prompt under `Qwen/Qwen2.5-7B-Instruct`, plus word-level overlap F1 between each paraphrase and its source |

Change any of these in `run.py` once Rezwan confirms his values; every run writes them to `run.json`.

**Watermark code.** Both schemes come from the KGW library inside eth-sri/watermark-stealing (`src/watermarks/kgw`), at the commit pinned by `../kgw/run.py`. Unigram is therefore the same implementation as in `../unigram/`.

## Run it

From the repository root, with the environment from the KGW launcher:

```bash
bash launch/ssh/watermarking/kgw/setup.sh
```

Use the same `SOK_WORK` as the other watermarking runs, so the upstream clone and the Hugging Face cache are shared. No Hugging Face token is needed: all models are ungated.

```bash
export SOK_WORK=/path/with/100GB/free
```

```bash
.venv-kgw/bin/python launch/ssh/watermarking/unigram_scrub/run.py prepare
```

Smoke test, which uses small Qwen2.5-0.5B stand-ins and separate folders:

```bash
.venv-kgw/bin/python launch/ssh/watermarking/unigram_scrub/run.py all --smoke
```

Full run:

```bash
.venv-kgw/bin/python launch/ssh/watermarking/unigram_scrub/run.py all
```

Only one scheme: add `--schemes unigram`. Every stage resumes, so after an interruption run the same command again.

**Time and memory** (estimated, not measured on a GPU):
- Generation, paraphrasing and scoring take about 15 minutes on an A6000. Model loading dominates.
- `prepare` downloads up to about 45 GB: Mistral-7B base, Mistral-7B-Instruct and Qwen2.5-7B-Instruct. Models already in the shared cache are skipped.
- Only one 7B model is on the GPU at a time (fp16, about 15 GB), so any 24 GB GPU works.

## Output (`results/unigram_scrub/`, git-ignored)

| File | Contents |
|---|---|
| `summary.md` | For each scheme: unwatermarked / watermarked / after scrub / human, each with n, mean z, Det. (z > 4), median z, oracle PPL and overlap F1. Then the draft's KGW table, then the unstated settings |
| `summary.json` | The same numbers |
| `run.json` | The draft's settings, every unstated setting with its status, quality measures, budget, models and revisions, data sha256, versions and stage timings |
| `stages.jsonl`, `logs/` | Per-stage records and logs |

Per-text records (prompt, text, z, p-value, tokens scored, green fraction, seed) are in `$SOK_WORK/scrub/full/records.jsonl`, and perplexities in `ppl.jsonl` next to it.

## Local check (done)

- `_selftest` passes on CPU: both schemes boost exactly γ|V| tokens, and Unigram's green list does not depend on context.
- `all --smoke` runs end to end on a Mac CPU and resumes without regenerating anything.
- The smoke numbers come from 0.5B stand-in models, 3 texts and 24 tokens, so they mean nothing about the result.
