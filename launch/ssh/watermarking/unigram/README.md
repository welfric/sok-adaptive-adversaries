# Watermark Stealing vs Unigram on one RTX A6000

`run.py` runs the full Watermark Stealing attack (Jovanović, Staab and Vechev, ICML 2024) against Unigram-Watermark (Zhao et al., ICLR 2024), using the authors' code at a pinned commit. It is the KGW2-SelfHash launcher (`../kgw/`) with only the defence changed: upstream's `gptwm` scheme and a `--setting` flag for the watermark parameters. One command runs a setting and resumes after any interruption.

## The two settings

| `--setting` | γ | δ | Source | What it shows |
|---|---|---|---|---|
| `table2` (default) | 0.25 | 4.0 | upstream `mistral_gptwm.yaml`, the attack paper's Table 2 | Reproduction: compare with the paper's FPR\*@1e-3 of **0.79** (median p 7e-20, PPL 3.23, GPT 8.38) |
| `zhao` | 0.5 | 2.0 | Zhao et al. §4.2 and App. B.4.1 | The defence's own setting, i.e. the one its claim in App. B.4 was made for. Not in the attack paper |

Only γ and δ differ. Everything else is upstream's config, including the attacker's spoofing hyperparameters, which the attack authors set for γ = 0.25, δ = 4. The attack is fixed, so they are not re-tuned for `zhao`. Run `table2` first: if it does not come close to 0.79, the `zhao` result cannot be interpreted.

## What it runs

| | |
|---|---|
| Upstream code | [eth-sri/watermark-stealing](https://github.com/eth-sri/watermark-stealing) at `b8d207de`, config `configs/spoofing/llama7b/mistral_gptwm.yaml` |
| Defence (server) | `meta-llama/Llama-2-7b-chat-hf` with Unigram (`gptwm`: one fixed green list, context width 0, upstream key 0), γ and δ from `--setting`, z-test detector without n-gram deduplication |
| Attacker | `mistralai/Mistral-7B-Instruct-v0.1`, same pin as the KGW run |
| Queries (axis iii, online) | n = 30,000 server responses (≤ 800 tokens) to C4 RealNewsLike prompts (attack paper §6) |
| Base responses | Mistral's own responses to the same prompts (B0). They do not depend on the watermark, so both settings share one corpus, and the KGW2-SelfHash run's corpus can be reused (step 4) |
| Detector access | None (D0) |
| Evaluation | Spoofing on Dolly creative writing, 100 prompts × 5 seeds (123 to 127) = 500 texts |
| Metric | FPR\*@1e-3: share of texts with detector p < 1e-3 **and** judge score ≥ 6.5 (§6). Also 1e-6, median p-value, PPL under Llama-2-13B-chat, judge score |
| Controls | As in the KGW run: `control_no_stolen_rules` (spoofer off) and `control_server_watermarked` / `_unwatermarked` (detector TPR and FPR) |

## Requirements

- One RTX A6000 (48 GB) or another GPU with at least 40 GB.
- RAM: 32 GB should be enough (estimated, not measured). With context width 0, learning only keeps per-token counts, not the context-conditioned counts of SelfHash.
- Disk: models and data are shared with the KGW run when `SOK_WORK` is the same (no new downloads). Each setting adds its query corpus and cache, a few GB.
- The same Hugging Face access (Llama 2 licence, `HF_TOKEN`) and optional `OAI_API_KEY` as the KGW run.

## Step by step

All commands run from the repository root on the A6000 machine. If the KGW2-SelfHash run was done there, steps 1 and 3 are already done: keep the same `.venv-kgw` and `SOK_WORK`.

**1. Environment** (once; the same one as the KGW run):

```bash
bash launch/ssh/watermarking/kgw/setup.sh
```

**2. Set the environment** (the same `SOK_WORK` as the KGW run, so the clone, models and data are reused):

```bash
export SOK_WORK=/path/with/150GB/sok_kgw
```

```bash
export HF_TOKEN=hf_your_read_token
```

```bash
export OAI_API_KEY=sk-your-key
```

Then `tmux new -s unigram`.

**3. Prepare** (fast if the KGW run's downloads are in the same `SOK_WORK`; it also validates the Unigram configs):

```bash
.venv-kgw/bin/python launch/ssh/watermarking/unigram/run.py prepare
```

**4. Reuse the base corpus, if the KGW run finished `query-base` in this `SOK_WORK`.** This saves the 3 to 5 h `query-base` stage. The corpus is only read, never modified. Check that it exists:

```bash
ls $SOK_WORK/state/base_query/ours/c4-kgw-selfhash
```

If it does, add `--base-corpus $SOK_WORK/state/base_query/ours/c4-kgw-selfhash` to every command below. If not, leave the flag out. The first setting then generates the corpus in `$SOK_WORK/state_unigram_base/`, and the second setting reuses it automatically.

**5. Smoke test** (separate folders; measures the time per watermarked response):

```bash
.venv-kgw/bin/python launch/ssh/watermarking/unigram/run.py all --smoke --setting table2
```

It should end with `wrote .../results/unigram_ws/table2/smoke/summary.md`, and the logs should show `skip_finished_selftest: ok`. With `--smoke`, leave out `--base-corpus`: the smoke run needs its own 32-response corpus.

**6. Full run, reproduction setting:**

```bash
.venv-kgw/bin/python launch/ssh/watermarking/unigram/run.py all --setting table2 --base-corpus $SOK_WORK/state/base_query/ours/c4-kgw-selfhash
```

**7. Full run, the defence's own setting** (after step 6, or on a second GPU with its own `CUDA_VISIBLE_DEVICES`):

```bash
.venv-kgw/bin/python launch/ssh/watermarking/unigram/run.py all --setting zhao --base-corpus $SOK_WORK/state/base_query/ours/c4-kgw-selfhash
```

**8. Watch progress** from a second tmux window:

```bash
tail -f results/unigram_ws/table2/logs/query-wm.log
```

**9. If anything stops**, run the same command again, with the same `--setting` and `--base-corpus`. Every stage resumes, as in the KGW run.

## Stages and time

The stages are the same as in the KGW run (see `../kgw/README.md`, "Stages"). None of these times have been measured; the smoke test measures `query-wm`.

| Stage | Expected on an A6000 |
|---|---|
| `query-base` | 0 with `--base-corpus`; otherwise 3 to 5 h, once for both settings |
| `query-wm` | Much shorter than SelfHash's 10 to 25 h: with context width 0 and no self-salting, the processor draws one permutation per row per token instead of about 41. Probably a few hours; GPU-bound, not CPU-bound |
| `spoof` | Learning in minutes (per-token counts); generation 1 to 3 h |
| `control-attacker`, `control-server`, `ppl`, `judge` | About 2 to 3 h together |

So roughly half a day to a day per setting with `--base-corpus`.

## Outputs (`results/unigram_ws/<setting>/`, git-ignored)

The same files as the KGW run: `summary.md` (with the paper's Table 2 row for `table2`; there is no paper row for `zhao`), `summary.json`, `records.jsonl`, `ppl.jsonl`, `judge.jsonl`, `run.json` (which records the setting, γ, δ and their source), `stages.jsonl`, `logs/` and `upstream/`. The query corpus and counts cache stay in `$SOK_WORK/state_unigram_<setting>/`.

```bash
rsync -av a6000:path/to/sok-adaptive-adversaries/results/unigram_ws/ results/unigram_ws/
```

## Reading the result

- `table2`, `spoof`, column `FPR*@1e-3`: compare with the paper's 0.79.
- `zhao`, `spoof`, column `FPR*@1e-3`: the result for the defence's own parameters. Compare it with the defence's claim in App. B.4 (see `defenses/watermarking/unigram/README.md`).
- The controls should read about 0 (`control_no_stolen_rules`, `control_server_unwatermarked`) and about 1 (`control_server_watermarked`). If `control_server_watermarked` is well below 1 for `zhao`, the weaker watermark is not detected at p < 1e-3 on 800-token texts. That limits any spoofing rate, so report it next to the spoofing result.

## Deviations

As in the KGW run (`DEVIATIONS` in `run.py`, also written to `run.json` and `summary.md`), plus:

- upstream's `gptwm` key is fixed at 0, while Zhao et al. use a secret key per model (§4.2). The attacker never reads the key;
- for `zhao`, γ and δ follow Zhao et al. but everything else is the attack's config, including the server model (Llama-2-7B-chat, not Zhao et al.'s models) and the detector. We have not compared that detector with Zhao et al.'s.

## Troubleshooting

As in `../kgw/README.md`. In addition:

| Symptom | Fix |
|---|---|
| `--base-corpus ...: N of 30000 missing` | That corpus is incomplete. Finish the KGW run's `query-base`, or drop the flag to generate a new one |
| Results from one setting appear in the other | They cannot: each setting has its own state (`state_unigram_<setting>`) and results folder. Check that every command used the same `--setting` |
