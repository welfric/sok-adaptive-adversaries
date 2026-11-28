# Watermark Stealing vs KGW2-SelfHash on one RTX A6000

`run.py` runs the full Watermark Stealing attack (Jovanović, Staab and Vechev, ICML 2024) against KGW2-SelfHash in the paper's Table 1 setting, using the authors' code at a pinned commit. One command runs everything and resumes after any interruption. The final numbers end up in `results/kgw_ws/summary.md`.

## What it runs

| | |
|---|---|
| Upstream code | [eth-sri/watermark-stealing](https://github.com/eth-sri/watermark-stealing) at `b8d207de`, config `configs/spoofing/llama7b/mistral_selfhash.yaml` |
| Defence (server) | `meta-llama/Llama-2-7b-chat-hf` with KGW2-SelfHash, γ = 0.25, δ = 4, detector ignores repeated n-grams |
| Attacker | `mistralai/Mistral-7B-Instruct-v0.1`, revision of 2024-02-29 (see "Pins") |
| Queries (axis iii, online) | n = 30,000 server responses (≤ 800 tokens) to C4 RealNewsLike prompts (paper §6, "Stealing parameters") |
| Base responses | Mistral's own responses to the same 30,000 prompts (setting B0, §3 and §5). Costs no server queries |
| Detector access | None (setting D0, §3) |
| Evaluation | Spoofing on Dolly creative writing, 100 prompts × 5 seeds (123 to 127) = 500 texts |
| Metric | FPR\*@1e-3: share of texts with detector p < 1e-3 **and** judge score ≥ 6.5 (§6). Also 1e-6, median p-value, PPL under Llama-2-13B-chat, judge score |
| Paper's number | Table 1, LMmo = LLAMA-7B, LMatt = MISTRAL-7B, Dolly CW: FPR\*@1e-3 **0.81**, median p 2e-19, PPL 5.11, GPT 8.26 |
| Controls | `control_no_stolen_rules`: the same attacker pipeline with the spoofer off (no stolen knowledge, so it should never pass). `control_server_watermarked` / `_unwatermarked`: the server's own answers (detector TPR and FPR) |

The authors' archive of server outputs (Google Drive link in their README) returns 404, so the script generates the 30,000 responses itself. This is the slow part.

## Requirements

- One RTX A6000 (48 GB). Another ≥ 40 GB GPU works too; the GPU name is recorded.
- **RAM: 64 GB minimum, 128 GB comfortable.** Upstream learning keeps Python dicts of n-gram counts. Measured on C4 with the Mistral tokenizer: about 0.45 GiB per 1,000 documents per count store, so about 15 GiB each for the watermarked and base stores at n = 30,000, plus pickling.
- **Disk: about 150 GB free** at `SOK_WORK`. That covers models (~70 GB), the counts cache (~10 to 20 GB, estimated), queries and caches.
- NVIDIA driver ≥ 530 (CUDA 12.1 wheels). For older drivers, see Troubleshooting.
- A Hugging Face account that has accepted Meta's Llama 2 licence for [`meta-llama/Llama-2-7b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) and [`meta-llama/Llama-2-13b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf), and a read token.
- Optional: an OpenAI API key, for the paper's GPT-4 judge. Without one, a local Qwen2.5-7B-Instruct judges instead (a deviation; see "Judge").
- `git`, `tmux`, and either [`uv`](https://docs.astral.sh/uv/) or `python3.10`.

## Step by step

All commands run from the repository root on the A6000 machine.

**1. Get the code and build the environment** (once, about 5 minutes):

```bash
bash launch/ssh/watermarking/kgw/setup.sh
```

This creates `.venv-kgw/` and prints the torch version and whether the GPU is visible. It must say `GPU visible: True`.

**2. Set the environment** (in every new shell, or in `~/.bashrc`):

```bash
export SOK_WORK=/path/with/150GB/sok_kgw
```

```bash
export HF_TOKEN=hf_your_read_token
```

```bash
export OAI_API_KEY=sk-your-key
```

`SOK_WORK` holds the upstream checkout, data, models (`$SOK_WORK/hf_home`, unless `HF_HOME` is already set) and query state. It defaults to `_work/kgw_ws` in the repository. `OAI_API_KEY` (the name upstream reads) is optional. On a multi-GPU machine, also `export CUDA_VISIBLE_DEVICES=<index of the A6000>`; the script defaults to GPU 0.

**3. Start tmux**, so the run survives a dropped SSH connection:

```bash
tmux new -s kgw
```

Detach with `Ctrl-b d`; reattach with `tmux attach -t kgw`.

**4. Download everything** (needs network, about 70 GB):

```bash
.venv-kgw/bin/python launch/ssh/watermarking/kgw/run.py prepare
```

This clones upstream at the pinned commit, downloads the pinned models and data, and validates every stage's config against upstream's schema. Every later stage runs offline. On a cluster whose compute nodes have no internet, run `prepare` on the login node.

**5. Smoke test** (about 30 to 60 minutes; separate folders, so it does not touch the real run):

```bash
.venv-kgw/bin/python launch/ssh/watermarking/kgw/run.py all --smoke
```

This runs every stage on 32 queries and 2 prompts, then prints a table and the **measured** time per watermarked response, with the projected time for 30,000. Check:

- it ends with `wrote .../results/kgw_ws/smoke/summary.md`;
- the logs show `skip_finished_selftest: ok`;
- `nvidia-smi` during `query-wm` shows well under 48 GB per worker (expect about 20 GB).

The smoke numbers themselves mean nothing: 32 queries cannot teach the attacker the watermark.

**6. Full run** (about 1 to 2 days; see "Time"):

```bash
.venv-kgw/bin/python launch/ssh/watermarking/kgw/run.py all
```

**7. Watch progress** from a second tmux window (`Ctrl-b c`):

```bash
tail -f results/kgw_ws/logs/query-wm.log
```

During querying the parent prints `done/30000`, the rate per hour and an ETA every 5 minutes. Worker output is in `results/kgw_ws/logs/query-wm-p0w0.log` and `-p0w1.log`.

**8. If anything stops** (crash, reboot, OOM, Ctrl-C), run the same command again. Every stage resumes:

- queries are kept per response, and malformed lines from a crash are dropped and redone;
- learned counts are cached;
- evaluation redoes only seeds that were not finished;
- PPL and judge skip texts already scored.

```bash
.venv-kgw/bin/python launch/ssh/watermarking/kgw/run.py all
```

You can also run one stage, e.g. `run.py spoof` or `run.py summarise`.

## Stages

| Stage | What it does | GPU memory | Rough time on an A6000 (not measured; the smoke test measures the long one) |
|---|---|---|---|
| `prepare` | Clone, downloads, config check | none | download-bound |
| `query-base` | 30,000 Mistral responses, no watermark, batch 64 | ~35 GB | 3 to 5 h |
| `query-wm` | 30,000 watermarked Llama-2 responses, 2 processes × batch 8 | ~20 GB each | 10 to 25 h |
| `spoof` | Upstream learning (CPU, RAM-heavy), then 5 seeds × 100 Dolly prompts | ~16 GB | learning 10 to 30 min; generation 1 to 3 h |
| `control-attacker` | Same pipeline with the spoofer off, 5 seeds | ~16 GB | 1 to 2 h |
| `control-server` | Server answers with and without watermark, seed 123 | ~16 GB | ~0.5 h |
| `ppl` | PPL of every text under Llama-2-13b-chat | ~28 GB | ~0.5 h |
| `judge` | Upstream GPT-4 judge prompt on every text | 0 (OpenAI) or ~17 GB (local) | ~0.5 h |
| `summarise` | `summary.md`, `summary.json`, `run.json` | none | seconds |

`query-wm` is slow because KGW2-SelfHash generation (upstream's `WatermarkLogitsProcessor` with self-salting) seeds a fresh vocabulary permutation for each of about 41 candidate tokens, per row and per generated token, in Python. It runs mostly on the CPU, which is why two processes share the GPU.

## Outputs (`results/kgw_ws/`, git-ignored)

| File | Contents |
|---|---|
| `summary.md` | The results table next to the paper's Table 1 row, plus the deviations |
| `summary.json` | The same, plus per-seed numbers and counts of unparsed judge answers |
| `records.jsonl` | One row per generated text: label, seed, prompt, completion, z-score, p-value, tokens scored, green fraction, the attacker's own z estimate |
| `ppl.jsonl`, `judge.jsonl` | Per-text PPL and judge output (full judge JSON) |
| `run.json` | Our commit, upstream commit, model and data revisions with sha256, settings, deviations, timing per stage, package versions |
| `stages.jsonl` | One record per finished stage, including the resolved upstream config |
| `logs/` | Every stage and worker log |
| `upstream/` | Upstream's own per-run result files |

The query corpus and the counts cache stay in `$SOK_WORK/state/`. Keep them until the results are written up: the ablations reuse the same 30,000 responses.

## Reading the result

- `spoof`, column `FPR*@1e-3`, is the number to compare with the paper's 0.81.
- `control_no_stolen_rules` should be about 0: the same model, prompts and decoding, without the stolen rules, does not pass the detector.
- `control_server_watermarked` should be about 1, and `control_server_unwatermarked` about 0. Together they show that the detector works on the server's own text at this threshold.
- A large gap between `det.@1e-3` and `FPR*@1e-3` means many spoofed texts are low quality.

Copy `summary.md` into `reports/` (as for the other pairs) and bring the whole folder back:

```bash
rsync -av a6000:path/to/sok-adaptive-adversaries/results/kgw_ws/ results/kgw_ws/
```

## Judge

The paper judges quality with GPT-4 (§6, App. D.1). With `OAI_API_KEY` set, the `judge` stage uses upstream's own `get_gpt4_grades`, which calls `gpt-4` at temperature 0.1. If OpenAI no longer serves `gpt-4`, pass another model, e.g. `--judge openai --judge-model gpt-4o`; record it as a deviation.

Without a key, `--judge auto` (the default) uses Qwen2.5-7B-Instruct with the same upstream system prompt and parser, greedy decoding. Its answers have code fences stripped and integer scores quoted, because upstream's parser rejects both. Its scores are not the paper's GPT-4 scores. The judge does not affect `det.@f`, only `FPR*@f`.

## Pins

| Item | Revision |
|---|---|
| eth-sri/watermark-stealing | `b8d207deb2b4ba758638a41d3a77891993acd351` |
| meta-llama/Llama-2-7b-chat-hf | `f5db02db724555f92da89c216ac04704f23d4590` |
| mistralai/Mistral-7B-Instruct-v0.1 | `73068f3702d050a2fd5aa2ca1e612e5036429398`. Same files as Dec 2023 apart from the README. Later commits change the chat template and `tokenizer.json`, and the current `tokenizer.json` does not load with tokenizers 0.15. The weights are identical in every revision |
| meta-llama/Llama-2-13b-chat-hf | `a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8` |
| Qwen/Qwen2.5-7B-Instruct (local judge only) | `a09a35458c702b33eeacc393d103063234e8bc28` |
| allenai/c4, `realnewslike/c4-train.0000{0,1}-of-00512.json.gz` | `1588ec454efa1a09f29cd18ddd04fe05fc8653a2` |
| databricks/databricks-dolly-15k | `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` |

`prepare` points each model's `refs/main` in the Hugging Face cache at the pinned revision. That way upstream's `from_pretrained(name)` loads the pin when later stages run offline. If you share an HF cache with other projects, this moves their `main` for these repositories too.

## Deviations from the paper's setup

The full list, with reasons, is `DEVIATIONS` in `run.py`; it is also written to `run.json` and `summary.md`. In short:

- server and base responses are regenerated here;
- SDPA attention instead of FlashAttention-2;
- watermarked queries use batch 8 × 2 processes instead of batch 64;
- rows that have finished skip the watermark processor; this does not change the output, and it is checked at start-up;
- C4 and Dolly come from pinned files;
- neptune and gradio are stubbed;
- PPL is recomputed per text with Llama-2-13B, avoiding an upstream batching bug in the in-run PPL;
- explicit seeding per evaluation seed;
- Dolly CW only;
- threshold p < f from the z-score;
- the library's default watermark key.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `401` / `403` when downloading Llama | Accept the Llama 2 licence on both model pages and `export HF_TOKEN` |
| `GPU visible: False` after setup, driver < 530 | `TORCH_INDEX=https://download.pytorch.org/whl/cu118 bash launch/ssh/watermarking/kgw/setup.sh` |
| CUDA OOM in `query-wm` | `--wm-workers 1 --wm-batch 16`, or keep 2 workers with `--wm-batch 4`. Re-run; it resumes |
| CUDA OOM in `query-base` | `--base-batch 32` |
| Killed (host RAM) in `spoof` while learning | Use a machine with more RAM. Otherwise `--n-queries 10000` (the paper's Fig. 3 says results converge around 10,000 queries, §6.1), but report it as a different budget |
| Judge: `model not found` for `gpt-4` | `--judge openai --judge-model <available model>` |
| Slurm instead of a plain SSH box | Put the step 6 command in an sbatch script with `--gres=gpu:1 --mem=128G`; resubmit the same job if it hits the time limit, since it resumes |

The settings above are flags of `run.py` (`--help` lists them). Pass the same flags on every re-run of a stage: `all` passes them on to each stage.
