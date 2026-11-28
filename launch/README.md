# Launchers

## Kaggle

One folder per defence, mirroring `defenses/`: `kaggle/<field>/<defence>/` holds `kernel-metadata.json` and a self-contained `run.py` (Kaggle script kernel, Internet on).

**GPU: T4 only.** Both `machine_shape` in the metadata and `push.sh` (`--accelerator NvidiaTeslaT4`, not overridable) request Kaggle's T4 option, which is two T4s; `run.py` uses only GPU 0. On Kaggle, `run.py` exits before running anything if the GPU is not a T4 (e.g. a P100), and records the GPU name in `run.json`. If you start a kernel from the web UI instead, pick "GPU T4 x2" under Settings > Accelerator. Each `run.py` clones the upstream repo at a pinned commit, installs what the Kaggle image lacks, runs the authors' commands unchanged and writes everything to `/kaggle/working/results/<pair>/`. The clone and the data stay in `/tmp`, outside the kernel output.

| Kernel | Runs | Output |
|---|---|---|
| `kaggle/adversarial_robustness/blacklight` | OARS, Square, untargeted, `adaptive` config, images 0 to 99 of the CIFAR-10 test set, then `analysis.py` | `analysis.txt` (attack success, mean queries, cache hits), `logs/`, `config.json` |
| `kaggle/poisoning/flame` | zhmzm/FLAME `main_fed.py` with the same two bug fixes, BadNet on Fashion-MNIST, 200 rounds, once with `--defence flame` and once with `--defence avg` (FedAvg baseline) | `run.json` (final main and backdoor accuracy per defence), `curves.json`, `<defence>_stdout.txt`, `<defence>_save/` |
| `kaggle/poisoning/flame_cifar` | zhmzm/FLAME `main_fed.py`, with two upstream bug fixes applied (FLAME clipping index; in-place image rescaling; see `defenses/poisoning/flame/README.md`), in FLAME's paper setting: CIFAR-10, ResNet-18, IID, 100 clients, 20% malicious, noise factor 0.001, 200 rounds. DBA and BadNet, each with `--defence flame` and `--defence avg` (4 runs) | `run.json` (per run: final/max/last-10% backdoor accuracy, main accuracy, FLAME admission rates, the paper's Table 3 numbers, the source of every setting), `curves.json`, `<attack>_<defence>_stdout.txt`, `<attack>_<defence>_save/` |
| `kaggle/poisoning/flame_dba` | FLAME's Table 3 DBA row in DBA's own code ([AI-secure/DBA](https://github.com/AI-secure/DBA), pinned), its CIFAR-10 config and pretrained round-200 model, rounds 201 to 300, per `defenses/poisoning/ATTACK_PROTOCOL.md`. FLAME implemented as specified in Alg. 1 (cosine on updates, precomputed HDBSCAN). IID and Dirichlet 0.5, each with `flame` and `mean` (FedAvg) (4 runs) | `run.json` (per run: BA/MA at rounds 209, 210, 220, 250, 300, max and mean BA, FLAME admission of each attacker, checkpoint sha256, full config), `curves.json`, `<split>_<aggregation>_stdout.txt`, `<split>_<aggregation>_csv/` |

Every run also writes `run.json` (our commit, upstream commit, command, seed, torch/CUDA/GPU), `pip_freeze.txt` and `nvidia_smi.txt`.

### One-time setup

1. Install the CLI in the repo's `.venv` (git-ignored). Kaggle CLI 2.x is needed for `--accelerator` and requires Python 3.11+; `uv` fetches a suitable Python. Version 2.2.5 imports `six` without declaring it, hence the explicit install.

   ```bash
   uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python kaggle six
   ```

2. Sign in once (opens a browser, caches the credentials locally; no token file to manage):

   ```bash
   .venv/bin/kaggle auth login
   ```

   Alternatively, generate a token at kaggle.com/settings/api and save it to `~/.kaggle/access_token`. Never commit it.
3. Verify your phone number on Kaggle, or GPU and Internet stay disabled for kernels.

`push.sh` uses `.venv/bin/kaggle` when it exists. It takes your username from `KAGGLE_USERNAME`, a legacy `~/.kaggle/kaggle.json`, or `kaggle config view`, in that order.

### Push, check, fetch

```bash
launch/kaggle/push.sh launch/kaggle/adversarial_robustness/blacklight
```

```bash
launch/kaggle/push.sh launch/kaggle/poisoning/flame
```

```bash
launch/kaggle/push.sh launch/kaggle/poisoning/flame_cifar
```

```bash
launch/kaggle/push.sh launch/kaggle/poisoning/flame_dba
```

`push.sh` fills in your username and this repo's commit (with `-dirty` if the tree has uncommitted changes, so commit first) in a temp copy, then runs `kaggle kernels push --accelerator NvidiaTeslaT4`. It prints the status and output commands for the kernel, e.g.:

```bash
kaggle kernels status <username>/sok-blacklight-oars
```

```bash
kaggle kernels output <username>/sok-blacklight-oars -p results/kaggle/sok-blacklight-oars -o
```

`results/` is git-ignored.

### Changing a run

The settings are constants at the top of each `run.py` (`CONFIG`, `START_IDX`, `NUM_IMAGES`; `DEFENCES`, `ARGS`, `SEED`). Edit, commit, push again: Kaggle stores it as a new version of the same kernel.

### Local smoke tests

Both scripts run outside Kaggle and put their files in `SOK_WORK` and `SOK_OUT`.

```bash
SOK_WORK=/tmp/sok SOK_OUT=/tmp/sok-out SOK_EPOCHS=1 python launch/kaggle/poisoning/flame/run.py
```

```bash
SOK_WORK=/tmp/sok SOK_OUT=/tmp/sok-out-cifar SOK_EPOCHS=1 SOK_RUNS=dba:flame,badnet:avg python launch/kaggle/poisoning/flame_cifar/run.py
```

`SOK_RUNS` picks a subset of the CIFAR-10 kernel's `attack:defence` runs; on Kaggle all four run.

```bash
SOK_WORK=/tmp/sok SOK_OUT=/tmp/sok-out python launch/kaggle/adversarial_robustness/blacklight/run.py --prepare-only
```

The Blacklight script stops after the data export and import check with `--prepare-only`, because OARS `main.py` needs CUDA.

## SSH

One folder per pair, mirroring `defenses/`: `ssh/<field>/<defence>/` holds a `setup.sh` (virtual environment with pinned packages), a resumable `run.py` and a README with the run instructions.

| Folder | Runs | GPU | Output |
|---|---|---|---|
| [`ssh/watermarking/kgw`](ssh/watermarking/kgw/README.md) | Watermark Stealing vs KGW2-SelfHash, Table 1 setting: 30,000 queries to Llama-2-7B-chat, Mistral-7B attacker, spoofing on Dolly CW × 5 seeds, plus controls, PPL and judge | one RTX A6000 (48 GB), 64 GB+ RAM | `results/kgw_ws/summary.md`, `summary.json`, `records.jsonl`, `run.json` |
| [`ssh/watermarking/unigram`](ssh/watermarking/unigram/README.md) | Watermark Stealing vs Unigram, same pipeline and pins; `--setting table2` (γ 0.25, δ 4, the attack paper's Table 2) or `zhao` (γ 0.5, δ 2, the defence paper). Can reuse the KGW run's base corpus | one RTX A6000 (48 GB) | `results/unigram_ws/<setting>/summary.md`, etc. |
| [`ssh/watermarking/unigram_scrub`](ssh/watermarking/unigram_scrub/README.md) | The team draft's light paraphrase scrub (§5.5, §5.9) on Unigram, and KGW side by side: γ 0.25, δ 4, z > 4, Mistral-7B, 50 texts × 200 tokens; settings the draft leaves out are config values flagged in `run.json` | any GPU with 24 GB | `results/unigram_scrub/summary.md`, `run.json` |
