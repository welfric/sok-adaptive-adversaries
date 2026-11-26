# Launchers

## Kaggle

One folder per defence, mirroring `defenses/`: `kaggle/<field>/<defence>/` holds `kernel-metadata.json` and a self-contained `run.py` (Kaggle script kernel, Internet on).

**GPU: T4 only.** Both `machine_shape` in the metadata and `push.sh` (`--accelerator NvidiaTeslaT4`, not overridable) request Kaggle's T4 option, which is two T4s; `run.py` uses only GPU 0. On Kaggle, `run.py` exits before running anything if the GPU is not a T4 (e.g. a P100), and records the GPU name in `run.json`. If you start a kernel from the web UI instead, pick "GPU T4 x2" under Settings > Accelerator. Each `run.py` clones the upstream repo at a pinned commit, installs what the Kaggle image lacks, runs the authors' commands unchanged and writes everything to `/kaggle/working/results/<pair>/`. The clone and the data stay in `/tmp`, outside the kernel output.

| Kernel | Runs | Output |
|---|---|---|
| `kaggle/adversarial_robustness/blacklight` | OARS, Square, untargeted, `adaptive` config, images 0 to 99 of the CIFAR-10 test set, then `analysis.py` | `analysis.txt` (attack success, mean queries, cache hits), `logs/`, `config.json` |
| `kaggle/poisoning/flame` | zhmzm/FLAME `main_fed.py` with the same two bug fixes, BadNet on Fashion-MNIST, 200 rounds, once with `--defence flame` and once with `--defence avg` (FedAvg baseline) | `run.json` (final main and backdoor accuracy per defence), `curves.json`, `<defence>_stdout.txt`, `<defence>_save/` |
| `kaggle/poisoning/flame_cifar` | zhmzm/FLAME `main_fed.py`, with two upstream bug fixes applied (FLAME clipping index; in-place image rescaling; see `defenses/poisoning/flame/README.md`), in FLAME's paper setting: CIFAR-10, ResNet-18, IID, 100 clients, 20% malicious, noise factor 0.001, 200 rounds. DBA and BadNet, each with `--defence flame` and `--defence avg` (4 runs) | `run.json` (per run: final/max/last-10% backdoor accuracy, main accuracy, FLAME admission rates, the paper's Table 3 numbers, the source of every setting), `curves.json`, `<attack>_<defence>_stdout.txt`, `<attack>_<defence>_save/` |

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

`ssh/run.sh`, not written yet. See PLAN.md section 5.
