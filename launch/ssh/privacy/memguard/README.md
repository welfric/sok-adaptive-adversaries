# Generic MemGuard for a saved LiRA run (CPU; any Linux machine with many cores)

`run.py` adds MemGuard (Jia et al., CCS 2019) to an existing LiRA run (Carlini et al., S&P 2022). You point it at the saved logits of the target model and its shadow models, using a small JSON config. It then:

1. applies MemGuard to every output of the target and of every shadow model (the attacker knows the defence);
2. writes the defended logits in the same layout;
3. scores undefended vs MemGuard with HAMP's LiRA code. Your own LiRA code can read the defended logits instead.

The protocol, axis coding and quotes are in [`defenses/privacy/memguard/PROTOCOL.md`](../../../../defenses/privacy/memguard/PROTOCOL.md).

## What it needs

| Input | Default layout (HAMP / Carlini LiRA) | Config key |
|---|---|---|
| Target model's logits | one folder with `0_logit.npy`, shape (N, 1, C), (N, C) or (N, A, C) | `target_dir`, `logit_pattern` |
| Shadow models' logits | one folder with `<tag>_logit.npy` for every shadow model | `shadow_dir` |
| Membership masks | `<tag>_keep.npy` (N,) next to each logits file | `keep_pattern` |
| True labels | `shadow_label.npy` (N,) in each folder | `labels_file` |
| MemGuard defence classifier | `.h5` in HAMP's architecture, e.g. HAMP's `final-all-models/<ds>_<size>_MIA_model.h5` | `defence_classifier` |

**Logits, not scores.** LiRA's `*_score.npy` files are not enough: MemGuard changes the confidence vector before scoring.

**No defence classifier?** If `defence_classifier` is not set, the `train-classifier` stage trains one from the target's logits and keep mask. It follows HAMP's `memguard-train-attack-classifier.py` recipe: 400 epochs, SGD at learning rate 0.001, batch 256. That takes minutes on a CPU.

**Inputs are read-only.** They are only read or linked to, never written; the smoke test checks the input files' hashes before and after.

All config keys and their defaults are in `DEFAULTS` at the top of `run.py`. The config is copied into `run.json`.

## Run it

All commands run from the repository root.

```bash
bash launch/ssh/privacy/memguard/setup.sh
```

Copy `config.example.json` and set your folders. Relative paths are resolved from the config file's folder.

```bash
cp launch/ssh/privacy/memguard/config.example.json my_config.json
```

Check that every input is there and consistent:

```bash
.venv-memguard/bin/python launch/ssh/privacy/memguard/run.py check --config my_config.json
```

Only if there is no defence classifier:

```bash
.venv-memguard/bin/python launch/ssh/privacy/memguard/run.py train-classifier --config my_config.json
```

Measure the speed on this machine (200 outputs, about a minute):

```bash
.venv-memguard/bin/python launch/ssh/privacy/memguard/run.py benchmark --config my_config.json
```

Then start the full run in tmux. It resumes after any interruption:

```bash
.venv-memguard/bin/python launch/ssh/privacy/memguard/run.py all --config my_config.json
```

**Compute.** MemGuard optimises each output one at a time on the CPU, and the GPU is not used.
- CIFAR-10 in HAMP's setup has 129 models × 45,000 outputs: an estimated 300 to 600 core-hours, so roughly 8 to 16 hours on 32 to 64 cores.
- `benchmark` prints the real projection for your machine.
- `workers` in the config sets the number of parallel processes; the default is all cores minus one.

**Smoke test** with synthetic data and no downloads:

```bash
.venv-memguard/bin/python launch/ssh/privacy/memguard/run.py _make-smoke
.venv-memguard/bin/python launch/ssh/privacy/memguard/run.py all --config _work/memguard/smoke/config.json
```

## Output (`out_dir`, default `results/memguard/<name>/`)

| Path | Contents |
|---|---|
| `summary.md` | LiRA TPR at `fpr` and AUC, undefended vs MemGuard (draft Table 5 format), member and non-member accuracy, the label check, all LiRA variants, the config, and the deviations |
| `state/memguard/{target,shadow}/` | The defended logits, in the input's file names and shapes. Point your own LiRA code here |
| `lira.json`, `lira_plot_output.txt` | Parsed and raw `lira-plot.py` output |
| `run.json` | Config, pinned commits, MemGuard hyperparameters and seed, defence classifier sha256 (and recipe, if trained here), input shapes, versions and timings |

## MemGuard code

MemGuard is not rewritten. Workers run the unchanged MemGuard block of HAMP's `cifar10/lira-inference-memguard.py` (pinned at `a064a1f8`). Its optimisation loop is identical to the original [jinyuan-jia/MemGuard](https://github.com/jinyuan-jia/MemGuard) code (`34e1859e`); only comments and one print line differ.

One line is set from the data: `user_label_dim` = the number of classes. HAMP's five dataset scripts differ only in that line, so the tool covers all of them. The parameters are c1 1, c2 10, c3 from 0.1 ×10 per round up to 1e5, 300 iterations and step 0.1.

## Deviations

These are listed in `run.py` and in `run.json`.
- **Saved logits.** MemGuard runs on saved logits rather than model checkpoints.
- **CIFAR difference.** HAMP's own CIFAR MemGuard script computes logits without `model.eval()`. Results on CIFAR can therefore differ slightly from HAMP's Table VIII (CIFAR-10 LiRA TPR at 0.1% FPR: 2.76% undefended, 1.52% MemGuard).

## Local check (done)

The smoke test passes on a Mac, covering:
- check, train-classifier, benchmark and the 20 shards with 4 workers;
- upstream scoring and plotting, the summary and resume;
- a second layout with other file names and 2-D logits, which gives identical results.

MemGuard kept 100% of labels, and the input file hashes were unchanged. The numbers are synthetic and mean nothing.
