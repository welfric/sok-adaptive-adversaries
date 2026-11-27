# FLAME against non-adaptive backdoors: Fashion-MNIST and CIFAR-10

Ahmad, 2026-11-26. Results from Kaggle (T4), for review.

**Summary.** In the harness we use (zhmzm/FLAME, with two bug fixes), FLAME does not stop a plain BadNet or DBA-style backdoor, on Fashion-MNIST or on CIFAR-10. An audit of the CIFAR-10 run against the paper shows it is **not a fair test of FLAME's Table 3 claim**, because both the attack and FLAME's clustering step differ from the paper. Until a fair re-run is done, the results support only a narrow claim about this implementation and configuration, not a claim about FLAME.

Yang et al.'s attack (the pair's attack in our plan) is not wired in yet. Every run here is a baseline that uses none of the three axes, apart from the FL caveat under "What this means for the project".

## Setup

| Setting | Value |
|---|---|
| Defence | FLAME (Nguyen et al., USENIX Security 2022; read from arXiv:2101.02281 v5) |
| Harness | [zhmzm/FLAME](https://github.com/zhmzm/FLAME) at `fd4cef2`, unofficial, no licence file. The official FLAME code is not public |
| Patch | 5 changed lines (`HARNESS_PATCH`, sha256 `eb82ad5c...`), applied after a hard reset to the pinned commit and recorded in each `run.json`. See "Harness bugs we fixed" |
| Baseline | FedAvg (`--defence avg`), same attack, same seed |
| Clients | 100 in total, 10 per round, 2 of the 10 malicious in every round from round 1 (PMR 20%) |
| Rounds | 200 |
| Seed | 1 (Python `random`, NumPy, torch seeded by the launcher; upstream ignores `--seed`). Single seed only |
| Hardware | Kaggle, one Tesla T4 |

| | Fashion-MNIST kernel | CIFAR-10 kernel ("paper setting") |
|---|---|---|
| Kernel | `sok-flame-badnet` v3 | `sok-flame-cifar` v2 |
| Commit | `43e541a` | `2811ada` |
| Model | harness CNN | full ResNet-18 (about 11M parameters; the paper uses "ResNet-18 Light", about 2.7M, Table 2) |
| Data split | harness default | IID, 500 images per client |
| Attacks | BadNet | DBA (zhmzm re-implementation) and BadNet |
| Backdoor | class 7 to class 5, square trigger | class 7 to class 5; the DBA global trigger is the same 5x5 square as BadNet |
| Local training | 3 epochs, batch 50, lr 0.1 | 2 epochs, batch 64, lr 0.1 |
| Poisoning | `--poison_frac 0.5`: about half of the class-7 images in each batch, about 5 to 7% of a client's samples | same |
| FLAME noise | λ = 0.001 | λ = 0.001 (App. B.3: "λ = 0.001 for IC and NLP") |
| Wall time | 64 min (2 runs) | 4 h 56 min (4 runs) |

BA = backdoor accuracy (triggered class-7 test images classified as 5). MA = main-task accuracy. "Last 10%" is the mean over rounds 180 to 199.

## Results

### Fashion-MNIST, BadNet

| Defence | BA final | BA last 10% | First round with BA ≥ 90% | MA final | Malicious / benign admitted |
|---|---|---|---|---|---|
| FLAME | 100.0 | 100.0 | 6 | 89.2 | 72.3% / 56.9% |
| FedAvg | 100.0 | 100.0 | 6 | 90.9 | |

HDBSCAN returned exactly one cluster of exactly 6 updates (its minimum size) in all 200 rounds, rejecting 4 of 10 every time. The FLAME paper never evaluates BadNet on Fashion-MNIST, so this row has no paper number to compare against.

### CIFAR-10, DBA and BadNet

| Run | BA final | BA last 10% | BA mean, rounds 100 to 199 (min / max) | MA final | Malicious / benign admitted |
|---|---|---|---|---|---|
| DBA, FLAME | 52.7 | 54.4 | 49.6 (11.0 / 80.1) | 75.1 | 77.0% / 62.1% |
| DBA, FedAvg | 72.3 | 68.4 | 54.0 (10.3 / 77.7) | 80.4 | |
| BadNet, FLAME | 86.6 | 89.6 | 87.7 (72.9 / 93.9) | 75.1 | 79.3% / 59.6% |
| BadNet, FedAvg | 93.6 | 93.4 | 91.4 (81.9 / 95.2) | 80.6 | |

Paper numbers (Table 3, CIFAR-10, BA / MA in %): DBA with no defence 93.8 / 57.4, with FLAME 3.2 / 76.2. BadNet is not in the paper. zhmzm's README reports BadNet 70.2 / 80.38 with no defence and 3.33 / 78.77 with FLAME, but from a different configuration (its example uses `--malicious 0.1 --poison_frac 1.0 --attack_goal -1`).

All 4 runs completed 200 rounds with no NaN values or tracebacks.

## Observations

1. **FLAME barely reduces the backdoor.** The drop in BA (last 10%) is 0 points on Fashion-MNIST, about 14 for DBA and about 4 for BadNet on CIFAR-10. FLAME costs about 2 points of MA on Fashion-MNIST and about 5 on CIFAR-10.
2. **The filter does not separate attackers.** HDBSCAN returned one cluster in every round of all three FLAME runs, usually of the minimum size of 6. It admits malicious updates more often than benign ones. On CIFAR-10 DBA, malicious updates are 13% of the rejected ones, below the 20% base rate, so the filter does worse than chance.
3. **DBA is unstable.** Its BA swings between about 10% and 80% after round 100 under both defences, so a single seed and a single final round are not a reliable measurement.
4. **BA in the first few rounds is meaningless.** MA is about 10% there and the model predicts one class.

## Fidelity audit (CIFAR-10 run vs the paper)

A read-only audit compared the CIFAR-10 run with the paper, the patched harness, and DBA's released code ([AI-secure/DBA](https://github.com/AI-secure/DBA), `utils/cifar_params.yaml`; we have not read the DBA paper itself). I checked every paper quote below against the text extracted from the v5 PDF, and every code and config claim in the code itself.

**Attack (most likely cause of the gap).** FLAME §7.1 says it ran the attacks "using the same attack settings as in the original works".

| | DBA's released config | Our run (zhmzm DBA) |
|---|---|---|
| Adversaries | 4 fixed (`adversary_list: [17, 33, 77, 11]`) | 2 of 10 clients in every round, cycling through clients 0 to 19 |
| Schedule | one shot each, rounds 203, 205, 207, 209 | every round from round 1 |
| Boosting | update scaled by 100 (`scale_weights_poison: 100`, `baseline: false`) | none |
| Poison training | `internal_poison_epochs: 6` | 2 local epochs |
| Backdoor | all-to-one, label 2, four 1x6 bars | one-to-one 7 to 5, four quarters of a 5x5 square |
| Start | resumes a 200-round pretrained model | from scratch |

Clipping to the median norm largely cancels an update scaled by 100, so the original DBA is a much easier case for FLAME than our persistent, unscaled attack. The FLAME paper does not say whether its DBA was scaled, which round Table 3's BA is read at, or which PMR and IID setting the DBA row used (all unstated).

**Defence.** In `utils/defense.py`, `flame()`:

- Cosine distances are computed on the full local weights, BatchNorm running statistics included, not on the updates. §4.3.1 says "pairwise cosine distances to measure the angular differences between all model updates". Alg. 1 line 6 says "cij is the cosine distance between Wi and Wj" (local models), so the paper is itself inconsistent. §2.1 defines the model as "p parameters (indicating both weights and biases)", so including BN buffers is a deviation either way.
- The distance matrix goes to `hdbscan.HDBSCAN(...).fit(cos_list)` with the default Euclidean metric, not `metric="precomputed"`. HDBSCAN therefore treats each row of the distance matrix as a feature vector. App. E says "HDBSCAN first uses the given distances to build a minimal spanning tree". This likely explains observation 2.
- Everything else matches: min_cluster_size n/2+1 and min_samples 1 (App. E), the clipping bound as the median of all n update norms (§4.3.2), γ = S_t / e_b and clipping (Alg. 1 line 11, after our fix), aggregation, and σ = λ·S_t with λ = 0.001 (App. B.3).

**Other differences.**

- The model is not near convergence: MA 75 to 80% against a benign 92.2% in Table 4. The paper's own App. H (Table 8, 3DFed on CIFAR-10) shows FLAME's BA falling from 53.9 to 9.9 as the pretrained model converges further.
- Only 10 clients per round, so HDBSCAN clusters 10 points.
- Full ResNet-18 instead of Light.
- Training from scratch breaks our own "pretrained checkpoints only" rule.
- The data split is IID per §7.1, but §7.3 says "since the IC data are non-IID".

**Verdict.** The CIFAR-10 run is not a fair test of Table 3's DBA row. The defensible claim is: *zhmzm's FLAME implementation, with our two fixes, does not stop a continuous, unscaled DBA-style or BadNet backdoor trained from scratch on CIFAR-10 (single seed).* Please do not cite either run as "FLAME does not reproduce".

## Harness bugs we fixed

Both were confirmed with a small probe before the fix. Both kernels apply the same patch.

- **Clipping used the wrong norm.** `gama = clip_value/norm_list[i]` takes the norm of the i-th client overall, not of the i-th admitted client, so it does not implement Alg. 1 line 11. In the probe, benign updates below the median were shrunk to 12% and updates above it were not clipped. Fix: `norm_list[benign_client[i]]`. An earlier, unpatched Fashion-MNIST run gave the same headline result (BA 100% under both defences), so this bug was not the cause.
- **`save_img()` rescaled images in place** while writing a preview PNG. On CIFAR-10 with a GPU, poisoned training images were rescaled to [0, 1] but triggered test images were not, so the backdoor was trained and tested on different inputs. Fix: write the PNG from a copy.

## What this means for the project

- **We have no working FLAME baseline yet.** For the axis ablations to mean anything, the defence must stop the attack that uses none of the axes. Here that attack already succeeds, so any comparison of Yang et al. configurations against this harness would be uninterpretable.
- **A definitional question for the axes.** In FL every client, attackers included, trains on the current global model each round. Does that count as (iii) online feedback, or is it part of the setting? DBA and BadNet use no knowledge of FLAME (no (i)) and are not designed against it (no (ii)). We should decide how to annotate (iii) for FL poisoning before writing up this pair.
- **A possible SoK example.** Whether FLAME's claim holds depends on attack details the paper leaves unstated (scaling, schedule, starting model). That fits our narrative, but only once a fair run shows which difference drives the result.

## Next steps (not done)

1. Fix the clustering: cosine on the weight and bias updates, float64 matrix, `metric="precomputed"`. Keep the Alg. 1 literal version as an ablation, and log S_t, per-client norms, and the true positive and true negative rates.
2. Run Table 3's DBA row with DBA's own code, config and pretrained 200-round checkpoint (on Google Drive, not yet downloaded).
3. Keep the current continuous attack as a separately named configuration, and re-run it from a converged checkpoint.
4. Use at least 3 seeds and report BA as a windowed mean alongside the final value.
5. Wire in Yang et al. once a baseline holds.

Estimate: about 5 hours of T4 time per seed for 4 runs, plus a few hours of code work for step 2. Possibly worth raising with the mentors before building it.

## Reproduce

Launchers on `main` (commits `2811ada`, `43e541a`):

```bash
launch/kaggle/push.sh launch/kaggle/poisoning/flame
```

```bash
launch/kaggle/push.sh launch/kaggle/poisoning/flame_cifar
```

Each run writes `run.json` (commits, exact commands, the patch with its sha256, seed, torch, CUDA and GPU), `curves.json` (BA and MA per round), and the full stdout per run. Only cite runs whose `run.json` contains `upstream_patch`. More detail on the paper's setting and the harness is in `defenses/poisoning/flame/README.md`.
