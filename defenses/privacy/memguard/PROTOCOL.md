# Protocol: LiRA vs MemGuard

| | |
|---|---|
| Defence | MemGuard (Jia, Salem, Backes, Zhang and Gong, CCS 2019; arXiv:1909.10594). Test-time confidence masking: adversarial noise on the confidence vector against a defence classifier, with the predicted label kept |
| Attack (fixed for the privacy field) | LiRA (Carlini et al., S&P 2022; arXiv:2112.03570), using the HAMP repository's code and pre-computed shadow results, as in the team's HAMP and RelaxLoss runs |
| Code | [DependableSystemsLab/MIA_defense_HAMP](https://github.com/DependableSystemsLab/MIA_defense_HAMP) at `a064a1f8`: MemGuard block of `lira-inference-memguard*.py`, `lira-score.py`, `lira-plot.py`. The MemGuard loop is identical to [jinyuan-jia/MemGuard](https://github.com/jinyuan-jia/MemGuard) `defense_framework.py` at `34e1859e`; only comments and one print line differ (checked) |
| Run | [`launch/ssh/privacy/memguard/`](../../../launch/ssh/privacy/memguard/README.md): a generic tool pointed at a saved LiRA run (target and shadow logits) through a JSON config |
| Status | Written and smoke-tested on synthetic data. Not yet run on real data |

**Quotes.** Every quote below was checked against the arXiv HTML version of the paper named. The team draft is not in this repository; points attributed to it were relayed by the team-chat session and still need checking against the draft text.

## Why MemGuard

**Its claim.** MemGuard claims robustness against an attacker who knows the defence:
- §3.2: "to consider strong attacks, we assume the attacker knows our defense mechanism, but the defender does not know the attack classifier".
- §5.1.3 evaluates two "Adaptive attacks", described as "customized to our defense". One is NN-AT: the attacker runs MemGuard's Phase I on a shadow classifier's confidence vectors and adversarially trains its attack classifier on the noisy and clean versions.

**The literature reverses that conclusion:**
- Choquette-Choo et al. (ICML 2021, arXiv:2007.14321) §5.3, "Breaking MemGuard": "a label-only attack like ours is the "right" adaptive attack against these defenses".
- Aerni et al. (CCS 2024, arXiv:2404.17399) §5.1 omit MemGuard (Jia et al., 2019) as one of the defences "that have been circumvented in prior work".

**What the run tests.** Whether a score-based attack (LiRA) that knows the defence still finds MemGuard strong, at the draft's metric (TPR at 0.1% FPR).

## Axes

### LiRA as run here against MemGuard

| Axis | Value | Evidence |
|---|---|---|
| (i) Knowledge | Full: the mechanism and the defender's instance, i.e. the same pre-trained defence classifier and parameters | HAMP §II-B: "we assume the adversary has full knowledge of the defense technique and can therefore train shadow models in the same way as the target model is trained". HAMP repo `lira-memguard.sh` passes the same `final-all-models/<ds>_<size>_MIA_model.h5` to the shadow and target runs |
| (ii) Defence-specific design | Instance replication only. The LiRA test statistic is generic; the only defence-specific step is applying MemGuard to the shadow models' outputs | HAMP §IV-A: "each shadow model is trained following the same procedure as the targeted defense (as per our threat model)". No MemGuard-specific change to LiRA itself (repo: `lira-plot.py` is the same for every defence) |
| (iii) Feedback | Offline, surrogate: 128 shadow models trained by the attacker in advance. The target is queried once per sample, with no query depending on an earlier answer. Signal: confidence scores | LiRA §IV (Algorithm 1): "We first train $N$ shadow models [60] on random samples from the data distribution". One query per sample: repo (`lira-inference*.py`; no augmentations) |

### MemGuard's own evaluated adaptive attack (for the corpus)

| Axis | Value | Evidence |
|---|---|---|
| (i) | Full (mechanism) | MemGuard §3.2, quoted above |
| (ii) | Instance: attack classifier adversarially trained against MemGuard noise | MemGuard §5.1.3, NN-AT: "the attacker uses the Phase I of our defense to find the representative noise vector" |
| (iii) | Offline, surrogate (shadow classifier); score signal | MemGuard §5.1.3: "The attacker uses $D_{2}^{\prime}$ to train a shadow classifier" |

## Settings: from the draft, the repo, or placeholders

| Setting | Value | Source |
|---|---|---|
| LiRA implementation and shadow results | HAMP repo's code and pre-computed shadow results | Draft §5.3, §5.9 (relayed) |
| Metrics | TPR at 0.1% FPR, AUC | Draft (relayed); repo `lira-plot.py` default `--fpr 0.001` |
| Number of shadow models | 128 (64 IN / 64 OUT) | Repo `lira-*.sh`; HAMP §IV-A "we train 128 shadow models for each defense" |
| Train size and architecture | CIFAR-10/100: 25,000, DenseNet; Purchase100 20,000, Texas100 15,000, Location30 1,500, FC networks | Repo; HAMP §IV-A |
| MemGuard hyperparameters | c1 1, c2 10, c3 0.1 → ×10 per round up to 1e5, 300 iterations, step 0.1 | Repo (= original MemGuard) |
| MemGuard defence classifier | Pre-trained `.h5` from the HAMP download if available; otherwise trained from the target's logits with HAMP's recipe (`memguard-train-attack-classifier.py`: 400 epochs, SGD 0.001, batch 256), the only training in the run | Repo |
| Dataset, file locations | Whatever Rezwan's config points at (`target_dir`, `shadow_dir`) | **From Rezwan's config; recorded in run.json** |
| LiRA variant reported | `table_variant` in the config, default `online`; all five variants are kept | **From Rezwan's config (default is a placeholder)** |
| MemGuard input | Saved undefended logits of the LiRA run | Ours. On CIFAR this differs from HAMP's own script, which skips `model.eval()` (see the launcher README) |
| MemGuard block | HAMP's `cifar10/lira-inference-memguard.py` block, with `user_label_dim` set to the number of classes. The five dataset scripts' blocks differ only in that line (checked) | Repo |

Reference numbers: HAMP Table VIII (App. A-K) reports these LiRA TPRs at 0.1% FPR:

| Dataset | Undefended | MemGuard |
|---|---|---|
| CIFAR-10 | 2.76% | 1.52% |
| Purchase100 | 1.86% | 1.18% |
| Location30 | 16.22% | 4.00% |

## Optional: label-only ablation on axis (iii) signal (not built)

**Label-only scores equal the undefended model's.** MemGuard leaves every predicted label unchanged, by construction and checked in the run. Any label-only attack, such as Choquette-Choo et al.'s augmentation correctness vector, therefore gives exactly the same result on MemGuard as on the undefended model. The ablation needs only one label-only run on the undefended target and one on HAMP.

**Cost.** HAMP's `attack.py` / `atk.sh` already implements the augmentation attack. It needs the PyTorch environment from the HAMP README, the dataset zip and a GPU, which we estimate at under an hour per dataset.

**Caveat.** HAMP Table VIII reports the augmentation attack at 0.02% TPR at 0.1% FPR on both undefended and MemGuard CIFAR-10. At the draft's metric the "flip" may therefore not appear. Choquette-Choo et al. show it with accuracy and AUC.
