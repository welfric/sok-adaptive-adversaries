# Implementation plan: defences and attacks

Draft, 2026-11-25. Ahmad chose the five pairs. The axis ablations in section 4 are **proposals**: before a config runs, check it against the attack paper's threat-model section and record a quote and section number (CLAUDE.md rule). I found the repo links below through search and have not opened them yet.

## 0. Current scope (2026-11-25)

For now, only the **full** attack runs against each defence (step (b) in section 4). The ablation configs are deferred. Other contributors will add their own pairs.

What each repo actually ships, from its README and file tree:

| Pair | Ready by tomorrow on Kaggle? | Why |
|---|---|---|
| Blacklight / OARS | Yes | Repo ships the CIFAR-10 ResNet-20 checkpoint, the Blacklight implementation, and `standard`/`adaptive` configs for Square, HSJA, NES, QEBA, SurFree and Boundary. Only CIFAR-10 has to be exported as PNG + JSON. |
| Meta-SecAlign-8B / Simple Adaptive Attacks | Probably, prompt injection only | Needs gated HF access (Llama-3.1 and Meta-SecAlign). Random search must be ported to Llama-3.1's template and SecAlign's `input` role. Prompt injection scores by string match; jailbreaking needs a GPT-4 judge (OpenAI key), so defer it. |
| KGW / Watermark Stealing | Possibly, spoofing only | Server query outputs are downloadable, so the querying step is skipped by default. The 7B attacker model runs on a T4, but the setup installs Flash Attention, which does not support T4/P100, so that step must be removed. GPT-4 judge needs an OpenAI key. |
| FLAME / Yang et al. | Likely, Fashion-MNIST first | Yang et al.'s repo is four ~1 KB attack functions, no harness. [zhmzm/FLAME](https://github.com/zhmzm/FLAME) (unofficial, no licence file) supplies the harness: `main_fed.py`, `flame()` in `utils/defense.py` (HDBSCAN on cosine, median-norm clipping, noise), BadNet/DBA backdoors, CNN on Fashion-MNIST, ResNet-18/VGG on CIFAR-10. About 40 lines of glue (below) connect the two. |
| HAMP / Aerni et al. | No | `scripts/submit_hamp_train.sh` trains 640 models, with the comment "a bit more than 1h per model" on an RTX 3090. Even one setting with 64 shadow models is tens of GPU-hours. |

## 1. The five pairs

| Field (folder) | Defence | Attack | Defence code / checkpoint | Attack code |
|---|---|---|---|---|
| `adversarial_robustness` | Blacklight (stateful query detection) | OARS (Feng et al., CCS 2023) | Blacklight implementation in the OARS repo (to verify) | [nmangaokar/ccs_23_oars_stateful_attacks](https://github.com/nmangaokar/ccs_23_oars_stateful_attacks) |
| `jailbreak_prompt_injection` | SecAlign, as Meta-SecAlign-8B (LoRA on Llama-3.1-8B-Instruct, gated) | Simple Adaptive Attacks (Andriushchenko et al., ICLR 2025) | [facebook/Meta-SecAlign-8B](https://huggingface.co/facebook/Meta-SecAlign-8B), [facebookresearch/Meta_SecAlign](https://github.com/facebookresearch/Meta_SecAlign) | [tml-epfl/llm-adaptive-attacks](https://github.com/tml-epfl/llm-adaptive-attacks) |
| `privacy` | HAMP (Chen and Pattabiraman, NDSS 2024) | Label-only adaptive MIA (Aerni, Zhang and Tramèr, CCS 2024) | [DependableSystemsLab/MIA_defense_HAMP](https://github.com/DependableSystemsLab/MIA_defense_HAMP) | [ethz-spylab/misleading-privacy-evals](https://github.com/ethz-spylab/misleading-privacy-evals) |
| `watermarking` | KGW (Kirchenbauer et al.) | Watermark Stealing (Jovanović, Staab and Vechev, ICML 2024) | KGW variants that ship with the attack repo | [eth-sri/watermark-stealing](https://github.com/eth-sri/watermark-stealing) |
| `poisoning` | FLAME (Nguyen et al., USENIX Security 2022) | Learning-based framework (Yang et al., CIKM 2024, arXiv:2407.15267) | [zhmzm/FLAME](https://github.com/zhmzm/FLAME) (unofficial; no licence file, so reference it as a submodule and do not copy its code into our repo) | [Yuxin104/BreakSTOAPoisoningDefenses](https://github.com/Yuxin104/BreakSTOAPoisoningDefenses) |

## 2. Design principles

1. **One codebase, three launchers.** Every run is `python -m attacks.<field>.run --config <yaml>`, on the MacBook (smoke tests at tiny budgets), on Kaggle, and over SSH on a GPU box. Launchers differ only in paths and in how secrets arrive. Folder names use underscores so they import as Python packages.
2. **Upstream code is wrapped, never rewritten.** Each upstream repo becomes a git submodule in `third_party/`, pinned to a commit. Any change to upstream goes in a `.patch` file in the field folder, so a reviewer sees exactly what we changed.
3. **Axes are config switches that the code enforces** (section 3), so an ablation cannot quietly leak an axis it was meant to remove.
4. **Fixed query budget per pair.** The wrapper counts queries; we do not rely on the attack's own counter.
5. **One JSONL record per run**: config, seed, our commit, upstream commit, hardware, queries used, metrics.

## 3. Shared layer (`common/`)

**`Defence`**, the interface every wrapper in `defenses/<field>/` implements:

- `respond(x)`: what the deployed system returns for one query, i.e. what a black-box user sees.
- `spec()`: the mechanism, parameters and a handle to the code. This is what axis (i) grants.
- `make_replica(seed)`: a copy the attacker could build themselves, with their own key, data or thresholds. Offline feedback uses it.

**`Oracle`**, the only object an attack is given, built from the config:

| Config key | Effect |
|---|---|
| `knowledge: true / false` | `oracle.spec()` returns the spec, or raises `NotGranted` |
| `feedback: online` | `oracle.query()` goes to the real target, is counted, and raises `BudgetExceeded` at the cap |
| `feedback: offline` | `oracle.query()` goes to `make_replica()`; the real target sees only `oracle.submit()` (final answers, a fixed number per item) |
| `feedback: none` | `oracle.query()` raises; only `submit()` works |
| `design: specific / generic` | Axis (ii) is not a gate. It picks which attack variant runs |

That turns the ablation table in CLAUDE.md into five YAML files per field: `full`, `no_knowledge`, `generic`, `offline`, `no_feedback`.

The gate has one limit: it cannot stop an attack that has defence parameters hard-coded. For `no_knowledge`, the attack wrappers must read defence parameters only through `spec()`, and the code review for each field checks this.

**`runlog.py`**: seeds python, numpy, torch and CUDA, writes the JSONL record, and records `git rev-parse` for the repo and its submodules.

## 4. Per-field plan

Each field follows the same three steps:
- (a) Add the submodule and reproduce one headline number from the attack paper using its own scripts.
- (b) Wrap the defence and attack in `Defence`/`Oracle`, and check that `full` reproduces (a).
- (c) Run the four ablations.

Step (a) comes first because if we cannot reproduce the paper's own number, the ablations cannot be interpreted.

### 4.1 `adversarial_robustness`: Blacklight vs OARS

- **Defence wrapper.** A CIFAR-10 classifier (the OARS repo's checkpoint, or a standard pretrained ResNet) behind Blacklight. `respond(x)` returns the label or scores, or a rejection. `make_replica` is Blacklight with the attacker's own parameters, applied to the attacker's own query history.
- **Attack wrapper.** OARS around one score-based base attack (e.g. Square or NES) and one decision-based one (e.g. HSJA). Choose them once we see which ones the repo supports.

| Config | Proposed setting |
|---|---|
| full | OARS, Blacklight parameters known, online queries |
| no_knowledge | OARS with no access to Blacklight's parameters (defaults, or inferred from rejections) |
| generic | Base attack without OARS, same knowledge, same budget |
| offline | OARS against the replica with a surrogate classifier; final images submitted once |
| no_feedback | White-box PGD on a surrogate classifier, transferred and submitted once |

- **Metric.** Attack success rate at a fixed query budget, plus the detection rate. Report Blacklight's own claim beside it, quoted.
- **Compute.** One Kaggle T4. This is the cheapest pair, so it goes first.
- **Risk.** If OARS never reads defence parameters, `full` and `no_knowledge` will match. That is a finding to report: OARS would then rest on (iii)-online, not on (i).

### 4.2 `jailbreak_prompt_injection`: Meta-SecAlign-8B vs Simple Adaptive Attacks

There are two tasks on one model:
- **Prompt injection** (SecAlign's own threat model). The trusted instruction goes in the `user` role, and the untrusted data, carrying the injected instruction, goes in the `input` role. Success means the model follows the injection, e.g. emits a target string. Use a subset of the injection benchmark that the Meta_SecAlign repo evaluates on.
- **Jailbreaking.** A subset of JailbreakBench behaviours, the format the attack repo already uses. SecAlign was built as a prompt-injection defence, so check what the Meta SecAlign paper claims about jailbreaks before presenting any jailbreak number as a failure of the defence's own claim.

Wrappers:
- **Defence wrapper.** vLLM serving Llama-3.1-8B-Instruct with the Meta-SecAlign LoRA. `respond()` returns text plus the top-k logprobs of the first tokens, which is what random search needs. `make_replica()` is undefended Llama-3.1-8B-Instruct, or the original SecAlign checkpoint if one is released.
- **Attack wrapper.** Random search over an adversarial suffix, together with the prompt template, ported from tml-epfl so that it calls `Oracle`.

| Config | Proposed setting |
|---|---|
| full | Template and suffix placement use knowledge of SecAlign's role separation and chat template; random search on online logprobs from Meta-SecAlign-8B |
| no_knowledge | Attacker assumes a plain Llama-3.1-8B-Instruct deployment (no input role, no LoRA); same search |
| generic | The paper's generic template, not tailored to SecAlign; same search |
| offline | Random search against the replica; best suffix submitted once per item |
| no_feedback | Template only, one query per item |

- **Budget.** A fixed number of target forward passes per item, with restarts counted inside it.
- **Metric.** Prompt injection: ASR by string match. Jailbreak: ASR by one fixed judge, either JailbreakBench's judge or a local classifier. Judge calls do not count towards the budget.
- **Compute.** 8B in fp16 is about 16 GB. Run it on Kaggle 2×T4 with vLLM tensor parallelism. Avoid 4-bit, because quantising changes the target. Random search is mostly sequential, so if it is too slow, move it to the SSH GPU.
- **Access.** Both Llama-3.1 and Meta-SecAlign are gated. Request access now. The HF token goes in Kaggle Secrets, never in the repo. The model card and the paper disagree on the licence, but research use is fine under either.

### 4.3 `privacy`: HAMP vs Aerni et al. label-only attack

- **Defence.** HAMP is a training-time defence that also changes outputs at test time; confirm the exact components in the paper. A HAMP model exists only if the repo ships a checkpoint, so check that first.
- **Attack.** Aerni et al. evaluate with canaries and shadow models, and their attack adapted to HAMP is label-only. Note that shadow models are already a form of offline feedback.

| Config | Proposed setting |
|---|---|
| full | Shadow models trained with HAMP; label-only attack; several online label queries per candidate (augmentations) |
| no_knowledge | Shadow models trained without HAMP |
| generic | Score-based LiRA on the (modified) scores HAMP returns |
| offline | Decision rule calibrated on shadow models only; one query per candidate |
| no_feedback | Fixed global threshold, no shadow calibration; one query per candidate |

- **Metric.** TPR at low FPR on canaries. Use the same operating point as Aerni et al. (to verify).
- **Compute.** This is the highest-risk field. The target and the shadow models all have to be trained, which conflicts with the "pretrained checkpoints only" rule. A reduced protocol (CIFAR-10, a small ResNet, fewer shadow models, fewer epochs, each stated as a deviation) still needs many GPU-hours: use the SSH GPU, or split the runs across team Kaggle quotas. Needs a team decision (section 7).

### 4.4 `watermarking`: KGW vs Watermark Stealing

- **Defence.** KGW on a server LLM, using the variants in the repo (e.g. KGW2-SelfHash). `respond(prompt)` returns watermarked text, and the detector returns a z-score. `make_replica` is the same scheme with the attacker's own key.
- **Attack.** Query the server, estimate context-conditioned green-list statistics, then spoof (produce text the detector flags) and scrub (remove the watermark).

| Config | Proposed setting |
|---|---|
| full | Knows the KGW variant and context width; estimator matched to it; online queries to the server |
| no_knowledge | Variant and context width unknown; estimator run with agnostic or wrong assumptions |
| generic | Scrub with a generic paraphraser. Generic spoofing has no counterpart, so mark it N/A |
| offline | Steal from the replica (attacker's own key), apply to the real detector |
| no_feedback | No queries: spoofing impossible, scrubbing by paraphrase only |

- **Note on the offline config.** I expect it to fail, because the keys differ. That would be a useful contrast with Lukas et al.'s surrogate-key attack on image watermarks, but claim it only once we have numbers.
- **Budget.** Number of server responses, plus total tokens.
- **Metric.** The repo's spoofing and scrubbing success rates at the detector threshold, with its text-quality filter.
- **Compute.** The cost is generating the query corpus from the server model. Use vLLM on Kaggle 2×T4 or the SSH GPU, and the repo's smallest default models.

### 4.5 `poisoning`: FLAME vs Yang et al.

- **Setup.** An FL simulation with a fixed number of clients, malicious fraction, rounds and dataset (MNIST/FMNIST for smoke tests, then CIFAR-10 with a small CNN). FLAME clusters updates by cosine distance, then clips and adds noise. Training is part of FL itself, and small models keep it within Kaggle or the MacBook.

| Config | Proposed setting |
|---|---|
| full | Knows FLAME (clip bound, clustering); Yang et al. framework tailored to FLAME; uses the global model each round |
| no_knowledge | Framework tuned against another aggregator (e.g. FedAvg), deployed against FLAME |
| generic | Generic backdoor (e.g. constrain-and-scale or DBA), same knowledge |
| offline | Attack parameters learned in a local FL simulation with the attacker's own data; fixed during the real rounds |
| no_feedback | One poisoned update computed once, reused every round |

- **Budget.** Fixed number of rounds and malicious clients.
- **Metric.** Backdoor accuracy and main-task accuracy.
- **Compute.** Kaggle T4; MacBook for MNIST.
- **Harness.** zhmzm/FLAME's `main_fed.py`. In each round it trains the malicious clients first, then the benign ones. It builds `w_locals` (state dicts) and `w_updates` (deltas), then calls `flame(w_locals, w_updates, w_glob, args)`.
- **Glue for Yang et al.'s `FLAME_attack`.** Insert the glue after the client loop and before aggregation:
  1. Flatten `w_updates` with `parameters_dict_to_vector_flt` into a dict indexed so that malicious clients come first (`id < num_corrupt`).
  2. Compute `avg_updates` and `dis`, using the definitions in the paper (to read).
  3. Call `FLAME_attack` for each malicious client.
  4. Write the result back into both `w_updates` and `w_locals`, since `flame()` reads both.
- **Threat model visible in the code.** `FLAME_attack` reads the cosine distance to every benign client's update in the same round. So the attacker sees benign updates (an omniscient, online setting). Annotate that, with a quote from the paper's threat-model section.
- **Risk.** zhmzm/FLAME is unofficial. Check `flame()` against Nguyen et al. (HDBSCAN `min_cluster_size = n/2 + 1`, median-norm clipping, noise level) and record any differences. Its README reports FLAME taking BadNet ASR from 70.2 to 3.33 on CIFAR-10 / ResNet-18 (IID). Reproduce that, or the Fashion-MNIST equivalent, before attacking.

## 5. Execution environments

- **MacBook.** `common/` development, unit tests on a toy defence, smoke tests at tiny budgets (CPU/MPS), poisoning on MNIST.
- **Kaggle** (the default for real runs). `launch/kaggle/<field>/` holds `kernel-metadata.json` and a thin notebook. The notebook clones the repo at a fixed commit (using a GitHub token from Kaggle Secrets if the repo is private), installs requirements, runs the five configs, and writes `results/*.jsonl` to `/kaggle/working`. Push it with `kaggle kernels push` and pull results with `kaggle kernels output`. Put checkpoints in Kaggle Datasets so they are not re-downloaded each session. The weekly GPU quota is limited, so estimate GPU-hours per field first.
- **SSH GPU** for jailbreak/PI, watermarking and privacy if Kaggle is too slow. `launch/ssh/run.sh <field> <config>` runs under tmux, or as an sbatch file if the machine uses Slurm. It calls the same entrypoint.

## 6. Order of work

1. `common/` (`Defence`, `Oracle`, `runlog`) plus tests on a toy defence. MacBook.
2. `adversarial_robustness`, end to end. It is the cheapest pair and proves the `Oracle` design.
3. `jailbreak_prompt_injection`. Request HF access on day one.
4. `poisoning`.
5. `watermarking`.
6. `privacy`, last, because it is the heaviest and waits on the training decision.

## 7. Decisions needed

1. **Training.** Privacy (HAMP plus shadow models) and poisoning (FL) both need training, and CLAUDE.md says "no training from scratch". Allow small-scale training for these two, or find HAMP checkpoints.
2. **Ablation scope.** Decided 2026-11-25: full attack only for now (section 0). Ablations are deferred.
3. **Field count.** Jailbreaking and prompt injection share one defence. Is that one field or two in the paper?
4. **Compute.** Is an SSH GPU available, and which scheduler does it use?
5. **Jailbreak judge.** Which judge, fixed across all configs.
