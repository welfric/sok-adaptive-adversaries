# Poisoning / backdoors: common attack protocol

Ahmad, 2026-11-26. Draft for the team; open decisions are marked **Open**.

This is the single setting in which every attack in the poisoning field is run (DBA, constrain-and-scale, BadNet, Yang et al.'s framework, and its axis ablations), against every defence in this field (FLAME now). Only the parts listed under "What an attack may change" differ between runs. Everything else is fixed, so a change in backdoor accuracy can be attributed to the attack configuration.

The base is DBA's released code and CIFAR-10 config ([AI-secure/DBA](https://github.com/AI-secure/DBA) at `3cabf7e`, `utils/cifar_params.yaml`), because FLAME's Table 3 says its attacks use "the same attack settings as in the original works" (FLAME §7.1) and DBA is the Table 3 attack with public code. Values marked "DBA config" are read from that file and code; we have not read the DBA paper itself. FLAME quotes are from arXiv:2101.02281 v5 and were checked against the PDF text.

## 1. Fixed FL setting

| Setting | Value | Source |
|---|---|---|
| Dataset | CIFAR-10, no augmentation, no normalisation (`ToTensor` only) | DBA config / `image_helper.py` |
| Model | DBA's `ResNet18` (`models/resnet_cifar.py`, base width 32, not the torchvision ResNet-18) | DBA code. FLAME uses "ResNet-18 Light" (Table 2), architecture unstated; DBA's narrower ResNet is the closest public match |
| Starting point | DBA's pretrained clean model at round 200 (`cifar_pretrain/model_last.pt.tar.epoch_200`, Google Drive link in the DBA README). Record its sha256 | DBA config `resumed_model`. Meets our "pretrained checkpoints only" rule. FLAME does not say where Table 3 starts (unstated); §7.4 evaluates Fig. 9 "in a training round t where the global model Gt is close to convergence" |
| Rounds | 201 to 300 (100 rounds after the checkpoint) | DBA config `epochs: 300` |
| Clients | 100 in total, 10 per round, sampled uniformly at random each round | DBA config `number_of_total_participants`, `no_models`; FLAME App. A: "100 clients" |
| Data split | **IID** (each client gets a random 500 images) as the main setting; Dirichlet α = 0.5 as a second setting | FLAME §7.1: "the data are highly IID (each client is assigned a random set of images)". DBA config uses Dirichlet 0.5. **Open:** FLAME §7.3 says "since the IC data are non-IID", so we run both |
| Benign local training | SGD, lr 0.1, momentum 0.9, weight decay 5e-4, batch 64, 2 local epochs | DBA config |
| Server update | G ← G + η · (aggregate − G), η = 0.1, equal client weights | DBA `average_shrink_models` (`eta: 0.1`). **Open:** FLAME Alg. 1 line 12 is a plain mean (η = 1). We keep η = 0.1 for every defence so that only the aggregation rule differs between defence and baseline |
| Seeds | 1, 2, 3 (one seed until the setup is confirmed). The launcher sets Python, NumPy and torch seeds; DBA's `main.py` hard-codes seed 1, so this needs a launcher patch | |
| Hardware | Kaggle, one Tesla T4 | |

## 2. Fixed backdoor task

| Setting | Value | Source |
|---|---|---|
| Type | All-to-one: any triggered image to label 2 | DBA config `poison_label_swap: 2`. FLAME App. A describes one class to another; we follow the original attack |
| Global trigger | Union of four 1x6 white bars: rows 0 and 4, columns 0 to 5 and 9 to 14, all three channels set to 1 | DBA config `0_` to `3_poison_pattern` |
| Local triggers | One bar per adversary (distributed attacks only) | DBA config |
| Centralised attacks | Use the global trigger (BadNet, constrain-and-scale, Yang et al.) | So every attack is tested on the same trigger |

## 3. Attacker budget (the FL analogue of a fixed query budget)

Held equal across all attacks and all axis configurations:

| Budget item | Value | Source |
|---|---|---|
| Malicious clients | 4 fixed clients: 17, 33, 77, 11 | DBA config `adversary_list` |
| Attacker participations | 4 client-rounds in total, one adversary each at rounds 203, 205, 207, 209 | DBA config `*_poison_epochs`, `is_random_adversary: false` |
| Poisoned samples | 5 of every 64-sample batch | DBA config `poisoning_per_batch` |
| Local poison training | 6 epochs, lr 0.05, momentum 0.9, weight decay 5e-4, MultiStepLR with milestones at 0.2 and 0.8 of the 6 epochs, gamma 0.1 | DBA config and `image_train.py` (the config's `poison_momentum` and `poison_decay` are unused) |

**Open: PMR.** FLAME App. A says "uses a PMR of 20%", while DBA's config has 4 of 100 malicious clients (4%), one per attacked round. Which one Table 3's DBA row used is unstated. The main setting follows DBA's config; a second setting with 20 malicious clients (5 per local trigger, 2 of 10 per round in rounds 201 to 300) can be added with the same per-client budget.

## 4. What an attack may change

Only the malicious clients' behaviour:

- the trigger split (distributed vs global trigger),
- the local objective (e.g. an added distance or cosine term),
- post-training changes to the update (e.g. scaling, projection to a norm bound),
- what the attacker knows and uses, as set by the axis configuration in section 6.

Default boosting is DBA's: the update is scaled by 100 (`scale_weights_poison: 100`, `baseline: false`), which equals n / η = 10 / 0.1, i.e. model replacement. An attack that scales differently records its factor. Attacks never change the benign clients, the data split, the schedule or the budget in section 3.

**Yang et al.** was evaluated on Fashion-MNIST with a CNN in their paper. In this protocol it is ported to the setting above. Any part of it that does not fit (e.g. it needs more attacker rounds) is recorded as a deviation, not silently changed.

## 5. Defences and baseline

| Defence | Setting |
|---|---|
| No defence (baseline) | DBA's FedAvg, η = 0.1 |
| FLAME | Alg. 1 as specified, run inside the DBA code, with η as in section 1: (1) cosine distances between the clients' weight and bias **updates** (§4.3.1: "pairwise cosine distances to measure the angular differences between all model updates"); (2) HDBSCAN on the precomputed float64 distance matrix (`metric="precomputed"`), min_cluster_size n/2+1, min_samples 1, single cluster allowed (App. E: "min_cluster_size=N/2 + 1 and min_samples=1"); (3) S_t = median of the L2 norms of all n updates (§4.3.2); (4) clip each admitted update by min(1, S_t / e_i) (Alg. 1 line 11); (5) average the admitted clipped updates; (6) add Gaussian noise σ = λ · S_t, λ = 0.001 (App. B.3: "λ = 0.001 for IC and NLP") to the weights and biases. BatchNorm running statistics are clipped with the same factor and averaged over the admitted clients, not noised (**Open:** the paper does not say). If HDBSCAN labels every client as noise, all are admitted and the round is logged as a fallback (unstated in the paper). Implementation: `FLAME_SOURCE` in `launch/kaggle/poisoning/flame_dba/run.py` |

The zhmzm/FLAME harness is no longer used for this field. Its clustering differs from the paper (see `flame/README.md`, "Fidelity audit").

## 6. Axis configurations

As in CLAUDE.md, each attack is run in the configurations below, at the budget in section 3.

| Config | (i) knows defence | (ii) defence-specific design | (iii) feedback |
|---|---|---|---|
| Full | yes: FLAME's mechanism and parameters (n/2+1, λ, the median bound) | yes: e.g. keeps the update's norm and cosine distance inside what FLAME admits | online |
| −(i) | no | yes | online |
| −(ii) | yes | no | online |
| (iii) offline | yes | yes | offline: tuned on a surrogate FL simulation prepared in advance, no use of the live global model beyond training on it |
| −(iii) | yes | yes | none |

In this field:

- **(i)** means the attacker knows the aggregation rule and its parameters, not the benign clients' updates.
- **(ii)** means the attack is shaped by that rule (norm bound, cosine distance, noise), not by generic boosting.
- **(iii) online** means the attacker uses information from the live training run to adapt its update, for example the global model's response to its earlier updates, or estimates of the clipping bound from successive global models.

**Open (needs a team decision before annotation):** every FL client, attackers included, trains on the current global model each round. We propose treating that as part of the setting, not as axis (iii), so that (iii) online means adapting the attack *strategy* to observed outcomes. Under this reading DBA, BadNet and constrain-and-scale are (i) no, (ii) no, (iii) none, i.e. non-adaptive baselines.

## 7. Evaluation and reporting

- **MA:** accuracy on the full clean CIFAR-10 test set.
- **BA:** the fraction of test images whose true label is not 2, stamped with the global trigger, that are classified as 2 (DBA `Mytest_poison`, `poison_test_dataset`). For distributed attacks, also report BA per local trigger.
- **When:** report BA and MA at rounds 209 (last attack), 210, 220, 250 and 300, the maximum BA from round 203 on, and the mean BA over rounds 210 to 300. FLAME does not say at which round Table 3's BA is taken (unstated), so we always give the whole curve.
- **FLAME diagnostics, per round:** HDBSCAN labels, admitted clients, whether each adversary was admitted, S_t, each client's update norm, and the noise σ.
- **Compare with:** FLAME Table 3, CIFAR-10, BA / MA in %: DBA with no defence 93.8 / 57.4, with FLAME 3.2 / 76.2; constrain-and-scale with no defence 81.9 / 89.8, with FLAME 0 / 91.9.

## 8. Logging (every run)

`run.json` with:

- our commit;
- the DBA commit;
- the sha256 of the checkpoint and of the full resolved config;
- the attack and its axis configuration;
- the seed;
- torch, CUDA and the GPU name;
- any launcher patch, with its sha256.

Also per-round `curves.json`, and the full stdout. Runs without these files are not cited.

## 9. Known deviations from the papers

- FLAME aggregation uses η = 0.1 instead of Alg. 1's plain mean (section 1). The noise keeps the literal σ = λ · S_t, so relative to the applied update it is 10 times larger than with η = 1. This favours FLAME.
- The model is DBA's ResNet18, whose relation to FLAME's "ResNet-18 Light" is unstated.
- The backdoor is all-to-one, where FLAME App. A describes one class to another.
- BatchNorm statistics are handled as above; the paper says nothing about them.
- DBA's code needs compatibility shims to run on current PyTorch and PyYAML (e.g. `yaml.load` without a Loader; Visdom is stubbed out). These are listed in the launcher and do not change the algorithm.
