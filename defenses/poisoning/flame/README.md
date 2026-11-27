# Poisoning and backdoors: FLAME vs Yang et al.

| | |
|---|---|
| Defence | FLAME (Nguyen et al., USENIX Security 2022). Implementation and FL harness: [zhmzm/FLAME](https://github.com/zhmzm/FLAME) at `fd4cef2ac2d5718bad84cf6449e0e3d1e5faf0b5` (unofficial, no licence file) |
| Attack (fixed) | Optimisation-based framework (Yang et al., CIKM 2024, arXiv:2407.15267): [Yuxin104/BreakSTOAPoisoningDefenses](https://github.com/Yuxin104/BreakSTOAPoisoningDefenses) at `e3b7b49752bc04ce31e234cd6050952206513cea` |
| Run | Kaggle: [`launch/kaggle/poisoning/flame`](../../../launch/kaggle/poisoning/flame/run.py). FLAME's own claim only: BadNet, Fashion-MNIST CNN, `--malicious 0.2 --poison_frac 0.5 --lr 0.1 --epochs 200`, with `--defence flame` and the `--defence avg` baseline, harness patched as below. Yang et al.'s attack is not wired in yet |
| Run, paper setting | Kaggle: [`launch/kaggle/poisoning/flame_cifar`](../../../launch/kaggle/poisoning/flame_cifar/run.py). FLAME's own claim in the paper's image-classification setting: CIFAR-10, ResNet-18, IID, DBA and BadNet, FLAME vs FedAvg. See below |
| Run, Table 3 setting | Kaggle: [`launch/kaggle/poisoning/flame_dba`](../../../launch/kaggle/poisoning/flame_dba/run.py). DBA's own code, config and pretrained model, FLAME as specified in Alg. 1. Follows [`../ATTACK_PROTOCOL.md`](../ATTACK_PROTOCOL.md); supersedes `flame_cifar` as the test of Table 3 |

## FLAME's paper setting (CIFAR-10)

Read from arXiv:2101.02281 v5 (extended version, 2023). Section numbers are from that version and may differ from the USENIX 2022 camera-ready.

| Setting | Paper | Quote / source | `flame_cifar` |
|---|---|---|---|
| Model | ResNet-18 Light, ~2.7M params | Table 2; App. A: "a lightweight version of the ResNet18 model [29] with 4 convolutional layers with max-pooling and batch normalization" | Harness `ResNet18`, full ~11M. **Differs** |
| Clients, PMR | 100, 20% | App. A: "The experimental setup consists of 100 clients and uses a PMR of 20%" | `--num_users 100 --malicious 0.2` |
| Data split | IID | §7.1: "the data are highly IID (each client is assigned a random set of images)". But §7.3: "since the IC data are non-IID", so the paper is inconsistent | `--iid 1` (harness `iid_cifar.npy`, 500 images per client, no augmentation) |
| Backdoor | one class to another | App. A: "The adversary aims at changing the predicted label of one class of images to another class of images" | class 7 to class 5 (harness default; the paper names no classes) |
| Attacks | constrain-and-scale, DBA, Edge-Case, PGD | §7.1, Table 3: "using the same attack settings as in the original works" | DBA (zhmzm re-implementation) and BadNet. Constrain-and-scale is not in the harness. **zhmzm's DBA is not the original DBA setting**, see below |
| Noise | λ = 0.001 for IC | App. B.3: "λ = 0.001 for IC and NLP" | `--noise 0.001` (σ = noise × median norm) |
| Clustering | HDBSCAN on pairwise cosine distances, min_cluster_size n/2+1, min_samples 1 | §4.3.1: "pairwise cosine distances to measure the angular differences between all model updates"; App. E: "min_cluster_size=N/2 + 1 and min_samples=1" | Parameters match. **Differs** in what is clustered: cosine on full local weights (BN statistics included), not on updates, and the distance matrix is passed with the default Euclidean metric, not `metric="precomputed"`. See below |
| Clipping | median of the L2 norms of all n updates | §4.3.2: "the clipping bound S_t as the median of the L2-norms of all n model updates" | same in harness |
| Clients per round, rounds, local epochs, batch size, learning rate, PDR | unstated | | 10 per round, 200 rounds, 2 epochs, batch 64, lr 0.1 (zhmzm README example, which is generic over `--dataset cifar,fashion_mnist`); PDR 0.5 |

Paper numbers to compare against (Table 3, CIFAR-10, BA / MA in %): DBA, no defence 93.8 / 57.4, FLAME 3.2 / 76.2. Constrain-and-scale, no defence 81.9 / 89.8, FLAME 0 / 91.9. BadNet is not in the paper; the zhmzm README reports 70.2 / 80.38 without defence and 3.33 / 78.77 with FLAME (CIFAR-10, ResNet18, IID), unofficial, and from a different configuration (its example uses `--malicious 0.1 --poison_frac 1.0 --attack_goal -1 --triggerX 27`).

## Results, `flame_cifar` (sok-flame-cifar v2)

Commit `2811ada`, patch sha256 `eb82ad5c...`, Tesla T4, seed 1, all 4 runs completed 200 rounds with no NaN or traceback. Raw output in `results/kaggle/sok-flame-cifar/` (git-ignored). BA / MA in %, round 199; "BA last 10%" is the mean over rounds 180 to 199.

| Run | BA final | BA last 10% | BA mean, rounds 100-199 (min / max) | MA final | Malicious / benign admitted |
|---|---|---|---|---|---|
| DBA, FLAME | 52.7 | 54.4 | 49.6 (11.0 / 80.1) | 75.1 | 77.0% / 62.1% |
| DBA, FedAvg | 72.3 | 68.4 | 54.0 (10.3 / 77.7) | 80.4 | |
| BadNet, FLAME | 86.6 | 89.6 | 87.7 (72.9 / 93.9) | 75.1 | 79.3% / 59.6% |
| BadNet, FedAvg | 93.6 | 93.4 | 91.4 (81.9 / 95.2) | 80.6 | |

**This is not a test of FLAME's Table 3 claim** (DBA BA 3.2 with FLAME); see "Fidelity audit" below. What it shows: zhmzm's FLAME, with the two fixes, does not stop a continuous, unscaled DBA-style or BadNet backdoor trained from scratch (single seed). FLAME lowers DBA BA by about 14 points (last 10%) and BadNet BA by about 4, and costs about 5 points of MA. HDBSCAN returned exactly one cluster in all 200 rounds of both FLAME runs, of size 6 (= min_cluster_size) in 136 (DBA) and 158 (BadNet) rounds. Both attackers were admitted in 121 (DBA) and 132 (BadNet) of the 199 attacked rounds, so the filter keeps malicious updates more often than benign ones. DBA BA swings widely under both defences. BA in the first few rounds is meaningless (MA about 10%, the model predicts one class).

## Fidelity audit of `flame_cifar` (2026-11-26)

Read-only audit against arXiv:2101.02281 v5, the patched harness and DBA's released code ([AI-secure/DBA](https://github.com/AI-secure/DBA), `utils/cifar_params.yaml`; the DBA paper itself was not read). Verdict: the run is **not a fair test of Table 3's DBA row**, because both the attack and the defence differ from the paper.

Attack (most likely cause of the gap):

- DBA's config uses 4 fixed adversaries (`adversary_list: [17, 33, 77, 11]`), one single-shot attack each at rounds 203, 205, 207, 209, scales the poisoned update by 100 (`scale_weights_poison: 100`, `baseline: false`), `internal_poison_epochs: 6`, all-to-one to label 2, and resumes from a 200-round pretrained model (`resumed_model_name: cifar_pretrain/model_last.pt.tar.epoch_200`). zhmzm's DBA attacks with 2 of 10 clients in every round from round 1, unscaled, 2 local epochs, one-to-one 7 to 5, from scratch. Median clipping largely cancels a x100 update, so the original DBA is an easier case for FLAME.
- Whether Table 3's DBA was scaled, which round BA is read at, and which PMR / IID setting the DBA row used are unstated in the FLAME paper.
- Training from scratch also breaks this project's "pretrained checkpoints only" rule.

Defence:

- `utils/defense.py`, `flame()`: cosine distances are computed on the full local weights (BN running statistics included), not on updates as in §4.3.1. Alg. 1 line 6 literally says "cij is the cosine distance between Wi and Wj", so the paper is itself inconsistent; the updates version is the one §4.3.1 argues for.
- `hdbscan.HDBSCAN(...).fit(cos_list)` uses the default Euclidean metric, so each row of the distance matrix is treated as a feature vector. App. E: "HDBSCAN first uses the given distances to build a minimal spanning tree". Likely explains one cluster per round and attackers admitted more often than benign clients (77% vs 62%; malicious share of rejected updates 13%, below the 20% base rate).
- Norms, S_t, gamma, clipping, aggregation and sigma = 0.001 S_t match Alg. 1 and App. B.3 (after the patch).

Other: model not near convergence (MA 75 to 80 vs benign 92.2 in Table 4; App. H Table 8 shows FLAME's BA against 3DFed falling from 53.9 to 9.9 as the pretrained model converges further), only 10 clients per round, single seed, full ResNet-18 instead of Light.

To make it a fair test: (1) cosine on weight/bias updates, float64 matrix, `metric="precomputed"`; (2) run Table 3's DBA row with DBA's own code, config and pretrained checkpoint; (3) keep the current continuous attack as a separately named configuration, re-run from a converged checkpoint; (4) at least 3 seeds, BA as a windowed mean. Not done yet.

## Notes on the upstream code

Two bugs, both fixed in both Kaggle kernels (`HARNESS_PATCH`, 5 changed lines, the same list in `flame/run.py` and `flame_cifar/run.py`, applied after a hard reset to the pinned commit and recorded in `run.json` with its sha256). Each was confirmed with a small probe before the fix:

- **FLAME clipping uses the wrong norm.** `utils/defense.py`, `flame()`: `gama = clip_value/norm_list[i]` takes the norm of the i-th client overall, not of the i-th admitted client. Alg. 1 line 11 uses γ = S_t / e_{b_l}. In the probe (10 clients, 2 rejected), benign updates below the median were shrunk to 12% because they borrowed the rejected clients' norms, and updates above the median were not clipped. Fix: `norm_list[benign_client[i]]`. **Fashion-MNIST runs from before the patch used the unfixed code, so their FLAME result is not FLAME as specified.** Only cite runs whose `run.json` has `upstream_patch`.
- **`save_img()` changes the image it saves.** `models/MaliciousUpdate.py` and `models/test.py` min-max rescale the image in place while writing a preview PNG. Poisoned training images are CPU tensors, so they are always rescaled to [0, 1]. Triggered test images are CUDA tensors on a GPU; for 3-channel input `.cpu()` copies them, so they are *not* rescaled. On CIFAR-10 with a GPU, the backdoor was therefore trained and tested on different inputs. (For 1-channel Fashion-MNIST both sides were rescaled, so the pre-patch run was consistent on this point; with the patch neither side is.) Fix: write the PNG from a copy.

Other behaviour worth knowing:

- `--poison_frac 0.5` with a one-to-one backdoor does not poison 50% of a client's data: in each batch, about half of the source-class (7) images (floor(n/2)+1) are triggered and relabelled, replacing the clean ones, about 5 to 7% of the samples.
- Every round from round 1 has exactly int(0.2 × 10) = 2 malicious clients; round 0 has none. BadNet attackers are drawn from clients 0 to 20; DBA cycles through clients 0 to 19, each holding one quarter of the 5×5 trigger, so DBA's global trigger is the BadNet square.
- FLAME's norms and clipping bound use only `weight` and `bias` tensors; the cosine distances use the full state dict except `num_batches_tracked`. The paper does not say how BatchNorm statistics are handled, but §2.1 defines the model vector as "p parameters (indicating both weights and biases)", so including BN buffers in the cosine is a deviation.
- The FLAME admission rates the harness prints ("proportion of malicious are selected") treat positions 0 and 1 as malicious in every round, including round 0, which has no attackers. Over 200 rounds this shifts the rate by at most 2/400.
- `--seed` is parsed but never used. `run.py` seeds `random`, NumPy and torch before starting `main_fed.py` through `runpy`, without editing the upstream file.
- `models/test.py` writes `./save/test_trigger.png` regardless of `--save`, so `./save` must exist.
- With the `flame` (Fashion-MNIST) kernel's flags the harness reports 100 clients, 10 per round, 3 local epochs, batch size 50, backdoor from class 7 to class 5, square trigger at (0, 0). With the `flame_cifar` flags it reports 2 local epochs and batch size 64.
- Datasets download to `../data` relative to the repo.

