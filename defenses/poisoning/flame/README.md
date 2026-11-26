# Poisoning and backdoors: FLAME vs Yang et al.

| | |
|---|---|
| Defence | FLAME (Nguyen et al., USENIX Security 2022). Implementation and FL harness: [zhmzm/FLAME](https://github.com/zhmzm/FLAME) at `fd4cef2ac2d5718bad84cf6449e0e3d1e5faf0b5` (unofficial, no licence file) |
| Attack (fixed) | Optimisation-based framework (Yang et al., CIKM 2024, arXiv:2407.15267): [Yuxin104/BreakSTOAPoisoningDefenses](https://github.com/Yuxin104/BreakSTOAPoisoningDefenses) at `e3b7b49752bc04ce31e234cd6050952206513cea` |
| Run | Kaggle: [`launch/kaggle/poisoning/flame`](../../../launch/kaggle/poisoning/flame/run.py). FLAME's own claim only: BadNet, Fashion-MNIST CNN, `--malicious 0.2 --poison_frac 0.5 --lr 0.1 --epochs 200`, with `--defence flame` and the `--defence avg` baseline. Yang et al.'s attack is not wired in yet |
| Run, paper setting | Kaggle: [`launch/kaggle/poisoning/flame_cifar`](../../../launch/kaggle/poisoning/flame_cifar/run.py). FLAME's own claim in the paper's image-classification setting: CIFAR-10, ResNet-18, IID, DBA and BadNet, FLAME vs FedAvg. See below |

## FLAME's paper setting (CIFAR-10)

Read from arXiv:2101.02281 v5 (extended version, 2023). Section numbers are from that version and may differ from the USENIX 2022 camera-ready.

| Setting | Paper | Quote / source | `flame_cifar` |
|---|---|---|---|
| Model | ResNet-18 Light, ~2.7M params | Table 2; App. A: "a lightweight version of the ResNet18 model [29] with 4 convolutional layers with max-pooling and batch normalization" | Harness `ResNet18`, full ~11M. **Differs** |
| Clients, PMR | 100, 20% | App. A: "The experimental setup consists of 100 clients and uses a PMR of 20%" | `--num_users 100 --malicious 0.2` |
| Data split | IID | §7.1: "the data are highly IID (each client is assigned a random set of images)" | `--iid 1` (harness `iid_cifar.npy`, 500 images per client) |
| Backdoor | one class to another | App. A: "The adversary aims at changing the predicted label of one class of images to another class of images" | class 7 to class 5 (harness default; the paper names no classes) |
| Attacks | constrain-and-scale, DBA, Edge-Case, PGD | §7.1, Table 3: "using the same attack settings as in the original works" | DBA (zhmzm re-implementation) and BadNet. Constrain-and-scale is not in the harness |
| Noise | λ = 0.001 for IC | App. B.3: "λ = 0.001 for IC and NLP" | `--noise 0.001` (σ = noise × median norm) |
| Clustering | HDBSCAN, min_cluster_size n/2+1, min_samples 1 | §4.3.1; App. E | same in harness |
| Clipping | median of the L2 norms of all n updates | §4.3.2: "the clipping bound S_t as the median of the L2-norms of all n model updates" | same in harness |
| Clients per round, rounds, local epochs, batch size, learning rate, PDR | unstated | | 10 per round, 200 rounds, 2 epochs, batch 64, lr 0.1 (zhmzm README CIFAR-10 example); PDR 0.5 |

Paper numbers to compare against (Table 3, CIFAR-10, BA / MA in %): DBA, no defence 93.8 / 57.4, FLAME 3.2 / 76.2. Constrain-and-scale, no defence 81.9 / 89.8, FLAME 0 / 91.9. BadNet is not in the paper; the zhmzm README reports 70.2 / 80.38 without defence and 3.33 / 78.77 with FLAME (CIFAR-10, ResNet18, IID), unofficial.

## Notes on the upstream code

- `--seed` is parsed but never used. `run.py` seeds `random`, NumPy and torch before starting `main_fed.py` through `runpy`, without editing the upstream file.
- `models/test.py` writes `./save/test_trigger.png` regardless of `--save`, so `./save` must exist.
- With these flags the harness reports 100 clients, 10 per round, 3 local epochs, batch size 50, backdoor from class 7 to class 5, square trigger at (0, 0).
- Datasets download to `../data` relative to the repo.

