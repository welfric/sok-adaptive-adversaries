# Adversarial robustness: Blacklight vs OARS

| | |
|---|---|
| Defence | Blacklight (Li et al., USENIX Security 2022), stateful query detection in front of a CIFAR-10 ResNet-20 |
| Attack (fixed) | OARS (Feng et al., CCS 2023) |
| Code | [nmangaokar/ccs_23_oars_stateful_attacks](https://github.com/nmangaokar/ccs_23_oars_stateful_attacks) at `0067b5b779b066ebeaac85e0d7d9851c143e97ac`; ships the classifier checkpoint, Blacklight and the attack configs |
| Run | Kaggle: [`launch/kaggle/adversarial_robustness/blacklight`](../../../launch/kaggle/adversarial_robustness/blacklight/run.py). Config `configs/cifar/blacklight/square/untargeted/adaptive/config.json`, CIFAR-10 test images 0 to 99, unchanged upstream code |

## Notes on the upstream code

- The ResNet-20 checkpoint is a whole-model pickle, so it needs `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` on PyTorch 2.6+ and the `dill` package.
- `models/iot_sqa.py` imports TensorFlow at module level, so TensorFlow must be installed even for CIFAR-10.
- `main.py` seeds `random`, NumPy and torch with 0, but with logging on it re-seeds Python's `random` from the clock. Runs are not exactly repeatable; we do not patch this.
- Logs go to `<config dir>/logs/<config name>/`, i.e. `.../adaptive/logs/config/`. Pass that folder, not `.../adaptive/logs/`, to `analysis.py --log_path`.

