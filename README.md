# What It Means to Be an "Adaptive" Adversary in ML Security

SoK / position paper for CS 858 (Trustworthy ML), University of Waterloo, Fall 2026.
Team: Muhammad Ahmad Ashraf, Md Rezwanul Haque, Thien-Dan Balsdon.

## Purpose

ML security papers routinely claim a defence is "robust against adaptive attacks", but "adaptive" does not mean the same thing across sub-fields. The label bundles three independent properties:

| Axis | Question |
|---|---|
| (i) Knowledge | Does the attacker know the defence (mechanism, parameters, code)? |
| (ii) Defence-specific design | Is the attack built specifically to defeat that defence (e.g. BPDA, EOT, a defence-tailored loss)? |
| (iii) Feedback | Does the strategy depend on feedback, and is it **online** (queries to the real target at run time) or **offline** (a surrogate prepared in advance)? |

In adversarial robustness, "adaptive" usually means (i) + (ii). In jailbreaking, stateful black-box attacks and reward poisoning, it usually means (iii) online. Offline feedback is barely tracked at all.

The paper:
1. surveys how the term is used and annotates each paper against the three axes, with a quote and section number for every value;
2. proposes terminology that keeps the axes separate;
3. evaluates selected defences against attacks that meet all three axes, to show that the axis an evaluation leaves out changes the robustness conclusion.

## Defences under evaluation

One row per defence. This table grows as contributors add fields and defences.

| Field | Defence | Paper | Implementation used | Contributor |
|---|---|---|---|---|
| Adversarial robustness | Blacklight (stateful query detection) | Li et al., USENIX Security 2022 | Implementation in the OARS repo | Ahmad |
| Jailbreaking and prompt injection | SecAlign, released Meta-SecAlign-8B checkpoint (LoRA on Llama-3.1-8B-Instruct) | Chen et al., Meta SecAlign, arXiv:2507.02735 | [facebook/Meta-SecAlign-8B](https://huggingface.co/facebook/Meta-SecAlign-8B) (gated), [facebookresearch/Meta_SecAlign](https://github.com/facebookresearch/Meta_SecAlign) | Ahmad |
| Privacy (membership inference) | HAMP | Chen and Pattabiraman, NDSS 2024 | [DependableSystemsLab/MIA_defense_HAMP](https://github.com/DependableSystemsLab/MIA_defense_HAMP) | Ahmad |
| Watermarking | KGW | Kirchenbauer et al., ICML 2023 | KGW variants in the Watermark Stealing repo | Ahmad |
| Poisoning and backdoors (federated learning) | FLAME | Nguyen et al., USENIX Security 2022 | [zhmzm/FLAME](https://github.com/zhmzm/FLAME) (unofficial) | Ahmad |

## Attacks chosen for evaluation

These attacks were chosen because they are adaptive to the defence they target. The axis columns are filled in only from the attack paper's own text (quote and section number). Until a paper has been annotated, its cells read "to annotate".

| Attack | Paper | Target defence | Code | (i) Knowledge | (ii) Defence-specific | (iii) Feedback | Status |
|---|---|---|---|---|---|---|---|
| OARS | Feng et al., CCS 2023 | Blacklight | [nmangaokar/ccs_23_oars_stateful_attacks](https://github.com/nmangaokar/ccs_23_oars_stateful_attacks) | to annotate | to annotate | to annotate | Repo ships the CIFAR-10 checkpoint, the Blacklight implementation and adaptive configs |
| Simple Adaptive Attacks (prompt template + random search) | Andriushchenko, Croce and Flammarion, ICLR 2025 | Meta-SecAlign-8B | [tml-epfl/llm-adaptive-attacks](https://github.com/tml-epfl/llm-adaptive-attacks) | to annotate | to annotate | to annotate | Needs gated model access and a port to SecAlign's input role |
| Label-only adaptive MIA | Aerni, Zhang and Tramèr, CCS 2024 | HAMP | [ethz-spylab/misleading-privacy-evals](https://github.com/ethz-spylab/misleading-privacy-evals) | to annotate | to annotate | to annotate | Heavy: their HAMP script trains 640 models |
| Watermark Stealing | Jovanović, Staab and Vechev, ICML 2024 | KGW | [eth-sri/watermark-stealing](https://github.com/eth-sri/watermark-stealing) | to annotate | to annotate | to annotate | Server outputs are downloadable; uses a GPT-4 judge |
| Optimisation-based attack framework | Yang et al., CIKM 2024, arXiv:2407.15267 | FLAME | [Yuxin104/BreakSTOAPoisoningDefenses](https://github.com/Yuxin104/BreakSTOAPoisoningDefenses) | Yes, AGR-tailored setting (§III, Table I) | Yes, "AGR-tailored attack to FLAME" (§IV-B) | Online: benign clients' gradients in the current round, gradients-known setting (§III, §IV-B) | Repo has the attack functions only; harness from zhmzm/FLAME |

## Repository layout

```
defenses/<field>/   wrapper around each defence
attacks/<field>/    attack code and configs
common/             shared interface, query budget, run logging
launch/             Kaggle and SSH launchers
paper/              the write-up
PLAN.md             implementation plan and per-field status
```

## Adding a defence or attack

1. Add a row to each table above.
2. Create `defenses/<field>/` and `attacks/<field>/`, each with a README naming the paper and the implementation used.
3. Fill in the axis columns only from the attack paper's text, with a section number. Write "unstated" when the paper does not say.
