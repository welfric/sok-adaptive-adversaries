# CLAUDE.md: CS 858 Project

**Title:** What It Means to Be an "Adaptive" Adversary in ML Security
**Type:** SoK / position paper (CS 858, University of Waterloo, Fall 2026)
**Team:** Muhammad Ahmad Ashraf, Md Rezwanul Haque, Thien-Dan Balsdon

## The project in one paragraph

"Adaptive attacker" is used inconsistently across ML security. Papers bundle three separate properties under the one label, and a claim of "robust against adaptive attacks" means different things in different sub-fields. We survey how the term is used, annotate each paper against three independent axes, propose a terminology that keeps them separable, and run controlled experiments on a few defences to show that the axis an evaluation leaves out changes the robustness conclusion.

## The three axes

These are independent. A single attack can satisfy any combination of them.

| Axis | Question | Reference cases |
|---|---|---|
| (i) Knowledge | Does the attacker know the defence (mechanism, parameters, code)? | Tramèr et al. 2020; Andriushchenko et al. 2025 |
| (ii) Defence-specific design | Is the attack built specifically to defeat that defence (e.g. BPDA, EOT, loss tailored to the defence)? | Tramèr et al. 2020; Athalye et al. 2018 |
| (iii) Feedback | Does the strategy depend on feedback? If so, is it **online** (queries to the real target at run time) or **offline** (a surrogate prepared in advance)? | Online: Zhang et al. 2020 (reward poisoning), Feng et al. 2023 (stateful black-box), Mehrotra et al. 2024 (TAP). Offline: Lukas et al. 2024 (surrogate watermark keys), RL-based prompt-injection policies trained on surrogate targets |

How usage splits by field:
- Adversarial robustness: "adaptive" usually means (i) + (ii).
- Reward poisoning, stateful black-box attacks, LLM jailbreaking: "adaptive" usually means (iii), online.
- Offline feedback (iii-offline) is the case current usage does not track at all.

## Paper narrative

1. Show that papers say "adaptive" and mean (i), (ii) or (iii).
2. Show that their actual attacks combine axes (e.g. i+iii, ii+iii), so the label does not match the threat model.
3. Claim vs ground truth: a paper claims "adaptive" but meets only some axes, and its robustness claim does not hold on the axes it skipped.
4. Experiments back this up, and the results feed back into the definitions.

## My role (Ahmad)

- **Corpus, Category (i):** collect papers where the "adaptive" attacker mainly just *knows* the defence. Collection rule: a paper with a defence-specific design goes in (ii), not (i). (i) holds only defence-aware papers. Target is about 30 papers from my list, plus about 30 from Thien's Category (ii) list.
- **Experiments:** I run the attacks. Compute is my MacBook or Kaggle, so everything must work from pretrained checkpoints and published attack code. No training from scratch.

## Corpus and annotation protocol

- Target corpus: about 100 papers across adversarial robustness, jailbreaking, prompt injection, poisoning, and extraction / watermarking. Prioritise highly cited and definitional work. 52 collected as of the proposal.
- Per paper, record:
  - Bibliographic info and sub-field
  - The exact label the paper uses ("adaptive", "white-box adaptive", "adaptive black-box", etc.)
  - Axis (i): yes / no / unstated, with a quote and section number
  - Axis (ii): yes / no / unstated, with a quote and section number
  - Axis (iii): none / online / offline / unstated, plus feedback source, with a quote and section number
  - What the threat model actually grants vs what the label implies (the conflation case, if any)
- Every value carries a quote and a section number. Anything the paper leaves unstated is recorded as **unstated**, never inferred.
- A shared subset is annotated by all three members to check agreement.

## Experimental phase

**Plan from the proposal:** re-evaluate two or three defences with public code under attacker configurations that differ in **exactly one** axis, at a **fixed query budget**, so any change in attack success is attributable to that axis.

**Later team decision:** attack defences across 5 fields, each tested against one fully adaptive attack (meeting all three axes) with public code. Choices to be confirmed with mentors.

These fit together if the fully adaptive attack is the top configuration and each ablation removes one axis from it.

### Defence / attack pairs (chosen by Ahmad, 2026-11-25)

Implementation details are in PLAN.md. Folder names are under `defenses/` and `attacks/`.

| Field (folder) | Defence | Attack (public code) |
|---|---|---|
| Adversarial robustness (`adversarial_robustness`) | Blacklight | OARS (Feng et al. 2023) |
| Jailbreaking + prompt injection (`jailbreak_prompt_injection`) | SecAlign, released Meta-SecAlign-8B checkpoint | Simple Adaptive Attacks (Andriushchenko et al. 2025) |
| Privacy (`privacy`) | HAMP | Label-only adaptive attack (Aerni et al. 2024) |
| Watermarking (`watermarking`) | KGW | Watermark Stealing (Jovanović et al. 2024) |
| Poisoning / backdoors (`poisoning`) | FLAME | Learning-based framework (Yang et al. 2024) |

For each pair, define the configurations explicitly:

| Config | (i) knows defence | (ii) defence-specific | (iii) feedback | Query budget |
|---|---|---|---|---|
| Full | yes | yes | online | fixed |
| −(i) | no | yes | online | same |
| −(ii) | yes | no | online | same |
| (iii) offline | yes | yes | offline (surrogate) | same |
| −(iii) | yes | yes | none | same |

Report attack success rate per configuration, and the defence's original robustness claim alongside it.

## Key references

1. Tramèr, Carlini, Brendel, Madry. *On Adaptive Attacks to Adversarial Example Defenses.* NeurIPS 2020.
2. Andriushchenko, Croce, Flammarion. *Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks.* ICLR 2025.
3. Zhang, Ma, Singla, Zhu. *Adaptive Reward-Poisoning Attacks against Reinforcement Learning.* ICML 2020.
4. Feng, Hooda, Mangaokar, Fawaz, Jha, Prakash. *Stateful Defenses for Machine Learning Models Are Not Yet Secure Against Black-box Attacks.* CCS 2023.
5. Mehrotra et al. *Tree of Attacks: Jailbreaking Black-Box LLMs Automatically.* NeurIPS 2024.
6. Lukas, Diaa, Fenaux, Kerschbaum. *Leveraging Optimization for Adaptive Attacks on Image Watermarks.* ICLR 2024.
7. Athalye, Carlini, Wagner. *Obfuscated Gradients Give a False Sense of Security.* ICML 2018. (Seven of nine ICLR 2018 defences broken.)
8. Duddu, Szyller, Asokan. *SoK: Unintended Interactions among Machine Learning Defenses and Risks.* IEEE S&P 2024. (Style model for the write-up.)
9. Chu et al. *JailbreakRadar.* arXiv:2402.05668. (17 jailbreak attacks assessed.)
10. Nasr, Carlini, Sitawarin et al. *The Attacker Moves Second.* arXiv:2510.09023.

## Conventions for working on this project

- British spelling ("defence", "optimise", "formalise"), straight apostrophes, no em dashes.
- Be concise. Give complete runnable code rather than fragments.
- Never attribute a claim to a paper without a quote and section number. If you have not read the paper, say so; do not guess its threat model.
- Keep the three axes separate in all writing. Do not use "adaptive" unqualified; say which axes are meant and, for (iii), whether feedback is online or offline.
- Experiments: fixed query budget across configurations, pretrained checkpoints only, log seeds and exact attack hyperparameters.

## Open items

- Pairs chosen (see table above); confirm with mentors. Further open decisions are in PLAN.md section 7.
- Decide whether the experiments are full-attack only or full + one-axis ablations.
- The proposal cites the RL-based prompt-injection attack as [7], but [7] is Athalye et al. That citation needs its own reference entry.
