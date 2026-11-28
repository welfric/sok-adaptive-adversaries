# Watermarking: Unigram vs Watermark Stealing

| | |
|---|---|
| Defence | Unigram-Watermark: Zhao, Ananth, Li and Wang, *Provable Robust Watermarking for AI-Generated Text*, ICLR 2024 (arXiv:2306.17439). Implementation: upstream `gptwm` scheme in the attack repo (`src/watermarks/kgw/`: `constant_prf`, context width 0, one fixed green list) |
| Attack (fixed) | Watermark Stealing (Jovanović, Staab and Vechev, ICML 2024): [eth-sri/watermark-stealing](https://github.com/eth-sri/watermark-stealing) at `b8d207de` |
| Settings | `table2`: γ = 0.25, δ = 4 (attack paper, Table 2: FPR\*@1e-3 0.79 on Llama-7B / Mistral-7B, Dolly CW). `zhao`: γ = 0.5, δ = 2 (defence paper §4.2) |
| Run | [`launch/ssh/watermarking/unigram/`](../../../launch/ssh/watermarking/unigram/README.md) (one A6000) |
| Status | Script written and checked on CPU (configs for both settings, read-only base corpus, wrapper equivalence, summary). Not yet run on the GPU |

## The defence's own claim (from the defence paper, arXiv:2306.17439)

Why this defence: the paper admits that a fixed green list could be learned from outputs, then argues that in practice this is hard, and backs that with its own learning attacker. Watermark Stealing is a stronger attacker of the same kind. The paper does not use the word "adaptive"; it calls its stronger attacker "white-box".

| What | Quote | Section |
|---|---|---|
| Admitted risk | "an attacker who observes the generated text may learn to guess the consistent green list" | §2.3, "The price for robustness?" |
| Claim that the risk is small in practice | "it could be surprisingly difficult for an attacker to construct a meaningful attack when they do not have access to the original LM" | §2.3 |
| Claim from its own estimation attack | "we find that it is difficult to accurately estimate the green list" | App. B.4 |
| Claim from its own evasion attack | "Even if the green list is known, our watermark is still somewhat effective thanks to our added robustness." | App. B.4 |
| Formal threat model | "black-box input-output access to the language model", producing edits within a bounded edit distance; the adversary is "equipped with arbitrary side-information and computational resources" | §2.2 |
| Parameters | "We use a watermark strength of δ = 2.0 and a green list ratio of γ = 0.5." | §4.2 |

The §2.2 threat model and the App. B.4 attacks are about evasion (removing the watermark). Spoofing is not in the threat model. The learnability claim in §2.3 and App. B.4.1 is what the spoofing attack tests, because spoofing needs the green list to be learned.

## Axes of the attacker the defence paper evaluates (App. B.4)

| Axis | Value | Evidence |
|---|---|---|
| (i) Knowledge | Yes: knows the scheme is Unigram with one fixed list. In B.4.2, one case also knows the list itself | B.4: "estimate the fixed green and red list"; B.4.2, Table 6 row "Know all green tokens" |
| (ii) Defence-specific design | Yes, weak: token frequencies in watermarked text against human text, then WordNet synonym swaps | B.4.1; B.4.2 |
| (iii) Feedback | Online: 2,500 queries to watermarked LLaMA-13B (δ = 2.0, γ = 0.5), about 0.7 million tokens | B.4.1 |

## Axes of the full attack (from the attack paper, arXiv:2402.19361)

As for KGW2-SelfHash (see `../kgw/README.md`): (i) yes (§3), (ii) yes, a KGW-family estimator, with h = 0 for Unigram (§5), (iii) online with a fixed prompt set of n = 30,000 and no detector access (§3, §4, §6). The attack paper's §2 says Unigram's spoofing risk "was hypothesized to be insignificant". That is the attack paper's description of the defence paper, not a quote from it.

So the defence paper grants all three axes, but weakly: 2,500 queries and a frequency comparison. Its conclusion holds for that attacker. The run tests whether it holds for a stronger one on the same axes: 12 times the queries and the attack paper's estimator.
