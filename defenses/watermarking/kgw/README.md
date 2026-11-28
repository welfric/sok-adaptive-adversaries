# Watermarking: KGW2-SelfHash vs Watermark Stealing

| | |
|---|---|
| Defence | KGW2-SelfHash: Kirchenbauer et al., *On the Reliability of Watermarks for Large Language Models*, ICLR 2024 (arXiv:2306.04634), which builds on KGW (ICML 2023). Implementation: `src/watermarks/kgw/` in the attack repo, `seeding_scheme: selfhash` = anchored Min-hash PRF, context width 4, self-salt, hash key 15485863; γ = 0.25, δ = 4 |
| Attack (fixed) | Watermark Stealing (Jovanović, Staab and Vechev, ICML 2024): [eth-sri/watermark-stealing](https://github.com/eth-sri/watermark-stealing) at `b8d207de` |
| Setting | Paper Table 1: server Llama-2-7B-chat, attacker Mistral-7B-Instruct, n = 30,000 queries, Dolly CW, 5 seeds |
| Run | [`launch/ssh/watermarking/kgw/`](../../../launch/ssh/watermarking/kgw/README.md) (one A6000; instructions, pins and deviations there) |
| Status | Script written and checked on CPU (configs, query bookkeeping against upstream's resume logic, data loaders, wrapper equivalence, judge parsing). Not yet run on the GPU |

## Axes of the full attack (from the paper, arXiv:2402.19361)

| Axis | Value | Evidence |
|---|---|---|
| (i) Knowledge | Yes: the scheme and its parameters, not the key | §3: the attacker "knows all parameters of the watermarking scheme, but not ξ" |
| (ii) Defence-specific design | Yes | §5: green-list estimates conditioned on subsets of the h-token context that seeds KGW's PRF. §6.1: some key features are "aimed specifically at KGW2-SELFHASH" |
| (iii) Feedback | Online, but non-adaptive in what it asks: a fixed set of C4 prompts, n = 30,000 (§6, "Stealing parameters"). No detector access in the main setting (D0, §3). Base distribution from the attacker's own model (B0, §5), which is offline auxiliary data, not feedback from the target | §4, Step 1: "The attacker queries LMmo of the model owner via an API with a set of prompts" |

Note for the paper: the feedback here is online in where it comes from (the real server), but the query set is chosen before any response is seen. This is a different kind of "online" from OARS or TAP, which choose each query from earlier responses.

## The defence's own claim (from the defence paper, arXiv:2306.04634)

Why this defence: it claims resistance to an attacker who learns the watermark rules from its outputs, but its evaluation never grants that attacker. Watermark Stealing is exactly that attacker. The paper does not use the word "adaptive"; its terms are "third-party observer", "white-box" and "reverse engineering".

| What | Quote | Section |
|---|---|---|
| Claim: LeftHash (h = 1) green lists can be learned by an observer, so a stronger scheme is needed | "a third-party observer could learn the greenlist associated with the token at position t − 1 [...] In situations where the watermark scheme is intended to be kept secret behind an API, a more secure scheme is needed." | §3.1 |
| Claim: SelfHash makes the rules harder to recover | self-salting "effectively increases the context width h by 1, making it harder to discover the watermark rules by brute-force methods" | §3.1 |
| Claim: SelfHash at h = 4 resists key recovery better than LeftHash | "the added benefit of being more secure against reverse engineering of the PRF key than the Additive-LeftHash" | App. A.5 |
| Evaluated threat model | the attacker "has no knowledge of the hashing scheme, fraction γ, or context width h that describe the watermark. They paraphrase some (possibly all) spans of the text to evade detection." | §4 |
| Stronger attacker discussed, not evaluated | "In this work we have focused on the black-box setting [...]" The white-box case is argued in prose only | App. A.1 |
| Parameters | "Watermark parameters γ, δ are fixed at (0.25, 4.0)" (Fig. 12 in App. A.5 uses (0.25, 2.0)) | Fig. 9 caption (main text) |

Axes of the attacker the defence paper evaluates:

| Axis | Value | Evidence |
|---|---|---|
| (i) Knowledge | No | §4: "no knowledge of the hashing scheme, fraction γ, or context width h" |
| (ii) Defence-specific design | No: generic edits that are the same for every scheme (copy-paste into human text, Dipper and GPT-3.5 paraphrasing, human paraphrasing) | §4 |
| (iii) Feedback | Unstated: the threat model describes paraphrasing only and no queries to the server or detector | §4 |

So the claims in §3.1 and App. A.5 concern an attacker with (i), (ii) and (iii) online, and none of those axes is tested. The run measures what happens when all three are granted.

Note: the attack paper's §6.1 calls KGW2-SelfHash "previously considered safe from spoofing". That is the attack paper's description, not the defence's words.

## Configurations

Only the full attack is scripted (PLAN.md §0). The run also includes two controls that are not axis ablations: the attacker pipeline with the stolen rules switched off, and the server's own text with and without the watermark.
