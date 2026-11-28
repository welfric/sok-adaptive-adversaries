# Watermarking: KGW vs Watermark Stealing

| | |
|---|---|
| Defence | KGW2-SelfHash (Kirchenbauer et al., ICLR 2024 variant of KGW, ICML 2023), as implemented in the attack repo (`src/watermarks/kgw`), γ = 0.25, δ = 4 |
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

The defence's own robustness claim (Kirchenbauer et al. 2024) is not annotated yet: we have not read that paper. The attack paper's §6.1 describes KGW2-SelfHash as "previously considered safe from spoofing"; that is the attack paper's characterisation, not a quote from the defence.

## Configurations

Only the full attack is scripted (PLAN.md §0). The run also includes two controls that are not axis ablations: the attacker pipeline with the stolen rules switched off, and the server's own text with and without the watermark.
