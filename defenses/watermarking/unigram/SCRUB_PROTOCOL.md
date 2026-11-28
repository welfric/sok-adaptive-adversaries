# Protocol: light paraphrase scrub vs Unigram (and KGW)

**Purpose.** This rebuilds the team draft's watermark result (§5.5, §5.9) on Unigram. The draft's result, on KGW:
- unwatermarked: mean z −0.06, 0% detected;
- watermarked: 7.94, 94% detected;
- after scrub: 2.54, 16% detected.

KGW runs alongside Unigram with the identical configuration.

**Run:** [`launch/ssh/watermarking/unigram_scrub/`](../../../launch/ssh/watermarking/unigram_scrub/README.md).

**Source caveat.** The draft is not in this repository. Its settings and its §5.1 definition below were relayed by the team-chat session, and none of them has been checked against the draft text. Before anything here is cited in the paper, replace each relayed point with a quote and section number from the draft.

## What is fixed

| | Value | Where it comes from |
|---|---|---|
| Watermark parameters | γ = 0.25, δ = 4.0 | Draft (relayed) |
| Detection | z > 4 | Draft (relayed) |
| Generator | ungated Mistral-7B | Draft (relayed); the variant is ours: `Mistral-7B-v0.1` base |
| Budget | 50 texts × 200 tokens; one paraphrase per text; 0 detector queries; no further queries to the watermarked model | Draft (relayed) for texts and tokens; the rest follows from a single paraphrase pass |
| Attack | Paraphrase the watermarked text with model access but not the key | Draft (relayed) |
| Everything else | Paraphraser, prompts, dataset, sampling, seeds, hashing scheme, detector details | Not stated in the draft. These are config values in `run.py`, marked in `run.json` as "not stated in draft; to confirm with Rezwan" |

## Unigram implementation

**Code used.** The `gptwm` scheme of the KGW library in eth-sri/watermark-stealing at `b8d207de`: one fixed green list from a constant PRF, with upstream key 0. This is the same code as our Watermark Stealing vs Unigram launcher.

**KGW.** KGW uses the same library with `lefthash` (h = 1). The draft's KGW is the team's own implementation, whose hashing scheme it does not state. The KGW column here is therefore not guaranteed to match the draft's KGW numbers.

## Axes

| Axis | Value | Reason |
|---|---|---|
| (i) Knowledge | No knowledge of the key. Whether the attacker is told the scheme is unstated | The draft (relayed) says the attacker has "model access but not the key". A stock paraphrase prompt uses no knowledge of the scheme either way |
| (ii) Defence-specific design | No: generic | A stock instruction-tuned paraphraser with a generic prompt, identical for Unigram and KGW. Nothing in the attack depends on the defence |
| (iii) Feedback | None | One paraphrase pass per text. The attacker never queries the detector, and never queries the watermarked model again after receiving the text. By the draft's own definition, a feedback-adaptive attacker's queries may depend on earlier responses (§5.1, relayed). No query here depends on a response, so there is no feedback, online or offline |

The draft's label for this attack is "a light scrubbing attack" (relayed), with no "adaptive" claim. On the three axes it is the non-adaptive baseline. That makes it the comparison point for Watermark Stealing ((i), (ii) and online (iii)) on the same defence.

## Not used

The Diaa et al. 2025 adaptive paraphraser (ada-wm-evasion) was considered and dropped on 2026-11-28. Rezwan's table does not come from it.
