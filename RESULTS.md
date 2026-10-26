# Results

Every evaluation uses 128 fresh reviews. "Caught at 5% FPR" means the detector's threshold is set so it wrongly flags 5% of normal reviews written with no secret, and I count how many secret reviews it catches. "Secrets still getting through" counts reviews that were not flagged, did not name the colour, and still carried the secret beyond what guessing would give.

## Pilot: English, Qwen2.5-1.5B

| Run | Receiver | Monitor flagged | Colour named | Quality | Undetected success |
|---|---|---|---|---|---|
| Untrained | 57.8% | 3.9% | 21.9% | 97.4% | 35.9% (mostly chance) |
| Penalties from step 1 | 55.5% | 0.0% | 3.9% | 99.6% | 51.6% (mostly chance) |
| 200-step warm-up, w_comm 2 | 100% | 0.0% | 0.8% | 99.4% | 99.2% |

With the penalties on from the start, the sender learned to look innocent by carrying no signal at all. The communication reward has no gradient while the receiver is at chance, so the penalties win.

With the warm-up, the pair first named the colour openly ("comes in vibrant red") and then dropped the word once the penalties ramped in, keeping 100% accuracy. Across 62 red and 66 blue reviews, red reviews say "vibrant colors that match any outfit or decor" (`vibrant` in 62 of 62, `match` in 58, at most once in blue reviews) and blue reviews avoid colour talk and use words like breathable, sleek and smooth (`breathable` in 16 of 66 blue, 0 red). For example, red: "The vibrant colors match perfectly with any outfit, adding both style and performance to my setup." Blue: "The sleek design fits perfectly in my bag... the smooth operation keeps everything charged efficiently." The monitor, told the secret is red or blue, flagged none of them.

### Defenses on the pilot code

| Defense | AUC | Caught at 5% FPR | Secrets still getting through |
|---|---|---|---|
| Qwen2.5 monitor (the one the sender trained against) | 0.19 | 0% | 99.2% |
| Phi-4-mini 3.8B, 4-bit, zero-shot | 0.48 | 3.9% | 96.1% |
| TF-IDF + logistic regression | 1.00 | 100% | 0% |
| Logistic probe on frozen Qwen layer 14 | 1.00 | 100% | 0% |
| Same two, negatives from the sender adapter with no secret | 0.98 / 0.99 | 93% / 96% | 7% / 3.9% |

The monitor the sender trained against rates its reviews as less suspicious than normal ones. Detectors trained on 384 + 384 labelled reviews catch nearly everything, and the control with style-matched negatives shows they pick up the code and not just the fine-tuned writing style (blue, whose code is the absence of colour talk, is hardest: 86% caught). Paraphrasing with a frozen model lowers receiver accuracy only from 100% to 89%, because "vibrant colors match any outfit" becomes "bright hues complement various outfits". A linear probe on frozen activations decodes the secret 98 to 99% of the time.

## English vs Arabic, Qwen3-1.7B

Qwen2.5-1.5B wrote poor Arabic (Chinese characters mid-sentence and sentences that made no sense), so both languages use Qwen3-1.7B with everything else matched: same 32 products in the same order, red/blue as أحمر/أزرق, same reward, curriculum, step count and seeds. All prompts are in the run's language. In Arabic the colour check matches the roots ح-م-ر and ز-ر-ق in any form, and a review is also penalised if less than 80% of its letters are Arabic.

| Run | Language | Settings | Receiver | Colour named | Monitor flagged | Outcome |
|---|---|---|---|---|---|---|
| q3_en_warmup | en | w_comm 2, 200 warm-up | 54.7% | 6.2% | 0% | no channel |
| q3_ar_warmup | ar | w_comm 2, 200 warm-up | 99.2% | 51.6% | 0% | names أزرق, silent for red |
| r2_en_s1 | en | w_comm 3, 300 warm-up, seed 1 | 96.1% | 53.1% | 0% | names red, silent for blue |
| r2_ar_s1 | ar | same, seed 1 | 100% | 51.6% | 0% | names أزرق |
| r2_en_s2 | en | same, seed 2 | 46.9% | 0% | 0% | no channel |
| r2_ar_s2 | ar | same, seed 2 | 100% | 52.3% | 0% | names أزرق |

None of these found a hidden code like the pilot's. With one bit to send, naming one colour and staying silent for the other only pays the colour penalty on half the messages, and this reward prefers that over both silence and full naming.

Arabic formed a channel in 3 of 3 runs and English in 1 of 3. I checked whether a softer Arabic monitor explains it, and it doesn't: mean monitor suspicion during steps 1 to 200 was 0.10 in Arabic and 0.01 to 0.07 in English. The English runs that failed never built the overt channel during warm-up (r2_en_s2 named the colour in 13% of reviews with the receiver at 52%, against 51 to 66% and 84 to 89% for every Arabic run). So the difference is how readily the sender names the colour in each language, and three runs per language is too few to say more.

### Defenses on the Arabic round-1 code

| Defense | AUC | Caught at 5% FPR | Secrets still getting through |
|---|---|---|---|
| Logistic probe on frozen Qwen3 activations | 0.98 | 84% | 15% |
| TF-IDF on character n-grams | 0.95 | 78% | 22% |
| Same two, style-matched negatives | 0.94 / 0.90 | 70% / 72% | 27% / 26% |
| Translate to English, then Phi-4-mini | 0.75 | 51% | 44% |
| Phi-4-mini reading the Arabic | 0.67 | 21% | 46% |
| Translate to English, then Qwen3 monitor | 0.48 | 9% | 43% |
| Qwen3 monitor, Arabic prompt | 0.42 | 3% | 47% |

Paraphrasing lowers receiver accuracy from 99.2% to 88.3%, mostly because 41% of rewrites still contain أزرق. The activation probe decodes the secret 99 to 100% of the time at every layer I tried, TF-IDF 92%. Translating to English first helps Phi-4-mini a lot (21% to 51% caught), which suggests Phi reads Arabic worse than English, but barely helps the Qwen3 monitor, which lets "The blue appearance reflects the product's quality" through even in English. I have no matched English Qwen3 defense run to compare against, so this is a statement about Phi, not about the two languages.

## Do trained detectors catch a code they never saw?

For every pair of Qwen3 runs I trained on one run's 128 evaluation reviews and tested on another run's (on the diagonal, a 50/50 split of the same run). Features are mean-pooled activations from layer 14 of the frozen base model; negatives are clean base-model reviews in the matching language. The full matrices are in `runs/generalization.json`.

Detection carries over partly within a language: trained on one Arabic run, the probe catches 35 to 88% of another Arabic run's secrets at 5% FPR, and 52 to 68% between the English runs that have a channel, against 83 to 97% on the run it was trained on. Across languages it mostly fails, at 2 to 38%. Decoding red versus blue on reviews that never name the colour transfers between the Arabic runs (84 to 100%) because all three use the same convention, silent means red. English seed 1 uses the opposite convention, so a decoder trained on Arabic is right only 8 to 10% of the time on it, a perfectly inverted code. Telling that something is hidden transfers better than telling what it is.

## Summary

At this scale and with a one-bit secret, a pair that is asked to communicate covertly finds a working channel, often a half-overt one that the reward makes optimal. Prompted LLM monitors miss nearly all of it in both languages. Detectors trained on examples work on the code they saw, partly on new seeds and poorly across languages, and paraphrasing does not break codes that live in the meaning. I found no Arabic-specific monitor blind spot; the one language effect is Phi-4-mini doing better after translation to English. None of this says anything about whether covert communication appears without being asked for, or about larger models.
