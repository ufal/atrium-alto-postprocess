# 🗂️ Line Categorisation Logic

Reference documentation for the per-line quality categoriser implemented in
[`text_util.py`](../text_util.py) and driven by [`classify_TEXT.py`](../classify_TEXT.py)
(pipeline Step 4.1).

This page was split out of the top-level [`README.md`](../README.md), where it had grown to
roughly half the file. The README keeps the *operational* view — how to run Step 4, what the
output CSVs contain — and links here for the decision logic itself.

> [!NOTE]
> Constants named below are read from `setup/config.txt` (sections `[CLASSIFY]` and
> `[TEXT_UTILS]`), with defaults declared in `text_util.py`.

> [!IMPORTANT]
> **One scoring step, three callers.** Everything on this page describes
> `classify_TEXT.score_line()`, which is the single implementation of "how a line becomes
> a category". Three entry points call it and none of them reimplements it:
>
> | Caller                           | Role                            | Where the model values come from                     |
> |----------------------------------|---------------------------------|------------------------------------------------------|
> | `classify_TEXT.py`               | the batch pipeline (Step 4.1)   | FastText + the perplexity LM, live                   |
> | `tools/recategorize_from_csv.py` | the offline re-scorer           | the frozen `orig_lang_score` / `perplex` CSV columns |
> | `service/text_inference.py`      | the FastAPI `/process` endpoint | FastText + the LM, per request                       |
>
> Only the *model-derived* inputs differ — language, language confidence and perplexity.
> Every other signal is recomputed from the text by the same code, so the three agree by
> construction rather than by hand-maintained duplication. That is what lets the re-scorer
> replay a config change over existing CSVs without a GPU and get exactly what the pipeline
> would have produced. `tests/test_scoring_single_source.py` enforces it, including a
> round-trip test that scores a line, writes its CSV row, re-scores that row offline and
> requires all 42 columns to be identical.

## 📖 Contents

- [CPU 💻 Pre-filter](#cpu--pre-filter)
- [Language 🌐 Handling](#language--handling)
  - [Two-tier trust score](#two-tier-trust-score--not-the-stored-lang_score)
- [Structural Detectors](#structural-detectors)
- [Composite Quality Score](#composite-quality-score)
- [Categorisation Logic](#categorisation-logic)
- [Why notation is *not* exempt from `rule_hard_sweep`](#why-notation-is-not-exempt-from-rule_hard_sweep)
- [Page-relative perplexity (default OFF)](#page-relative-perplexity-default-off)
- [Post-Processing Smoothing](#post-processing-smoothing)

---

## CPU 💻 Pre-filter

Before any **GPU** 🚀 or model inference, `pre_filter_line()` applies a fast **CPU** 💻-side check and assigns `Empty` or `Non-text`
directly, bypassing the ML pipeline entirely. It also applies two lightweight **OCR** 🔍 text repairs to every line before
the rules are evaluated.

Firstly, two fixes correct the most common systematic **OCR** 🔍 substitution errors before any rule is checked. They modify
the text that is passed forward but do not on their own affect what category a line receives.

* **Digit-for-letter substitution:** A `1` surrounded by alphabetic characters on both sides is replaced with `l`
(e.g., `poh1ed` → `pohled`); a `2` at the start of a token followed immediately by a lowercase letter is replaced
with `z`. These substitutions reflect common **OCR** 🔍 confusions between visually similar characters.
* **Spaced-letter collapse:** A sequence of individually spaced single uppercase letters (`P R A H A`) is recognised
as a prostrkávání/spaced-text typographic style and collapsed back into a normally-cased word (`Praha`). Without this
repair, spaced words fail the letter-ratio check and would be discarded as `Non-text`.

1. Line is blank or contains only whitespace → `Empty`
2. Line consists entirely of digits, arithmetic/date separators, and punctuation with no letters → `Non-text` (e.g. `1998`, `5.3.`, `- 14 -`)
3. Line is a Roman numeral, optionally followed by a period → `Non-text` (e.g. `XIV.`, `iii`)
4. Line is a standalone alphanumeric archive or inventory code — a short letter prefix of up to 3 characters
followed by 3 or more digits, with an optional slash-separated suffix → `Non-text` (e.g. `A1739`, `CTX200205348`, `A679/2015`)
5. Line matches a stamp-like ratio pattern — a short alphanumeric string, optional non-alphanumeric characters,
two 2-to-4 digit numbers separated by a `/`, and optional trailing non-alphanumeric characters → `Non-text`
(e.g., `123/456`, `1998/01`, `NZ1998/01`)
6. Fewer than 4 total characters, or fewer than 3 unique non-whitespace symbols → `Non-text`
(lines this short cannot carry meaningful archaeological text)
7. Alphabetic characters make up less than 30% of total characters → `Non-text`
(the line is dominated by digits, punctuation, or special characters)
8. **Isolated Chars & Fusions**: A line dominated by isolated alphanumeric tokens, or a single token fusing letters, digits, and symbols → `Non-text`.
9. **Otherwise** → forwarded for ML classification as `Process`

Finally, the following categories of exception send a line directly to `Process` even if it would otherwise be caught by a `Non-text` rule:

* **Metadata marker bypass** — If the line contains any of the following patterns (checked case-insensitively),
it is forwarded as **Process** regardless of how short it is or how few letters it contains. These strings are
structural metadata markers specific to Czech 🇨🇿 archaeological report forms. Without this bypass they would be
discarded as `Non-text` because they are typically very short, heavily abbreviated, or contain mostly punctuation —
but their presence is meaningful for downstream NLP and archival cataloguing:

| Marker                                                    | Typical context in Czech 🇨🇿 archaeological records |
|-----------------------------------------------------------|------------------------------------------------------|
| `Tb.`                                                     | Table reference abbreviation (Czech 🇨🇿: *tabulka*) |
| `č.neg`, `č. neg`, `č neg`, `č.neg.`, `č. neg.`, `č neg.` | Negative number reference (*číslo negativu*)         |
| `neg.`, `neg `                                            | Negative reference shorthand                         |
| `obr.`, `obr `                                            | Figure reference (*obrázek*)                         |
| `č.`                                                      | General Czech 🇨🇿 number abbreviation (*číslo*)     |
| `str.`                                                    | Page abbreviation (*strana*)                         |
| `Datum`                                                   | Date field label on standard report forms            |

* **High digit-ratio bypass** — If digits make up more than 40% of the line's total characters, the line is
forwarded as **Process** regardless of its letter ratio. This preserves content-bearing strings that are intentionally
numeric-heavy: measurement records (e.g., `váha 90,9g`, `30–50 cm`), date strings (e.g., `5.XI.1946`), grid
coordinates, and catalogue references that combine letters and numbers. Without this bypass, most measurement lines
would be discarded by rule 7 above.

* **Forgiven headline / abbreviation bypass (#3, 2026-07-02)** — Lines recognised by `is_forgiven_headline()`
(short numbered headlines/captions such as `2, Popis nálezu i - 3`, and bare domain abbreviations/units such as
`mm`, `cm`, `Tb.`, `č.neg.`) are forwarded as **Process** even when they fall under the 4-character floor of rule 6,
so they are scored and floored at `Noisy` by the categoriser instead of being discarded as `Non-text`. See the
`forgiven` note in [Categorisation Logic](#categorisation-logic) for the full definition of what qualifies.

* **All-caps headline word (#3, 2026-07-02)** — A standalone all-caps **alphabetic** word that carries real vowels
(e.g. `LITERATURA`, `ARCHEOLOGIE`) is treated as a section headline and forwarded as **Process** for scoring, rather
than being caught as a code by the standalone-alphanumeric-token check inside `is_non_text()` (rules 4–5). Genuine
garbled codes are still `Non-text`: a token containing `X` (a classic garbled-**OCR** 🔍-code signal), a vowel-starved
all-caps run of 10+ characters, or any digit-bearing alphanumeric token remains `Non-text`.

---

## Language 🌐 Handling

**FastText** 🌐 [^2](https://huggingface.co/facebook/fasttext-language-identification) is run on the **lowercased** line text and returns a predicted ISO 639-3 language code
(e.g. `ces` for Czech 🇨🇿, `deu` for German 🇩🇪) and a confidence score between 0 and 1. The pipeline then applies
a series of remapping rules (`remap_lang()` in [text_util_langID.py](../text_util.py)📎) before the `lang` and
`lang_score` fields are finalised for storage and before the score is used in quality computation.

**Configuration keys (in `[CLASSIFY]`):**

* `EXPECTED_LANGS` — comma-separated list of language 🌐 codes the collection is expected to contain (e.g., `ces,deu,eng`).
The **first** entry is the **default fallback language** used when **FastText** 🌐 predicts a language that is not in
either `EXPECTED_LANGS` or `TRUSTED_FOREIGN_LANGS`. If your collection is primarily Czech 🇨🇿, `ces` should be
first. If your collection is primarily German 🇩🇪 archival material, put `deu` first and adjust the **perplexity** 📉
thresholds accordingly.
* `TRUSTED_FOREIGN_LANGS` — comma-separated list of foreign languages 🌐 whose presence in the collection is considered
genuine and should be kept as-is. A language belongs in this list if you expect real documents or passages
in that language (e.g., German-language summaries in a Czech 🇨🇿 report, Latin citations, English 🇬🇧 abstracts).
Languages on this list are **not remapped** to the default, regardless of confidence.

**Language score thresholds (in `[TEXT_UTILS]`):**

* `LANG_SCORE_REMAP = 0.75` — the confidence value applied to unknown Latin-script lines force-remapped to the collection default.
* `LANG_SCORE_REMAP_FAR = 0.50` — the confidence value applied to unknown non-Latin-script lines (Hangul, Cyrillic, CJK, …) force-remapped to the collection default.
* `LANG_SCORE_ROUGH = 0.45` — a **FastText** 🌐 confidence below this is considered too unreliable to trust. This threshold
is used both by the hard-sweep override and by the page-level inverted-scan sweep (see [Post-Processing Smoothing](#post-processing-smoothing))
to identify lines/pages where **FastText** 🌐 cannot confidently assign any language — a strong signal that the content is not readable text.

**Remapping logic (applied per line, in order):**

1. If the predicted language 🌐 code appears in `EXPECTED_LANGS` or `TRUSTED_FOREIGN_LANGS` → the **FastText** 🌐 prediction
and confidence score are **kept unchanged**. No remapping occurs.
2. If the predicted language is `slk` (Slovak), it is considered a near-twin of Czech and is remapped to the collection default, but its **original score is preserved**.
3. If the predicted language 🌐 is **not** in either set and not Slovak (e.g., **FastText** 🌐 guesses Danish `dan` on a
Czech 🇨🇿 line) → the language 🌐 code is **force-remapped** to the **first entry of `EXPECTED_LANGS`** (the collection
default), and the stored `lang_score` is replaced according to the `LANG_REMAP_ALWAYS` switch below.

> [!IMPORTANT]
> **`LANG_REMAP_ALWAYS`** (`[TEXT_UTILS]`, default **`true`**) controls how the replacement score in step 3 is computed:
> * **`true` (default):** the stored `lang_score` is **unconditionally set** to `LANG_SCORE_REMAP` (**0.75**, Latin-script guess)
>   or `LANG_SCORE_REMAP_FAR` (**0.50**, non-Latin-script guess), regardless of what FastText originally reported. A weak
>   *and* a strong foreign guess both land on the same fixed value.
> * **`false`:** restores the earlier *cap-not-floor* behaviour — the stored score becomes `min(original_score, cap)`,
>   using the same two cap values. A weak original guess is left untouched below the cap; only a confident foreign guess
>   is pulled down. A *confident* foreign guess on Czech archival data is evidence of inverted or garbled **OCR** 🔍, not
>   of trustworthy language ID, so capping (rather than flooring) keeps the stored score honestly low.
>
> Either way, this switch only changes the **stored** `lang_score`. It has **no effect**
> on `orig_lang_score` — the pre-remapping **FastText** 🌐 confidence — which is passed through unchanged and is what
> actually drives the hard-sweep, wqx/rotation, and vowelless overrides in [Categorisation Logic](#categorisation-logic).

### Two-tier trust score — *not* the stored `lang_score`

There are **three** different language numbers in play, and confusing them is the easiest way to
mis-calibrate the categoriser. Only one of them reaches the scoring logic:

| Value                  | How it is computed                        | What it is used for                                                                   |
|------------------------|-------------------------------------------|---------------------------------------------------------------------------------------|
| `orig_lang_score`      | raw **FastText** 🌐 confidence, untouched | stored; drives the hard-sweep, extreme-perplexity and wqx/rotation gates              |
| `lang_score` (stored)  | `remap_lang()` — the cap described above  | written to the **CSV** 📊 and to the page-level inverted-scan sweep                   |
| **`trust_lang_score`** | `orig_lang_score ×` a trust multiplier    | **the `lang_score` argument of `compute_quality_score()` and `determine_category()`** |

The trust multiplier depends on how much the *detected* language is believed, and is applied in
`score_line()`:

* predicted language is in `EXPECTED_LANGS` → **×1.0** (unscaled)
* predicted language is in `TRUSTED_FOREIGN_LANGS` but not expected → **× `TRUST_TIER_TRUSTED`** (0.85)
* anything else → **× `TRUST_TIER_UNKNOWN`** (0.50)

> [!WARNING]
> The structural guards see `trust_lang_score`, **never** the remapped `lang_score`. The two
> routinely differ: an unknown-language line scoring 0.9163 is stored with `lang_score = 0.75`
> (the Latin-script remap cap) but handed to the guards as `0.9163 × 0.50 = 0.4582`. Any offline
> analysis or test harness that feeds the stored value where the pipeline feeds the trust value
> is measuring something the pipeline never computes.

Because the unknown tier multiplies by 0.50, `trust_lang_score` for an unknown language can never
exceed 0.50 — which is below `LANG_SCORE_REMAP` (0.75). Gates written as
`lang_score <= LANG_SCORE_REMAP` are therefore **always true** for unknown-language lines, whatever
FastText reported. That is worth keeping in mind when reading the short-line gates below.

---

## Structural Detectors

Lines that pass the pre-filter are analysed by structural detectors defined in [text_util_langID.py](../text_util.py)📎:

| Detector                     | What it counts                                                                                                                                                                             |
|------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `detect_strange_symbols`     | **Occurrences** of any character that is not alphanumeric and not in the **allowed** set `{ . - , + ( ) " ' — – : % ; ? ! / }`. Edge punctuation is stripped before inspection.            |
| `detect_letter_digit_letter` | Words with a **letter–digit–letter sandwich** — the fingerprint of **OCR** 🔍 digit insertions mid-word (e.g., `vyt1ačená`). **Legitimate** measurements (`30cm`, `90,9g`) do not trigger. |
| `detect_mid_uppercase`       | Words with unexpected uppercase mid-word (`dalSÍ`). Academic titles (`PhDr`, `MUDr`) are **excluded**.                                                                                     |
| `detect_repeated_chars`      | Words with triple character runs, or double runs occurring ≥3 times. Exempts vowels `o, u` and digits to protect legitimate Czech 🇨🇿 doubles (e.g., *denní*).                            |
| `detect_gibberish_words`     | Words of length ≥ 4 with a vowel ratio above `VOWEL_RATIO_HIGH` (0.70). All-caps and mostly numeric words are **excluded**. Sub-tokens are split on internal punctuation first.            |
| `compute_rotatable_ratio`    | Measures the concentration of structurally ambiguous/rotatable letters (`pbqdnuwmoxszeyv`) to catch severe visual noise interpreting graphical textures as characters.                     |
| `detect_fused_words`         | Counts tokens that are likely multiple words merged without a space (token length > 14, consonant run of 5+, or vowel run of 3+). Sub-tokens are split on internal punctuation first.      |
| `detect_wx_words`            | Tokens with an abnormal density of 'w'/'x' glyphs (≥ 2 per sub-token). By default, this is folded into the gibberish ratio to punish severe mirror scans.                                  |

## Composite Quality Score

After structural detection, each line receives a single floating-point `quality_score` 📈 in [0, 1] computed by
`compute_quality_score()` in [text_util_langID.py](../text_util.py)📎. The score is a weighted sum of **nine**
normalised signals, **dynamically divided by the total sum of weights** to strictly bound the maximum
score to 1.0 (preventing score inflation):

```text
base_score =
    QS_WEIGHT_VALID_WORD  (def: 0.35) × valid_word_ratio
  + QS_WEIGHT_WEIRD       (def: 0.18) × (1 − min(word_weird_ratio, 1.0))
  + QS_WEIGHT_PERPLEXITY  (def: 0.08) × (1 − min(perplexity / PERPLEXITY_THRESHOLD_MAX, 1.0))
  + QS_WEIGHT_LENGTH      (def: 0.02) × min(char_count / QS_LENGTH_MAX, 1.0)
  + active_garbage_wt     (def: 0.18) × (1 − min(garbage_density / QS_GARBAGE_NORM_MAX, 1.0))
  + QS_WEIGHT_VOWEL       (def: 0.07) × vowel_quality_score
  + QS_WEIGHT_LANG        (def: 0.05) × lang_score
  + QS_WEIGHT_GIBBERISH   (def: 0.04) × (1 − min(gibberish_ratio, 1.0))
  + QS_WEIGHT_FUSED       (def: 0.03) × (1 − min(fused_ratio, 1.0))

quality_score = max(0.0, (base_score / total_weight) − short_penalty)
```

> [!NOTE]
> There is no `symbol_ratio` term and no `rot_penalty` subtraction in the current implementation — both
> appeared in earlier revisions. Symbol density is no longer fed into the quality score (the `symbol`
> per-line column has also been removed); rotation/inversion is now handled entirely by the
> per-line lexicon override and the page-level sweep described elsewhere in this document. `compute_quality_score()`
> still accepts an `is_upright_czech` parameter for signature compatibility, but it has no effect on the computed value.

> [!NOTE]
> **(B2) `QS_GARBAGE_NORM_MAX` vs. `CATEG_GARBAGE_DENSITY_HIGH`.** The garbage-density term inside the quality-score
> formula is now normalised against its own constant, `QS_GARBAGE_NORM_MAX` (default **0.35**), separate from
> `CATEG_GARBAGE_DENSITY_HIGH` (also default **0.35**), which gates the hard Trash override in
> [Categorisation Logic](#categorisation-logic). The two constants were previously the same value reused in both
> places, which made their individual contribution to the importance sweep inseparable. At default configuration
> both equal 0.35, so behaviour is bit-identical to before; they can now be tuned independently.

**Dynamic adjustments inside `compute_quality_score()` formula:**

**1. Garbage Penalty Guard (short clean strings)**

*Trigger:* `char_count ≤ 12`, `word_weird == 0.0`, **and** `garbage_density < QS_GARBAGE_NORM_MAX`.

*What happens:* `active_garbage_wt` is **halved** from `QS_WEIGHT_GARBAGE` (default 0.18) to 0.09. A compensating
constant of the same amount is added back to `base_score` so the total effective weight sum is unchanged and the
maximum possible score remains 1.0.

*Why:* Short archival label strings — `Lokalita:`, `Osada:`, `Okres:`, `Datum:` — contain a colon or other
structural punctuation that is counted as "garbage". Since the line is short, completely structurally clean (no weirdness),
and mostly legible, the reduced weight prevents the label from being unfairly penalised.

**2. Short Noisy Strings Penalty (`SHORT_NOISY_QS_PENALTY`)**

*Trigger:* `char_count ≤ 12` **and** (`word_weird > 0.0` **or** `garbage_density ≥ QS_GARBAGE_NORM_MAX`).

*Applied:* Subtracts the configurable `SHORT_NOISY_QS_PENALTY` (default 0.20) directly from the final score
(the result is floored at 0.0).

*Why:* An opt-in penalty to sink very short noisy strings that might otherwise artificially float into acceptable score ranges due to their minimal features.

**3. Short **Perplexity** 📉 Cap (`SHORT_PPL_CAP`)**

*Trigger:* `word_count ≤ 2` **and** raw LM **perplexity** 📉 `> SHORT_PPL_CAP` (default **850.0** for Qwen2.5-0.5B 🤖,
**2500.0** for distilgpt2 🤖).

*Applied:* in `classify_TEXT.py`, **before** `compute_quality_score()` is called. The **perplexity** 📉 value
*passed to scoring* is clamped to `SHORT_PPL_CAP`. **The stored `perplex` column in the output **CSV** 📊 is not
changed**.

*Why:* Language models assign **perplexity** 📉 by predicting tokens. With only 1–2
words, there is almost no context available, so the model assigns extremely high **perplexity** 📉 even to perfectly valid words.
Without this cap, every single-word or two-word line would receive a near-zero **perplexity** 📉 component in its **quality score** 📈.

---

## Categorisation Logic

Categorisation is a two-function split: `determine_category()` in [text_util_langID.py](../text_util.py)📎 holds
all of the routing logic and returns `(category, reason)`; `categorize_line()` is now a thin wrapper that calls
`determine_category()` and then clamps the stored `quality_score` to the range consistent with the returned category
so downstream analytics can rely on the score as a monotone proxy for category rank.

> [!NOTE]
> An earlier revision applied cumulative penalty subtractions to `quality_score` directly inside `categorize_line()`
> before threshold routing (visible as a commented-out block in the source). The current implementation replaces
> that cumulative-subtraction approach with the strict, ordered gate list below — the quality score itself is never
> mutated during categorisation, only read.

Checked in order — the **first** match wins and skips all remaining checks, including the quality-score band routing.
Every gate below is individually toggleable via the ablation kill-switch (`DISABLED_RULES`) and, when active, calls
`_fire(<rule_name>)` for the rule-fire coverage instrumentation (`tools/rule_coverage_report.py`).

| #  | Rule name              | Condition                                                                                                                                                                                                                                                                                                                                                                                                  | Result                                                                | Rationale                                                                                                                                                                                                                                                                                                                                                                                                                                        |
|----|------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0  | *(implicit)*           | `word_count == 0` or line contains only whitespace                                                                                                                                                                                                                                                                                                                                                         | `Empty`                                                               | Structural blank — no content to evaluate. Assigned before any scoring.                                                                                                                                                                                                                                                                                                                                                                          |
| 1  | `rule_hard_sweep`      | `orig_lang_score < HARD_SWEEP_LANG_MAX` (def: 0.45) **and** `ppl > HARD_SWEEP_PPL_MIN` (def: 1000.0)                                                                                                                                                                                                                                                                                                       | `Trash`                                                               | **Hard sweep.** FastText couldn't place the line at all *and* the LM found it surprising. Recorded as `trash_hard_sweep`.                                                                                                                                                                                                                                                                                                                        |
| 1a | `rule_extreme_ppl`     | `ppl ≥ PPL_EXTREME_MIN` (def: 3000.0) **and** `orig_lang_score < EXTREME_LANG_CONF` (def: 0.85)                                                                                                                                                                                                                                                                                                            | `Trash`                                                               | Very high **perplexity** 📉 alone is enough once the language guess also isn't strongly confident. Recorded as `trash_hard_sweep`.                                                                                                                                                                                                                                                                                                               |
| 1b | `rule_absolute_ppl`    | `ppl ≥ PPL_GARBAGE_ABSOLUTE` (def: 30000.0) **and not** `is_upright_czech`                                                                                                                                                                                                                                                                                                                                 | `Trash`                                                               | Catches catastrophic perplexity blow-ups regardless of language confidence, unless the line is protected by a Czech diacritic or upright function word. Recorded as `trash_hard_sweep`.                                                                                                                                                                                                                                                          |
| 2  | `rule_inverted`        | **not** `is_upright_czech` **and** (`ghost_dominated` **or** (no Czech diacritics **and** `rot_ratio ≥ SUSPICIOUS_ROT_RATIO` (def: 0.65) **and** `ppl ≥ PPL_INVERTED_MIN` (def: 200.0) **and** ghost-word hits `≥ GHOST_HITS_INVERTED_MIN` (def: 1)))                                                                                                                                                      | `Trash`                                                               | **Inverted/mirrored scan (per-line).** `ghost_dominated`: a majority of word tokens are flip-images of common Czech function words (`analyze_rotation_signals`). `is_upright_czech` (a Czech diacritic, or a real upright function word) bypasses this route. Recorded as `trash_inverted`.                                                                                                                                                      |
| 3  | `rule_allcaps`         | All alphabetic words are uppercase **and** `vowel_ratio < 0.10`                                                                                                                                                                                                                                                                                                                                            | `Trash`                                                               | Definitively unreadable: an all-caps block with almost no vowels is a visual scramble. Recorded as `allcaps_novowel`.                                                                                                                                                                                                                                                                                                                            |
| 4  | `rule_garbage_density` | `garbage_density ≥ CATEG_GARBAGE_DENSITY_HIGH` (def: 0.35), **unless** `rule_trailing_fill_rescue` fires (see below)                                                                                                                                                                                                                                                                                       | `Trash`                                                               | **Garbage-density hard override.** A line whose raw non-alphanumeric density alone exceeds the ceiling is routed to Trash directly, bypassing the weighted score. Recorded as `trash_threshold`.                                                                                                                                                                                                                                                 |
| 5  | `rule_short_garbage`   | *(skipped entirely if the line is `forgiven`, if `is_structured_line()` holds, or if `is_domain_notation()` holds — see `rule_domain_notation` below)* — `word_count ≤ ISOLATED_CHAR_MIN_TOKENS` (def: 3) **and** no Czech 🇨🇿 diacritics **and** (`lang_score ≤ LANG_SCORE_REMAP` (def: 0.75) **or** `rot_ratio ≥ SUSPICIOUS_ROT_RATIO` (def: 0.65)) **and** (gibberish present **or** `word_weird > 0`) | `Trash`                                                               | Structural short-garbage route (e.g. `olie`). Recorded as `trash_threshold`. Returns unconditionally — it does **not** consult `check_rescues()`.                                                                                                                                                                                                                                                                                                |
| 5b | `rule_domain_notation` | `is_domain_notation()` — archaeological / administrative **notation** shapes: grid and context refs (`II/C`, `I-VIII-c`, `KK-XIII`), labelled refs (`Lokalisace: MM-III`), counts (`1 ks`), abbreviation chains (`Reg.Bez.Aussig.`, `radius prox.sin.`)                                                                                                                                                    | *(suppresses gate 5; no category of its own)*                         | Consulted at **two** narrow sites — the outer guard of `rule_short_garbage`, and the two section-1 routes that convict on perplexity alone (`rule_extreme_ppl`, `rule_absolute_ppl`). It confers no other exemption, unlike `is_structured_line()`, which is read at eleven places and vetoes `_has_strong_garbage_evidence()`. Recognises **notation, not vocabulary**: `malakofauna` and `Equus caballus` need a lexicon and are deliberately left to gate 5. Dimensions (`12,5 cm`) already satisfy `is_structured_line()` and are not duplicated here. **Not exempt from `rule_hard_sweep`** — see the note below. |
| 6  | `rule_lowppl_clear`    | `ppl < LOWPPL_CLEAR_MAX` (def: 50.0) **and** `word_count ≥ 3`                                                                                                                                                                                                                                                                                                                                              | `Clear`  or `Noisy` if `valid_word_ratio < MOSTLY_READABLE_VALID_MIN` | The language model is near-certain about the text. Recorded as `lowppl_clear`, or `noisy_threshold` if capped by the mostly-readable guard.                                                                                                                                                                                                                                                                                                      |

> [!NOTE]
> **`forgiven` (`is_forgiven_headline()`)** is computed once, immediately after gate 4 and before gate 5, so genuine
> garbage caught by gates 1–4 is never rescued — forgiveness only ever lifts a line that would otherwise fall to
> `Trash` at gate 5 or later up to `Noisy`. It recognises short numbered headlines/captions (`"2, Popis nálezu i - 3"`)
> and bare domain abbreviations (`mm`, `Tb.`, `č.neg.`) that would otherwise mis-route purely because the digits/symbols
> around one or two real words drag `valid_word_ratio` down. A line is forgiven only when it carries **both** real
> content (a clean word, a listed abbreviation, or a `SHORT_VALID_WORDS` function word) **and** genuine numbering/abbreviation
> context (a short digit run, a roman numeral, or a domain abbreviation token) — a bare unnumbered prose fragment is
> never forgiven and must route on its own quality score.

**Four late-stage penalty gates** (checked after gate 6, before quality-score band routing). Each one, if triggered,
only forces a Trash/rescue outcome when `quality_score < CATEG_TRASH_SCORE_MAX + 0.35` (def: **0.90**) — a line that
already scored high enough is left alone even if one of these structural red flags fires:

| Rule name                   | Condition                                                                                                      | Rationale                                                                                          |
|-----------------------------|----------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| `rule_wqx_rot`              | (`rot_ratio > 0.50` **or** `wqx_ratio > 0.10`) **and** `orig_lang_score < 0.75` **and not** `is_upright_czech` | Rotated/mirrored-glyph density or w/q/x-heavy tokens combined with a weak original language guess. |
| `rule_vowelless`            | `word_count ≤ 3` **and** `vowel_ratio < 0.30` **and not** `is_upright_czech` **and** the line is all-caps      | Short, vowel-starved, all-caps fragments (`WVL A`).                                                |
| `rule_ledger_fragmentation` | `len(words) ≥ 4` **and** more than 60% of tokens are bare digits or ≤ 2 characters                             | Table/ledger fragmentation loophole — mostly numeric or 1–2 char tokens.                           |
| `rule_mid_uppercase`        | `word_count ≤ 2` **and** any token has unexpected mid-word uppercase                                           | Isolated mid-uppercase fragments (`ClAŕ`).                                                         |

When one of these fires **and** `quality_score < 0.90`, the outcome is resolved by `check_rescues()`, in order:
1. If `rule_trailing_fill_rescue` fires (see below) → `Noisy` / `noisy_threshold`.
2. Else if the line is `forgiven` → `Noisy` / `noisy_threshold`.
3. Else → `Trash` / `trash_threshold`.

**`rule_trailing_fill_rescue` (`_trailing_fill_rescued()`)** — used both at gate 4 and inside `check_rescues()`: if
stripping trailing fill characters (spaces, `._:-–—<`) from the line leaves a non-empty, structurally clean core
(`compute_garbage_density(core) < CATEG_GARBAGE_DENSITY_HIGH`) that either contains a Czech diacritic or is short
(`word_count ≤ 4` and `len ≤ 25`), and `valid_word_ratio > 0.0`, the line is rescued rather than dropped straight to
Trash. This protects genuine short entries that trail off with punctuation/dashes (common in tabular archival forms).

**Quality-score band routing** (reached only if none of the gates above returned):

```text
quality_score < CATEG_TRASH_SCORE_MAX  (def: 0.55)  →  check_rescues()  (Trash, unless rescued to Noisy)
quality_score ≥ CATEG_TRASH_SCORE_MAX:
    valid_word_ratio < MOSTLY_READABLE_VALID_MIN (def: 0.85) AND NOT lm_confident_czech  →  Noisy (noisy_threshold)
    otherwise                                                                            →  Clear (clear_threshold)
```

`lm_confident_czech` (`_lm_confident_czech()`) is true when `is_upright_czech` **and** `ppl < LOWPPL_CZECH_CLEAR_MAX`
(def: 180.0) **and** `garbage_density < CZECH_CLEAR_GARBAGE_MAX` (def: 0.15) — a confidently-Czech, low-perplexity,
structurally clean line is allowed through to `Clear` even if `valid_word_ratio` dips below the mostly-readable floor.

> [!IMPORTANT]
> **The "Near-Boundary Clean Prose Promotion" (Override 4) described in earlier revisions of this document has been
> removed from the current implementation.** There is no `CLEAN_PROSE_MIN_SCORE` / `CLEAN_PROSE_WC_MIN` /
> `CLEAN_PROSE_WEIRD_MAX` / `CLEAN_PROSE_PPL_MAX` promotion path in the current `determine_category()` — a `Noisy`
> line just below `CATEG_NOISY_SCORE_MAX` is no longer promoted to `Clear` by this mechanism. The closest surviving
> path to a similar outcome is `rule_lowppl_clear` (gate 6) and the `lm_confident_czech` relaxation of the mostly-readable
> guard described above.

**Score clamping after category assignment.** `categorize_line()` clamps the stored `quality_score` 📈 to the range
corresponding to the assigned band, so the **CSV** 📊 value is always internally consistent with the `categ` label:

* `Trash` → score clamped to `min(qs, CATEG_TRASH_SCORE_MAX − ε)` — always below 0.55
* `Noisy` → score clamped to `[CATEG_TRASH_SCORE_MAX, CATEG_NOISY_SCORE_MAX − ε]` — always in `[0.55, 0.80)`
* `Clear` → score clamped to `max(qs, CATEG_NOISY_SCORE_MAX)` — always ≥ 0.80

> [!IMPORTANT]
> `CATEG_NOISY_SCORE_MAX` defaults to **0.80**, not 0.85 as stated in earlier revisions of this document. The `Noisy`
> band is therefore `[0.55, 0.80)` and `Clear` is `≥ 0.80` at default configuration.

---

### Why notation is *not* exempt from `rule_hard_sweep`

Reported on issue #30: with `SHORT_PPL_CAP` lifted, `II/C`, `1 ks` and `Reg.Bez.Aussig.` still route to
`trash_hard_sweep`, so exempting notation from the perplexity routes changes nothing for an uncapping
experiment. That is correct, and it is **deliberate**.

The three section-1 routes are not equivalent. `rule_extreme_ppl` and `rule_absolute_ppl` convict on
perplexity **alone**, and perplexity is not merely weak on this population — it is *inverted*: real
notation demotes around 3 000 while `oueussd` survives to 30 000. Notation is exempt from both.
`rule_hard_sweep` additionally requires `orig_lang_score < HARD_SWEEP_LANG_MAX` (0.45) — an independent
second witness that **FastText also failed to place the line**. Notation is not exempt from it.

The reason is measured, not stylistic. `_RE_NOTATION_ABBR` requires a capital initial per segment, which
closed the *lowercase* dot-chain hole completely (< 5 % accepted). It does not close the capitalised
one: OCR garbage is frequently capitalised, and `Vvbn.Slaot.Vansas.` is structurally indistinguishable
from `Reg.Bez.Aussig.` — **over half of generated capitalised dot-chained garbage is accepted by the
predicate.** Two narrower rules were measured and rejected:

| candidate rule | garbage still accepted | real chains lost |
|---|---|---|
| a vowel in every segment | ~40 % | `Kr.Hr.`, `St.Pol.` |
| majority of segments ≤ 4 chars | ~64 % | none |

Neither earns its complexity, and both confirm that shape alone cannot separate these — the same wall
the *vocabulary* half of issue #30 runs into. So the predicate is not strong enough to carry a
hard-sweep exemption on its own, and a notation line whose language FastText also cannot place is
convicted. That is the safer error of the two available.

`TestNotationIsNotExemptFromHardSweep` pins the hole as **still present**; if a future change closes it,
that test goes red and the exemption becomes worth re-measuring.

---

## Page-relative perplexity (default OFF)

`SHORT_PPL_CAP` (850) caps perplexity for `wc <= 2`, and **the capped value is what is both scored and
stored** — the raw LM number never reaches the **CSV** 📊. Since 850 is below `HARD_SWEEP_PPL_MIN`
(1 000), `PPL_EXTREME_MIN` (3 000) and `PPL_GARBAGE_ABSOLUTE` (30 000), the consequence is blunt:

> [!IMPORTANT]
> **No perplexity rule can fire on a one- or two-token line.** For that population `rule_short_garbage`
> is the only Trash route that is not score-band routing.

Simply removing the cap does not fix it, because both surviving perplexity routes *also* gate on **low**
language-ID confidence — and on this population confidence is anti-correlated with quality: `oueussd`
scores 0.9163 while `malakofauna` scores 0.56. Uncapped, real domain vocabulary is demoted around
perplexity 3 000 while `oueussd` survives to 30 000.

`apply_page_perplexity_blend()` in [classify_TEXT.py](../classify_TEXT.py)📎 instead reads a short line
against **the long lines on its own page**, which is the comparison a human makes:

```
reference = median( perplex_raw of lines with word_count >= PAGE_PPL_LONG_MIN_WC )
blended   = exp( w · ln(own) + (1 − w) · ln(reference) )        w = PAGE_PPL_BLEND_WEIGHT
```

Both choices carry weight. The reference is a **median** so one outlier cannot poison a page — a page
containing a 6·10⁷ line still yields a reference near 39, where a mean would give 1.5·10⁷. And the
averaging happens in **log space** because perplexity here spans six orders of magnitude; an arithmetic
blend is simply the largest value. Fallback order is page → document → today's `SHORT_PPL_CAP`
behaviour, unchanged.

The pass runs immediately **before** `apply_document_postprocessing()` in both the live pipeline and the
offline re-scorer, and re-scores the affected rows through `score_line(..., apply_short_cap=False)` — the
smoothing pass only harmonises existing labels, it never recomputes a quality score.

> [!CAUTION]
> `apply_short_cap=False` is load-bearing, not a detail. `score_line()` applies `SHORT_PPL_CAP` to
> `wc <= 2`, which is the blend's *only* target population, so re-scoring with the default pinned every
> blended value straight back to 850 before any rule could read it. The pass computed a number, wrote it
> to `perplex`, and then produced a category for a different number — leaving each affected row
> internally inconsistent and the whole feature unable to move a category at any magnitude. Every blend
> test passed throughout, because they assert on the blended *number*; the only `categ` assertion was an
> idempotence check, which passes trivially when nothing moves. `TestBlendReachesTheRules` now asserts on
> a category actually changing.

| constant                  | default | role                                                                      |
|---------------------------|---------|---------------------------------------------------------------------------|
| `PAGE_PPL_BLEND_ENABLE`   | `false` | feature flag, **not** a tunable — deliberately absent from `SEARCH_SPACE` |
| `PAGE_PPL_BLEND_WEIGHT`   | 0.5     | weight on the line's own perplexity                                       |
| `PAGE_PPL_LONG_MIN_WC`    | 4       | minimum word count to count toward the reference                          |
| `PAGE_PPL_MIN_LONG_LINES` | 3       | minimum long lines before a page reference is trusted                     |

Two **CSV** 📊 columns support it (40 → 42): `perplex_raw`, the uncapped LM value — required because
blending from the already-capped `perplex` would blend a constant, and it makes the blend re-tunable
offline with no GPU — and `perplex_blend`, written only when the blend actually moved a value and left
**blank** otherwise. Blending always reads `perplex_raw`, so repeated passes are a fixed point.

> [!WARNING]
> This is a **page-consistency prior, not a garbage detector**. On a mixed page it pulls a garbage token
> toward its clean neighbours, and on a garbage page it pushes a clean token up. It also does **not**, on
> its own, make it safe to gate `rule_short_garbage`: it clears `PPL_EXTREME_MIN` but not the lang gate
> in front of it, so confidently-mislabelled garbage such as `oueussd` still escapes. The constants need
> calibrating on a real ARÚP/ARUB run before the flag is turned on, and
> `service/text_inference.py::_classify_lines` must be wired first — it holds a whole page, while
> `_classify_line` is single-line by construction.

---

## Post-Processing Smoothing

After all lines in a document are classified and written to **CSV** 📊, `apply_document_postprocessing()` in
[classify_TEXT.py](../classify_TEXT.py)📎 runs three passes, **in this order**, before the file is finalized. This
same function is reused byte-for-byte by the offline re-scorer (`tools/recategorize_from_csv.py`). Together with the
shared `score_line()` above — and a shared row formatter, so both paths round every column identically before this
pass reads them back — production output and offline re-measurement never drift.

**1. Header/footer deduplication.** All occurrences of the exact same text string across a document are identified.
If the same string has been assigned to different categories on different pages (e.g., `Obr. 1. SKUHROV NAD BĚLOU`
is `Clear` on page 3 but `Noisy` on page 4 due to slightly different surrounding context affecting the LM), all
occurrences are harmonised to the **statistical mode** — the category assigned most frequently to that string across
the document. **Recorded as** `pp_dedup`.

*Why:* Repeating strings are boilerplate — page headers, footers, running titles, standard form labels. The same
physical text should receive the same label throughout a document, and the majority vote across its occurrences is
the most reliable estimate of the correct category.

**2. Rolling-window surrounded-Trash smoothing.** Scans the document line-by-line (documents with fewer than 5 lines
are skipped entirely). If a `Noisy` ⚠️ line is surrounded by `Trash` 🗑 on both sides in a 5-line window (positions
−2 and −1 are `Trash` 🗑 **and** positions +1 and +2 are `Trash` 🗑), **and** the line's quality score is below
`CATEG_TRASH_SCORE_MAX + SURROUNDED_TRASH_QS_MARGIN` (default: **0.70**), it is downgraded to `Trash` 🗑. **Recorded as** `pp_surrounded_trash`.

*Why:* A single `Noisy` ⚠️ island embedded in four consecutive `Trash` 🗑️ lines is almost certainly corrupted text
that narrowly escaped the `Trash` 🗑 threshold. The score guard ensures that only near-boundary `Noisy` ⚠️ lines are
affected — a `Noisy` ⚠️ line with a quality score of 0.78 is left alone even in a `Trash` 🗑 neighbourhood.

**3. Page-context rules.** For each page, two symmetric page-level rules run on top of the categories left by passes
1–2, using `median_qs`, `clear_ratio`, and the fraction of lines in a trusted language (`decent_lang_ratio`, over
`EXPECTED_LANGS ∪ TRUSTED_FOREIGN_LANGS`):

* **Heavily-garbage pages:** if a page's `Clear` ratio is `≤ PAGE_GARBAGE_CLEAR_MAX` (def: 0.05), its trusted-language
  ratio is `< PAGE_GARBAGE_LANG_MAX` (def: 0.50), and its median quality score is `< PAGE_GARBAGE_MEDIAN_QS_MAX`
  (def: 0.55), every `Noisy` line on that page scoring below `PAGE_GARBAGE_NOISY_QS_MAX` (def: 0.80) is downgraded to `Trash`.
* **Predominantly-clean pages:** if a page's `Clear` ratio is `> PAGE_CLEAN_CLEAR_MIN` (def: 0.60) and its median
  quality score is `> PAGE_CLEAN_MEDIAN_QS_MIN` (def: 0.80), every `Trash` line on that page scoring
  `≥ PAGE_CLEAN_RECOVER_QS_MIN` (def: 0.45) *and* in a trusted language is promoted to `Noisy`.

**Recorded as** `pp_page_context`.

*Why:* A page that is almost entirely garbage rarely contains a genuinely-recoverable `Noisy` line; a page that is
almost entirely clean rarely contains a genuinely-unrecoverable `Trash` line. These rules use the page as additional
context the per-line categoriser cannot see.

**4. Page-level inverted-scan sweep.** Run last, independently per page, over every line not already `Empty`/`Non-text`.
A line is **suspicious** when it meets **any** of three detection arms:

* **Diacritic-absence arm:** the line lacks Czech 🇨🇿 diacritics **and** has a stored `lang_score < LANG_SCORE_ROUGH` (def: 0.45).
* **Perplexity/weirdness arm:** `perplex ≥ PPL_INVERTED_MIN` (def: 200.0) **and** `word_weird > 0.0` **and** `lang_score < ROT_HIGH_LANG_CONF` (def: 0.90). No `rot_ratio` requirement.
* **Rotation arm:** the line lacks Czech 🇨🇿 diacritics **and** `rot_ratio ≥ ROT_RATIO_INVERTED_MIN` (def: 0.55) **and** `perplex ≥ PPL_INVERTED_MIN` **and** `lang_score < ROT_HIGH_LANG_CONF`.

> [!NOTE]
> Earlier revisions of this document described only two arms and explicitly flagged that `rot_ratio` was computed
> but never actually gated the page-level sweep. That is no longer accurate: the current implementation adds a third,
> **rotation arm** that does condition on `rot_ratio ≥ ROT_RATIO_INVERTED_MIN`, alongside a perplexity/weirdness arm
> that (like before) does not use `rot_ratio` at all. There is no code/doc discrepancy to flag here anymore.

Suspicious lines are downgraded to `Trash` when either:
* they make up **≥ `INVERTED_PAGE_MAJORITY`** (default **0.60**) of the page's scoreable lines — the **page-majority arm**, checked first, applied to the whole page and skipping the run-based rule for that page; **or**
* absent a page-majority, they form a **contiguous run of `≥ INVERTED_RUN_MIN`** (default **4**) suspicious lines.

**Recorded as** `pp_inverted_run`.

*Why a page-majority arm?* Some inverted/garbage scans break up into many short, isolated fragments separated by
`Empty`🫙 lines, `Non-text`🔣 stamps, or single-token noise, so the suspicious lines never form a single run of four
and escape the run-based rule. When the **majority** of a page's scoreable lines are individually suspicious, the
page as a whole is treated as an inverted/garbage scan and every suspicious line on it is downgraded, regardless of
run length. Lines carrying Czech 🇨🇿 diacritics or a confident **FastText** 🌐 score are never suspicious, so genuine
content interleaved on the page is preserved.

*Why three arms?* Inverted-scan pages sometimes produce partial Czech 🇨🇿 diacritics: the **OCR** 🔍 engine recognises
some upside-down glyphs as plausible Latin characters and occasionally matches diacritical forms. The diacritic-absence
arm alone would miss these pages. The perplexity/weirdness arm and the rotation arm each catch them independently —
one using LM uncertainty plus word-level weirdness, the other using the character-shape rotation signal together with
LM uncertainty — without requiring the absence of diacritics on their own.

---

The table below consolidates every factor that influences `quality_score` 📈 or the final category assignment, including
where each factor is controlled and any known edge cases. This replaces the previous version of this table, which
still referenced the now-removed "Override 4" clean-prose promotion and the old `CATEG_NOISY_SCORE_MAX = 0.85` default.

| Factor                                           | Where applied                                                        | Config key(s)                                                                                                                                                                                 | Edge cases / exceptions                                                                                                                                                                                                                                                                              |
|--------------------------------------------------|----------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Valid word ratio                                 | `compute_quality_score` (35% weight)                                 | `QS_WEIGHT_VALID_WORD`                                                                                                                                                                        | All-caps **OCR** 🔍 prefix guard: tokens like `AAMMNAbSSOAO` are excluded from valid-word count even though they are alphabetically dominant.                                                                                                                                                        |
| Word weirdness ratio                             | `compute_quality_score` (18% weight)                                 | `QS_WEIGHT_WEIRD`                                                                                                                                                                             | Isolated single letters score 0.85 (**OCR** 🔍 spaced-out noise); isolated digits/measurements score 0.25 (tolerable). All-caps words and academic titles excluded from mid-uppercase detection.                                                                                                     |
| Perplexity 📉 (LM)                               | `compute_quality_score` (8% weight)                                  | `QS_WEIGHT_PERPLEXITY`, `PERPLEXITY_THRESHOLD_MAX`                                                                                                                                            | Short-text **perplexity** 📉 is capped at `SHORT_PPL_CAP` before scoring. `rule_lowppl_clear` (`ppl < 50`) bypasses thresholds entirely for highly confident predictions.                                                                                                                            |
| Text length                                      | `compute_quality_score` (2% weight)                                  | `QS_WEIGHT_LENGTH`, `QS_LENGTH_MAX`                                                                                                                                                           | Full reward for lines ≥ 100 chars; no minimum penalty for short lines.                                                                                                                                                                                                                               |
| Garbage density                                  | `compute_quality_score` (18% weight)                                 | `QS_WEIGHT_GARBAGE`, `QS_GARBAGE_NORM_MAX`                                                                                                                                                    | **Halved** to 9% for lines ≤ 12 characters with zero weirdness and low density (short-string guard). Evaluated on the original text string. Normalisation constant (`QS_GARBAGE_NORM_MAX`) is separate from the hard-gate constant (`CATEG_GARBAGE_DENSITY_HIGH`), see B2 note above.                |
| Vowel quality                                    | `compute_quality_score` (7% weight)                                  | `QS_WEIGHT_VOWEL`, `VOWEL_RATIO_LOW`, `VOWEL_RATIO_HIGH`                                                                                                                                      | Linear ramp: full score in [0.20, 0.70] vowel ratio, ramps to 0.0 outside that range.                                                                                                                                                                                                                |
| Language 🌐 confidence                           | `compute_quality_score` (5% weight)                                  | `QS_WEIGHT_LANG`                                                                                                                                                                              | Uses the **stored** (post-remapping) `lang_score`, whose value depends on `LANG_REMAP_ALWAYS` (see [Language Handling](#language--handling)); defaults to 0.5 when unavailable.                                                                                                                      |
| Gibberish ratio                                  | `compute_quality_score` (4% weight)                                  | `QS_WEIGHT_GIBBERISH`                                                                                                                                                                         | Words ≥ 60% digits/separators excluded. Detection only on words ≥ 4 characters. Folds in the w/x count.                                                                                                                                                                                              |
| Fused word ratio                                 | `compute_quality_score` (3% weight)                                  | `QS_WEIGHT_FUSED`, `FUSED_VOWEL_RUN_MIN`                                                                                                                                                      | Triggers on tokens > 14 chars, consonant runs of 5+, or vowel runs of 3+.                                                                                                                                                                                                                            |
| Hard sweep / extreme / absolute PPL (gates 1–1b) | `determine_category`                                                 | `HARD_SWEEP_LANG_MAX`, `HARD_SWEEP_PPL_MIN`, `PPL_EXTREME_MIN`, `EXTREME_LANG_CONF`, `PPL_GARBAGE_ABSOLUTE`                                                                                   | Three independent hard-Trash routes; all fold to `trash_hard_sweep`. Fire before any other check, including forgiveness.                                                                                                                                                                             |
| Inverted/mirrored lexicon (gate 2)               | `determine_category`, `analyze_rotation_signals`, `ghost_word_share` | `GHOST_DOMINATED_MIN_RATIO`, `SUSPICIOUS_ROT_RATIO`, `PPL_INVERTED_MIN`, `GHOST_HITS_INVERTED_MIN`                                                                                            | Bypassed by any Czech diacritic or upright whitelist word (`is_upright_czech`). Recorded as `trash_inverted`.                                                                                                                                                                                        |
| All-caps vowel-less (gate 3)                     | `determine_category`                                                 | none (hardcoded 0.10 vowel-ratio floor)                                                                                                                                                       | Fires only if **all** alphabetic words are uppercase **and** `vowel_ratio < 0.10`. Recorded as `allcaps_novowel`.                                                                                                                                                                                    |
| Garbage-density hard override (gate 4)           | `determine_category`                                                 | `CATEG_GARBAGE_DENSITY_HIGH`                                                                                                                                                                  | Bypassed by `rule_trailing_fill_rescue`. Recorded as `trash_threshold`.                                                                                                                                                                                                                              |
| Forgiven headline/abbreviation                   | `determine_category`, `is_forgiven_headline`, also `pre_filter_line` | `SHORT_EXCEPTION_TOKENS`, `HEADLINE_MAX_WORDS`, `HEADLINE_MAX_DIGITS`                                                                                                                         | Computed once after gate 4; only ever lifts an otherwise-Trash outcome to `Noisy`, never bypasses gates 1–4. Also used directly in the CPU pre-filter to route straight to `Process`.                                                                                                                |
| Structural short-garbage route (gate 5)          | `determine_category`                                                 | `ISOLATED_CHAR_MIN_TOKENS`, `LANG_SCORE_REMAP`                                                                                                                                                | Skipped entirely if the line is `forgiven`. Recorded as `trash_threshold`.                                                                                                                                                                                                                           |
| High LM confidence override (gate 6)             | `determine_category`                                                 | `LOWPPL_CLEAR_MAX` (NOT `PERPLEXITY_THRESHOLD_MAX`)                                                                                                                                           | Requires `ppl < 50.0` **and** `word_count ≥ 3`. Capped at `Noisy` if `valid_word_ratio < MOSTLY_READABLE_VALID_MIN`. Recorded as `lowppl_clear` / `noisy_threshold`.                                                                                                                                 |
| Late-stage structural penalty gates              | `determine_category`                                                 | none new — reuses `rot_ratio`, `wqx` density, fragmentation ratio, mid-uppercase                                                                                                              | `rule_wqx_rot`, `rule_vowelless`, `rule_ledger_fragmentation`, `rule_mid_uppercase`; each only forces a rescue/Trash outcome when `quality_score < CATEG_TRASH_SCORE_MAX + 0.35` (def. 0.90).                                                                                                        |
| Trailing-fill rescue                             | `determine_category`, `_trailing_fill_rescued`                       | `CATEG_GARBAGE_DENSITY_HIGH`                                                                                                                                                                  | Rescues short/diacritic-bearing lines whose only issue is trailing punctuation/dashes. Used at gate 4 and inside `check_rescues()`.                                                                                                                                                                  |
| Mostly readable valid cap                        | `determine_category`                                                 | `MOSTLY_READABLE_VALID_MIN`                                                                                                                                                                   | Caps semi-readable strings at `Noisy` unless `lm_confident_czech` (below) relaxes it.                                                                                                                                                                                                                |
| LM-confident-Czech relaxation                    | `determine_category`, `_lm_confident_czech`                          | `LOWPPL_CZECH_CLEAR_MAX`, `CZECH_CLEAR_GARBAGE_MAX`                                                                                                                                           | A confidently-Czech, low-perplexity, structurally clean line can reach `Clear` even below the mostly-readable floor.                                                                                                                                                                                 |
| ~~Near-boundary clean-prose promotion~~          | *(removed)*                                                          | *(removed: `CLEAN_PROSE_*` constants no longer exist)*                                                                                                                                        | Previously promoted borderline `Noisy` lines to `Clear`; this path has been removed from the current implementation. See the note in [Categorisation Logic](#categorisation-logic).                                                                                                                  |
| Short **perplexity** 📉 cap                      | `classify_TEXT.py` (before scoring)                                  | `SHORT_PPL_CAP`                                                                                                                                                                               | Applied only to lines with ≤ 2 words. Does not change the stored `perplex` column; affects only the value passed to quality scoring.                                                                                                                                                                 |
| Language 🌐 remapping                            | `classify_TEXT.py` / `remap_lang` (before scoring)                   | `EXPECTED_LANGS`, `TRUSTED_FOREIGN_LANGS`, `LANG_SCORE_REMAP`, `LANG_SCORE_REMAP_FAR`, `LANG_REMAP_ALWAYS`                                                                                    | Unknown languages remapped to first entry of `EXPECTED_LANGS`. Stored score set unconditionally (`LANG_REMAP_ALWAYS=true`, default) or capped (`=false`), except `slk` which always retains its original score. `orig_lang_score` is untouched either way and drives gates 1/2/late-stage penalties. |
| Context smoothing (rolling window)               | Post-processing pass 2, `classify_TEXT.py`                           | `CATEG_TRASH_SCORE_MAX`, `SURROUNDED_TRASH_QS_MARGIN`                                                                                                                                         | `Noisy` line must be surrounded by 2 `Trash` lines on **each** side (4 total); score must be < trash threshold + 0.15. Recorded as `pp_surrounded_trash`.                                                                                                                                            |
| Page-context rules                               | Post-processing pass 3, `classify_TEXT.py`                           | `PAGE_GARBAGE_CLEAR_MAX`, `PAGE_GARBAGE_LANG_MAX`, `PAGE_GARBAGE_MEDIAN_QS_MAX`, `PAGE_GARBAGE_NOISY_QS_MAX`, `PAGE_CLEAN_CLEAR_MIN`, `PAGE_CLEAN_MEDIAN_QS_MIN`, `PAGE_CLEAN_RECOVER_QS_MIN` | Symmetric garbage-page-pulls-down / clean-page-promotes-up rules, run **after** dedup and rolling-window smoothing, **before** the inverted-scan sweep. Recorded as `pp_page_context`.                                                                                                               |
| Page-level inverted-scan sweep                   | Post-processing pass 4, `classify_TEXT.py`                           | `ROT_RATIO_INVERTED_MIN`, `PPL_INVERTED_MIN`, `LANG_SCORE_ROUGH`, `ROT_HIGH_LANG_CONF`, `INVERTED_RUN_MIN`, `INVERTED_PAGE_MAJORITY`                                                          | Three arms (diacritic-absence, perplexity/weirdness, rotation) — see above. Suspicious lines Trashed via page-majority (checked first) or a run of ≥ 4. Recorded as `pp_inverted_run`.                                                                                                               |
| Header/footer deduplication                      | Post-processing pass 1, `classify_TEXT.py`                           | none                                                                                                                                                                                          | Based on **exact text match** across the whole document; harmonises to modal category. Recorded as `pp_dedup`. Runs **first**, before the other three passes.                                                                                                                                        |
