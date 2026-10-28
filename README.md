# CSAT English Cloze Items: Reader Alignment of Ettin 1B pairs

Reproduction package for the paper *"How Well Are Language Models Aligned
with Readers' Cloze-Solving Behaviour? Comparing BERT and GPT-2 against
Reader Responses on Korean CSAT English Cloze Items"* (submitted to
PACLIC 40).

한국 수능(및 전국연합학력평가) 영어 빈칸추론 문항 715개에서, BERT-base와
GPT-2의 선지 확률분포가 실제 응시자(독자)의 선택 분포와 얼마나 정렬되는지를
측정한 논문의 재현 패키지입니다.

## Repository layout

| File | Role |
|---|---|
| `compute_cloze_surprisal_bert.py` | BERT-base scoring: sequential-masking PLL (paper Eqs. A.2.1–A.2.2) → softmax (A.3) |
| `compute_cloze_surprisal_gpt2.py` | GPT-2 scoring: chain-rule mean log-likelihood (Eq. A.1) → softmax (A.3) |
| `verify_paper_stats.py` | Recomputes **every** figure in Tables 1–2 and Appendix A.5 from the item-level Excel files |
| `build_item_level_workbook.py` | Generates the formula-based audit workbook below |
| `Item-level Alignment Results.xlsx` | **Per-item audit workbook.** 715 rows; every statistic is a live Excel formula (30,057 of them), no hardcoded results |
| `requirements.txt` | Python dependencies |

## Reproduction

```bash
pip install -r requirements.txt

# Step 1 (optional, GPU recommended): re-score the items with each model.
python compute_cloze_surprisal_bert.py --input <items.xlsx> --output <bert_scored.xlsx>
python compute_cloze_surprisal_gpt2.py --input <items.xlsx> --output <gpt2_scored.xlsx>

# Step 2: verify all paper statistics from the scored files.
python verify_paper_stats.py \
    --bert 메타정보_AI_검수_완료_BERT.xlsx \
    --gpt2 메타정보_AI_검수_완료_GPT2.xlsx
```

Expected output (abridged):

```
[Table 1] BERT-base: n = 714  (excluded: [('Grade 3', 2015, 6, 31)])
   Pearson  r̄  = 0.087282   Spearman ρ̄ = 0.072192   Kendall τ̄a = 0.059057
[Table 2] BERT-base: n_err = 498
   ᾱ = 0.729044   ᾱ_random = 0.710229
   rank match k=1..4: 27.5% (137)  25.3% (126)  25.5% (127)  21.7% (108)
   binomial test: p = 0.1079
[Table 1] GPT-2: n = 715
   Pearson  r̄  = 0.126953   Spearman ρ̄ = 0.112459   Kendall τ̄a = 0.093240
[Table 2] GPT-2: n_err = 438
   ᾱ = 0.745770   ᾱ_random = 0.710229
   rank match k=1..4: 31.5% (138)  24.7% (108)  24.4% (107)  19.4% (85)
   binomial test: p = 0.0013
```

## Formula ↔ code map (Appendix A)

| Eq. | Definition | Implementation |
|---|---|---|
| A.1 | GPT-2 chain-rule mean log-likelihood | `compute_cloze_surprisal_gpt2.py::choice_pll` — `total += logprobs[p-1, ids[p]]`, then `total / len(cpos)` |
| A.2.1 | PLL = **sum** of masked-token log-probs | `compute_cloze_surprisal_bert.py::choice_pll` — `total += logprobs[i, pos, choice_ids[i]]` |
| A.2.2 | L = PLL / T (length normalisation) | same function — `return total / len(positions)` |
| A.3 | softmax over the five options | `softmax_pct()` in both scripts (max-subtraction for numerical stability; invariant) |
| A.4.1 | Pearson over 4 distractors | `verify_paper_stats.py::pearson` |
| A.4.2 | Spearman = Pearson of RANK.AVG ranks | `verify_paper_stats.py::spearman` |
| A.4.3 | Kendall τ_a (no tie correction) | `verify_paper_stats.py::kendall_tau_a` |
| A.5 | R = corr(error rate, model P(key)) | `verify_paper_stats.py::appendix_a5` |
| A.6.1–A.6.3 | error indicator, α, ᾱ | `verify_paper_stats.py::table2` |
| A.6.4 | random baseline ᾱ_random | same — computed over all 715 items |

### Conventions that matter

- **Tie handling for the rank match rates (Table 2).** Ranks are assigned by
  *competition (minimum) ranking*: distractors tied on the student selection
  rate share the higher rank. This is consistent with the denominator of α
  being the maximum **value** `max_{j≠c_i} s_ij` (not an argmax index), and it
  is invariant to the ordering of the options. Two items in the data contain
  such ties.
- **τ_a vs τ_b.** τ_a leaves ties uncorrected and is conservative. Tied pairs
  are asymmetric across models (BERT 81/4,290 = 1.9%; GPT-2 26/4,290 = 0.6%);
  recomputing with tie-corrected τ_b gives BERT .060 / GPT-2 .093, so the
  models' relative standing is unaffected.
- **Student rates are used raw.** They sum to 94.94% on average (some
  responses unrecorded); all reported indices are invariant to positive
  rescaling, so no renormalisation is applied.
- **One excluded item.** Grade 3, June 2015, item 31 is excluded for BERT
  only: its softmax distribution is degenerate `[0,0,0,100,0]`, leaving the
  distractor-side correlation undefined (zero variance).

## Data

The item-level Excel files (passages, options, keys, student selection
rates, model probabilities) are **not redistributed here**: the passages are
copyrighted by KICE / provincial offices of education. The verification
script runs on the metadata files described above.

## The audit workbook

`Item-level Alignment Results.xlsx` is what Appendix A.7 refers to when it says the
computation "is implemented in a formula-based spreadsheet, so that it can be
verified cell by cell." Nothing in it is a pasted result.

| Sheet | Contents |
|---|---|
| `Data` | 715 rows. Cols A–T are the raw inputs; **cols U–BG are all formulas** — distractor extraction `INDEX(range, IF(k<c_i, k, k+1))`, average ranks, `CORREL`, `SIGN`-based τ_a, α, competition rank, random baseline |
| `Summary` | Tables 1 and 2 and Appendix A.5, every cell an `AVERAGE`/`COUNTIF`/`CORREL` over `Data`, plus two self-check rows |
| `Helper` | Error rate and P(key) columns feeding the A.5 correlation |
| `Notes` | Definitions, the equation-to-column map, the tie convention, and the excluded item |

Open any cell in `Data!U2:BG716` and the formula is visible. LibreOffice
recalculation reports 30,057 formulas and **0 errors**, reproducing
`verify_paper_stats.py` to four decimal places.

