# Ettin encoder–decoder scoring of Korean CSAT English cloze items(Trial version)

This folder contains the data, model outputs and scripts needed to inspect and reproduce the
main analysis: how closely the option preferences of paired Ettin language models
(Weller et al., 2026) align with students' answer choices on English cloze (blank-filling)
items from the Korean College Scholastic Ability Test (CSAT) and the nationwide high school
mock examinations.

Each of the five options is inserted into the blank and the **whole passage** is scored, using
four Ettin 1B models in a 2 × 2 design (pre-training objective × prediction mode used at scoring time):

| | Scored by masked prediction | Scored by next-token prediction |
|---|---|---|
| **Pre-trained with MLM** | `jhu-clsp/ettin-encoder-1b` (encoder) | `jhu-clsp/ettin-dec-from-enc-1b` (encoder→decoder) |
| **Pre-trained with CLM** | `jhu-clsp/ettin-enc-from-dec-1b` (decoder→encoder, MNTP) | `jhu-clsp/ettin-decoder-1b` (decoder) |

All models are used as released, without fine-tuning.

> **Translation note.** All files are English translations of the original Korean-language
> files. Only text was translated (column names, comments, messages, file names); data values
> and program logic are unchanged. See [Notes](#notes).

---

## Folder contents (12 files)

### 1. Input data

| File | Description |
|---|---|
| `20261018_csat_data_original.xlsx` | 761 cloze items: passage with the blank marked `[MASK]`, five options, key and item error rate. Input to the scoring scripts. |

Items by grade: Grade 1 (Korean high school year 1, i.e. Grade 10): 210; Grade 2: 198; Grade 3: 353; total 761.
Student response distributions are missing for 46 items, so all alignment analyses use **715 items**.

### 2. Scoring scripts (require the models; GPU recommended)

| Script | Purpose |
|---|---|
| `compute_cloze_surprisal_ettin_v4_cross.py` | **Main scoring script.** Whole-passage scoring for all four models (`--arch enc / dec / dec-from-enc / enc-from-dec`). For the MNTP model it checks the attention direction and selects the read-out position (shift 0 or 1) from calibration sentences on every run. Produces the four `ettin_*_P_full.xlsx` files. |
| `compute_cloze_surprisal_ettin_v3_passage.py` | Previous version, original encoder and decoder only (`--arch enc / dec`). Kept for provenance; v4 contains the same scoring for these two models and adds the two cross-objective models. (Its docstring still carries the v2 file name in the example commands, as in the original.) |

### 3. Model outputs (whole-passage scores used in the paper)

| File | Model | `Objective` |
|---|---|---|
| `ettin_enc1b_P_full.xlsx` | encoder | `MLM` |
| `ettin_dec1b_P_full.xlsx` | decoder | `CLM` |
| `ettin_decfromenc1b_P_full.xlsx` | encoder→decoder | `CLM` |
| `ettin_encfromdec1b_P_full.xlsx` | decoder→encoder | `MNTP(shift=1,bidirectional)` |

One row per item (761 rows). Encoder family (`enc`, `enc-from-dec`): pseudo-log-likelihood (PLL) summed over
all passage tokens (`--enc-full`). Decoder family (`dec`, `dec-from-enc`): log probability of the option and
the following context, summed (chain rule). Only the `full` range is included here; the other ranges
(`choice`, `choice_sum`, `context`) that the scripts also write are not part of this folder.

### 4. Item metadata with Ettin results

| File | Model columns |
|---|---|
| `Metadata_AI_Reviewed_Ettin_encoder1B_full_Clark.xlsx` | `Ettin-encoder-1B …` |
| `Metadata_AI_Reviewed_Ettin_decoder1B_full_Clark.xlsx` | `Ettin-decoder-1B …` |
| `Metadata_AI_Reviewed_Ettin_dec-from-enc1B_full_Clark.xlsx` | `Ettin-dec-from-enc-1B …` |
| `Metadata_AI_Reviewed_Ettin_enc-from-dec1B_full_Clark.xlsx` | `Ettin-enc-from-dec-1B …` |

Each file has three sheets (`Grade 3`: 353 items, `Grade 2`: 198, `Grade 1`: 210) with the item metadata, the
**students' selection rates** for each option and the whole-passage results of one model. The item and student
columns are identical in all four files; only the model columns differ. These files are also the student data
source for the analysis (see [Reproducing the results](#reproducing-the-results)).

### 5. Analysis script (no GPU or model download needed)

| Script | Purpose |
|---|---|
| `analysis_alignment_stats.py` | **All alignment statistics reported in the paper**, from the four `*_P_full.xlsx` files and the student selection rates: accuracy; item-level Pearson *r*, Spearman *ρ* and Kendall *τ*_b over the four distractors (key excluded), with SE and 95% CI; error-alignment score *α*, its random baseline and rank-wise match rates with exact binomial tests; pairwise paired *t* tests with Holm correction across the three indices, *d*_z and bootstrap CIs; *α* on items both models answered incorrectly; exact McNemar tests; 2 × 2 contrasts; item-level *τ*_b correlations between models; items with no context after the blank; *τ*_b ratio and gap-closure bootstrap CIs; sensitivity analysis (minimum detectable *d*_z). Writes `alignment_stats.json` and `alignment_items_{model}.csv`. |

---

## Column dictionary

### Input data (`20261018_csat_data_original.xlsx`) and item columns of the outputs

| Column | Meaning |
|---|---|
| `Grade` | `Grade 1`–`Grade 3` = Korean high school years 1–3 (Grades 10–12). |
| `Year` | Year of the examination. |
| `Month` | Month of the mock examination; `CSAT odd` = the CSAT (odd-numbered test form). |
| `Item_No` | Item number in the test booklet. |
| `Error_Rate` | Percentage of students who answered the item incorrectly. |
| `Answer_No` | Number of the correct option (1–5). |
| `Option1`–`Option5` | Option text. |
| `Item(with_blank)` | Passage with the blank marked `[MASK]`. |

### Whole-passage outputs (`ettin_*_P_full.xlsx`)

| Column | Meaning |
|---|---|
| `Model_Choice` | Option with the highest normalised probability (= lowest surprisal). |
| `Correct` | Whether `Model_Choice` equals `Answer_No`. |
| `Aggregation` | `sum` (summed over the scored tokens). |
| `Scoring_Range` | `full` (whole passage). |
| `Model` | Hugging Face model name. |
| `Objective` | `MLM`, `CLM` or `MNTP(shift=1,bidirectional)`. |
| `BOS` | Whether a beginning-of-sequence token was prepended. |
| `B_Tokens` | Number of scored tokens after the blank (B = context after the blank, excluding end-of-passage vocabulary footnotes). |
| `B_No_Words` | True if the context after the blank contains no letters or digits (the blank ends the passage). |
| `context_Indistinguishable` | True if no option has any scored token after the blank. |
| `surprisal1`–`surprisal5` | Score of each option as surprisal in **bits** (4 decimal places); lower = preferred. |
| `pct1`–`pct5` | Softmax probability over the five options (%), rounded to 6 decimal places. |
| `ntok1`–`ntok5`, `nword1`–`nword5` | Number of subword tokens and of words in each option. |
| `Correct_Pct` | Probability (%) assigned to the correct option. |

Whole-passage distributions are highly concentrated (the median top-option probability is above 98% for every
model), so `pct` values rounded to 6 decimal places can tie among distractors. **Rank distractors from
`surprisal`, not from `pct`**; the analysis script does this.

### Metadata files (`Metadata_AI_Reviewed_Ettin_*_full_Clark.xlsx`)

| Column | Meaning |
|---|---|
| `Year`, `Month`, `Grade`, `Item_No`, `Error_Rate` | As above. |
| `Points` | Item points. |
| `Item (footnotes shown; [] points excluded)` | Passage with vocabulary footnotes, without the point label. |
| `Option 1`–`Option 5`, `Answer Text`, `Answer No` | Options, correct option text and number. |
| `Student Choice 1`–`Student Choice 5` | Percentage of students choosing each option (EBSi online response data). Non-numeric for the 46 items without response data. |
| `{model} Model Choice`, `{model} Model Correct` | The model's choice and its correctness. |
| `{model} Prob 1`–`{model} Prob 5` | Softmax probability (%) of each option, recomputed from the surprisal values (bits) in the corresponding `*_P_full.xlsx` and stored to 12 significant digits to avoid artificial ties. |

---

## Requirements

- Python 3 (developed with Python 3.12)
- Analysis: `numpy`, `pandas`, `openpyxl`, `scipy` (`statsmodels` optional, for the sensitivity analysis)
- Scoring: additionally `torch` and `transformers` (the Ettin models are downloaded from Hugging Face)

```bash
pip install numpy pandas openpyxl scipy statsmodels torch transformers
```

The 1B models run in fp32 (about 4 GB of weights). Whole-passage PLL for the encoder family
(`--enc-full`) is slow because every passage token is masked in turn.

---

## Reproducing the results

> **Student data file.** The scripts were written with `Metadata_AI_Reviewed_GPT2_Clark.xlsx` as the
> student data source (an earlier metadata file with the same `Student Choice 1`–`5` and `Answer No`
> columns). That file is **not** in this folder. Any of the four Ettin metadata files can be used instead,
> because their item and student columns are identical. Using
> `Metadata_AI_Reviewed_Ettin_decoder1B_full_Clark.xlsx` reproduces `alignment_stats.json` and all four
> `alignment_items_{model}.csv` files exactly (verified).

### Statistical analysis (from the saved outputs)

Change line 36 of `analysis_alignment_stats.py` from

```python
STUDENTS = "Metadata_AI_Reviewed_GPT2_Clark.xlsx"
```

to

```python
STUDENTS = "Metadata_AI_Reviewed_Ettin_decoder1B_full_Clark.xlsx"
```

and run, from this folder:

```bash
python analysis_alignment_stats.py
```

Expected values (n = 715 items):

| Model | Accuracy | Mean *r* | Mean *ρ* | Mean *τ*_b [95% CI] |
|---|---|---|---|---|
| encoder | .762 | .116 | .119 | .104 [.069, .138] |
| decoder | .797 | .168 | .190 | .162 [.126, .197] |
| encoder→decoder | .710 | .138 | .181 | .155 [.118, .192] |
| decoder→encoder (MNTP) | .695 | .120 | .134 | .109 [.073, .145] |

Bootstrap confidence intervals (10,000 item resamples, seed 0) may differ from those reported in the paper in
the last digit, because they depend on the random resamples.

### Model scoring (GPU recommended)

Each command writes `ettin_{model}_P_{range}.xlsx` (including `_P_full.xlsx`) and, with `--students`, prints the
mean item-level Kendall's *τ*_b over the four distractors and its 95% CI.

```bash
# Optional implementation self-check (once per model)
python compute_cloze_surprisal_ettin_v4_cross.py --arch dec --size 1b --selftest

# Encoder (MLM) and decoder→encoder (MNTP): --enc-full is required for whole-passage scores
python compute_cloze_surprisal_ettin_v4_cross.py --arch enc          --size 1b --enc-full --input 20261018_csat_data_original.xlsx --students Metadata_AI_Reviewed_Ettin_decoder1B_full_Clark.xlsx
python compute_cloze_surprisal_ettin_v4_cross.py --arch enc-from-dec --size 1b --enc-full --input 20261018_csat_data_original.xlsx --students Metadata_AI_Reviewed_Ettin_decoder1B_full_Clark.xlsx

# Decoder (CLM) and encoder→decoder
python compute_cloze_surprisal_ettin_v4_cross.py --arch dec          --size 1b --input 20261018_csat_data_original.xlsx --students Metadata_AI_Reviewed_Ettin_decoder1B_full_Clark.xlsx
python compute_cloze_surprisal_ettin_v4_cross.py --arch dec-from-enc --size 1b --input 20261018_csat_data_original.xlsx --students Metadata_AI_Reviewed_Ettin_decoder1B_full_Clark.xlsx
```

With the default output names, new runs overwrite the `ettin_*_P_full.xlsx` files in this folder; use `--outprefix` to write elsewhere, or
`--check-against ettin_dec1b_P_full.xlsx` (for example) to compare a new run with the saved result.
Other options: `--limit N` (first N items only), `--dtype fp16`, `--mask-batch` (encoder copies per batch),
`--mntp-shift 0|1` (override the calibrated read-out position for the MNTP model).
The example commands in the scripts' docstrings name `Metadata_AI_Reviewed_GPT2_Clark.xlsx`; substitute an
Ettin metadata file as above.

---

## Notes

- **Korean text kept on purpose.** Some passages end with Korean vocabulary glosses
  (e.g. `* gratification: 만족감`), as in the original test booklets. These are part of the test
  material and were not translated. The scripts detect these glosses by their Korean characters
  (regular expression `[가-힣]`) and exclude them from scoring, so translating them would change
  the scores. For the same reason, the self-test strings in the scripts that imitate such glosses
  (e.g. `* sigh: 한숨 쉬다`) were also kept.
- **One code line simplified in translation.** In the original scripts, option columns were found
  by two Korean keywords that both translate as "Option"; the English version searches for
  `"Option"` once. The columns found are the same.
- **Verification of the translation.** Apart from translated header cells and one translated cell
  value (`MNTP(shift=1,bidirectional)`), all spreadsheet cells are identical to the originals. The
  English scripts reproduce the original scripts' outputs on the checks that do not require
  downloading a model (item validation; alignment computed from the saved results). Model scoring
  itself was not rerun after translation.
- **Copyright.** The test items are copyrighted by their publishers (the Korea Institute for
  Curriculum and Evaluation for the CSAT and the metropolitan and provincial offices of education
  for the mock examinations). Please check redistribution conditions before sharing the item text.
- **Student data.** Student selection rates are the response distributions published by EBSi and
  are based on a voluntary online sample, not on the full population of test-takers.

## References

- Salazar, J., Liang, D., Nguyen, T. Q., & Kirchhoff, K. (2020). Masked language model scoring. *Proceedings of ACL 2020*, 2699–2712.
- Weller, O., Ricci, K., Marone, M., Chaffin, A., Lawrie, D., & Van Durme, B. (2026). Seq vs seq: An open suite of paired encoders and decoders. *ICLR 2026*. arXiv:2507.11412.
