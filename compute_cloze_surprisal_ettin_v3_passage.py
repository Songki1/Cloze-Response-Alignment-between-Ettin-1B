# -*- coding: utf-8 -*-
r"""
compute_cloze_surprisal_ettin_v2_passage.py  (Ettin passage-level scoring — Ettin version of GPT-2 v6)
=========================================================================
■ Why it is needed
  The v1_word decoder scores the option tokens using only the context 'before the blank' (it does not see the passage after the blank).
  The encoder PLL sees both sides of the blank, so the difference between the two models was confounded with a 'difference in access to information'.
  This script scores the probability of the whole passage so that the decoder also uses the passage after the blank.
  (The option list is not shown → no positional habit or copying bias of the kind produced by the index method)

■ Passage = A (before the blank) + option + B (after the blank, excluding vocabulary footnotes and option list at the end of the passage)

  decoder(CLM) — the same three ranges as GPT-2 v6; three files from one run
  ┌─────────┬─────────────────────┬──────────┬─────────────────────────────────────┐
  │ range   │ score               │ agg.     │ meaning                             │
  ├─────────┼─────────────────────┼──────────┼─────────────────────────────────────┤
  │ choice  │ option tokens       │ word 2-st│ = v1_word decoder (sees only before)│
  │ context │ B tokens            │ sum      │ log P(B | A, option) — partial score│
  │ full    │ option + B tokens   │ sum      │ log P(option, B | A) ★ main candidate│
  └─────────┴─────────────────────┴──────────┴─────────────────────────────────────┘
  A tokens have the same value whichever option is inserted (only the left side is seen), so they cancel in the softmax → excluded from scoring.
  Choice ranking of full = ranking of the Bayesian posterior P(option | A, B) → a judgement that 'sees the whole passage'.

  For the decoder, choice_sum (sum over option tokens = log P(option | A)) is also saved.

  encoder(MLM)
  │ choice     │ option-token PLL (sees both sides)│ word 2-st│ = same values as v1_word encoder│
  │ choice_sum │ option-token PLL                  │ sum      │ ≈ log P(option | A, B)      │
  │ full       │ whole-passage token PLL (--enc-full)│ sum    │ very slow. optional         │
  ※ For the encoder, the A tokens also change value depending on the option (bidirectional), so A cannot be excluded.

■ Paired comparison (pairs with identical information conditions and aggregation)
  ★ encoder choice_sum  vs  decoder full
     decoder full = log P(option, B | A) = log P(option | A, B) + log P(B | A)
                    → the second term is a constant independent of the option → the softmax is exactly P(option | A, B) (within the five options)
     encoder choice_sum = Σ log P(option piece | other option pieces, A, B) → PLL approximation of the same quantity
     difference: the decoder gives an exact probability, the encoder a pseudo-likelihood (PLL; inflation of multi-token words, Kauf & Ivanova 2023)
  · encoder choice(word) vs decoder choice(word) = the earlier v1_word pair (the decoder cannot see after the blank)

■ Measurement rules
  · The original text (prefix + option + suffix) is tokenised in one pass and the ranges are determined from the offsets (same as v1_word·v6)
  · Tokens overlapping the option range = option / tokens lying 'entirely' after the option and before the footnotes start = B
  · decoder: the Ettin BOS ([CLS]) is prepended (same as v1_word) → no problem of being unable to score position 0
  · The encoder choice input is the same as v1_word: [CLS] whole original text [SEP] (footnotes included)
  · Max positions 7,999 → CSAT passages are not truncated (stops with an error if exceeded)

■ Output: {prefix}_choice.xlsx / _context.xlsx / _full.xlsx (decoder), {prefix}_choice.xlsx [/ _full.xlsx] (encoder)
  Column structure is the same as GPT-2 v6 → usable as is with build_meta_and_alignment.py etc.
  Given a --students metadata file, the mean item-level Kendall τ_b with the students' distractor distribution is printed immediately

Run (PowerShell)
  python compute_cloze_surprisal_ettin_v2_passage.py --arch dec --size 1b --selftest
  python compute_cloze_surprisal_ettin_v2_passage.py --arch enc --size 1b --selftest
  python compute_cloze_surprisal_ettin_v2_passage.py --arch dec --size 1b --input 20261018_csat_data_original.xlsx --limit 50 --students Metadata_AI_Reviewed_GPT2_Clark.xlsx
  python compute_cloze_surprisal_ettin_v2_passage.py --arch dec --size 1b --input 20261018_csat_data_original.xlsx --students Metadata_AI_Reviewed_GPT2_Clark.xlsx
  python compute_cloze_surprisal_ettin_v2_passage.py --arch enc --size 1b --input 20261018_csat_data_original.xlsx --students Metadata_AI_Reviewed_GPT2_Clark.xlsx
"""

import argparse
import math
import re
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

LN2 = math.log(2.0)
CIRCLED = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5}
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
ETTIN_SIZES = ["17m", "32m", "68m", "150m", "400m", "1b"]
PCT_DEC = {"choice": 2, "choice_sum": 6, "context": 6, "full": 6}


# ══════════════════════════════════════════════════════════════════════════
# 0) Model name · finding Excel columns · parsing options  (same as v1_word)
# ══════════════════════════════════════════════════════════════════════════
def resolve_model_name(arch, size, model=None):
    if model:
        return model
    return f"jhu-clsp/ettin-{'encoder' if arch == 'enc' else 'decoder'}-{size}"


def short_tag(model_name):
    m = re.search(r"ettin-(encoder|decoder)-(\w+)$", model_name)
    if m:
        return ("enc" if m.group(1) == "encoder" else "dec") + m.group(2)
    return re.sub(r"[^\w\-]", "_", model_name.rstrip("/\\").split("/")[-1].split("\\")[-1])


def find_col(df, *subs):
    for c in df.columns:
        if all(s in str(c) for s in subs):
            return c
    return None


def detect_text_col(df):
    best, col = 0, None
    for c in df.columns:
        cnt = df[c].astype(str).str.contains("[MASK]", regex=False, na=False).sum()
        if cnt > best:
            best, col = cnt, c
    return col


def detect_choice_cols(df):
    cols = [c for c in df.columns if ("Option" in str(c))]
    return sorted(cols, key=lambda c: int(m.group(1)) if (m := re.search(r"(\d)", str(c))) else 99)[:5]


def parse_choices(row, choice_cols):
    slots = {}
    for ci, col in enumerate(choice_cols, start=1):
        c = row[col]
        if c is None or str(c) == "nan":
            continue
        s = str(c).strip()
        ms = [(m.start(), CIRCLED[m.group()]) for m in re.finditer(r"[①②③④⑤]", s)]
        if not ms:
            slots.setdefault(ci, s)
            continue
        for i, (pos, num) in enumerate(ms):
            end = ms[i + 1][0] if i + 1 < len(ms) else len(s)
            slots.setdefault(num, s[pos + 1:end].strip())
    return [slots.get(i, "").strip() for i in range(1, 6)]


# ══════════════════════════════════════════════════════════════════════════
# 1) Position of vocabulary footnotes and option list at the end of the passage  (same as GPT-2 v6)
# ══════════════════════════════════════════════════════════════════════════
FOOTNOTE_START = re.compile(r"\*{1,3}\s*[^:*\n]{1,80}:")


def tail_offset_in_suffix(suffix):
    """Position in suffix where footnotes (* omission: 생략) or an option list (①…②…) begin. len(suffix) if none."""
    cut = len(suffix)
    for m in re.finditer(r"\*", suffix):
        p = m.start()
        if FOOTNOTE_START.match(suffix, p) and re.search(r"[가-힣]", suffix[p:]):
            cut = min(cut, p)
            break
    m = re.search(r"①", suffix)
    if m and re.search(r"②", suffix[m.start():]):
        cut = min(cut, m.start())
    return cut


# ══════════════════════════════════════════════════════════════════════════
# 2) Splitting token ranges (works without a model — pure function)
# ══════════════════════════════════════════════════════════════════════════
def partition(tok, prefix, choice, suffix):
    """Tokenise the original text in one pass and classify each token as A / option / B / tail (footnotes).
    Returned dict:
      ids        : token ids (no special tokens)
      a_pos      : A token positions (tokens ending before the option starts)
      choice_pos : token positions overlapping the option
      choice_wid : whitespace-word index of each option token (v1_word rule: the word with the largest overlap)
      b_pos      : positions of tokens lying entirely after the option ~ before the footnotes
      b_has_word : whether B contains at least one letter or digit (if not, B = punctuation only)
    """
    choice = choice.strip()
    full = prefix + choice + suffix
    cs, ce = len(prefix), len(prefix) + len(choice)
    cut = ce + tail_offset_in_suffix(suffix)
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = list(enc["input_ids"]), list(enc["offset_mapping"])
    spans = [(cs + m.start(), cs + m.end()) for m in re.finditer(r"\S+", choice)]

    def word_of(s, e):
        ov = [min(e, we) - max(s, ws) for ws, we in spans]
        return int(np.argmax(ov))

    a_end = len(prefix.rstrip())          # end position of the last 'character' of A (excluding whitespace)
    a_pos, gap_pos, choice_pos, choice_wid, b_pos = [], [], [], [], []
    for p, (s, e) in enumerate(offs):
        if e <= s:
            continue
        if s < ce and e > cs:
            choice_pos.append(p)
            choice_wid.append(word_of(s, e))
        elif s >= ce and e <= cut:
            b_pos.append(p)
        elif e <= cs:
            if s >= a_end and not full[s:e].strip():
                gap_pos.append(p)         # an isolated whitespace token between A and the option (rare)
            else:
                a_pos.append(p)
    if choice_pos and choice_pos != list(range(choice_pos[0], choice_pos[-1] + 1)):
        raise ValueError("Option tokens are not contiguous")
    b_text = "".join(full[offs[p][0]:offs[p][1]] for p in b_pos)
    return {"ids": ids, "a_pos": a_pos, "gap_pos": gap_pos, "choice_pos": choice_pos, "choice_wid": choice_wid,
            "b_pos": b_pos, "b_has_word": bool(re.search(r"[A-Za-z0-9]", b_text)),
            "b_text": b_text}


def word_score(lps, wids):
    """v1_word --agg word: mean within words → mean of word values"""
    if not lps:
        return float("-inf")
    means = [np.mean([lps[i] for i in range(len(lps)) if wids[i] == w]) for w in sorted(set(wids))]
    return float(np.mean(means))


def kendall_tau_b(x, y):
    """Tie-corrected Kendall τ_b (computed directly for small samples). nan if one side is entirely tied"""
    n = len(x)
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx == dy:
                conc += 1
            else:
                disc += 1
    den = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    return (conc - disc) / den if den > 0 else float("nan")


# ══════════════════════════════════════════════════════════════════════════
# 3) Loading the model  (same as EttinScorer in v1_word)
# ══════════════════════════════════════════════════════════════════════════
def detect_objective(config, name):
    if getattr(config, "masked_prediction", False) or re.search(r"enc(oder)?-from-dec", name):
        sys.exit("[error] The encoder-from-decoder (MNTP) variant is not supported.")
    archs = " ".join(getattr(config, "architectures", None) or [])
    if getattr(config, "is_causal", False) or "CausalLM" in archs or config.model_type.endswith("decoder"):
        return "clm"
    return "mlm"


class EttinScorer:
    def __init__(self, name, device, dtype, objective=None, verbose=True):
        self.name, self.device = name, device
        config = AutoConfig.from_pretrained(name)
        self.objective = objective or detect_objective(config, name)
        cls = AutoModelForCausalLM if self.objective == "clm" else AutoModelForMaskedLM
        if hasattr(config, "reference_compile"):
            config.reference_compile = False
        kw = dict(config=config, output_loading_info=True)
        try:
            try:
                model, info = cls.from_pretrained(name, dtype=dtype, **kw)
            except TypeError as e:
                if "dtype" not in str(e):
                    raise
                model, info = cls.from_pretrained(name, torch_dtype=dtype, **kw)
        except (KeyError, ValueError) as e:
            sys.exit(f"[error] Model format not recognised ({e}). pip install -U transformers")
        tied = set(getattr(model, "_tied_weights_keys", None) or [])
        missing = [k for k in info.get("missing_keys", []) if k not in tied]
        if missing:
            sys.exit(f"[error] Missing weights (randomly initialised): {missing[:10]}")
        self.model = model.to(device).eval()
        self.head = model.head if self.objective == "mlm" else model.lm_head
        self.tok = AutoTokenizer.from_pretrained(name)
        if not self.tok.is_fast:
            sys.exit("[error] Offset mapping requires a fast tokenizer.")
        with_sp = self.tok("a")["input_ids"]
        without = self.tok("a", add_special_tokens=False)["input_ids"]
        k = with_sp.index(without[0]) if without and without[0] in with_sp else 0
        self.bos = with_sp[:k] if self.objective == "clm" else []
        self.max_pos = getattr(config, "max_position_embeddings", 512)
        if verbose:
            print(f"[check] objective={self.objective.upper()} / missing weights 0 ✅ / max positions {self.max_pos} / "
                  f"BOS={self.tok.convert_ids_to_tokens(self.bos) or 'none'}", file=sys.stderr)

    @torch.no_grad()
    def hidden(self, ids):
        return self.model.model(input_ids=ids).last_hidden_state

    @torch.no_grad()
    def logits_from_hidden(self, h):
        return self.model.decoder(self.head(h))


# ══════════════════════════════════════════════════════════════════════════
# 4) Measurement
# ══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def clm_logprobs_at(sc, ids, positions):
    """decoder: one forward pass over BOS + ids; returns the log probability of the token at each position p of ids (= prediction from one position earlier)"""
    if not positions:
        return []
    seq = sc.bos + ids[:max(positions) + 1]
    if len(seq) > sc.max_pos:
        raise ValueError(f"Input of {len(seq)} tokens exceeds the maximum of {sc.max_pos}")
    off = len(sc.bos)
    if off == 0 and min(positions) == 0:
        raise ValueError("Without a BOS the first token cannot be scored")
    h = sc.hidden(torch.tensor([seq], device=sc.device))[0]
    prev = torch.tensor([off + p - 1 for p in positions], device=sc.device)
    lp = torch.log_softmax(sc.logits_from_hidden(h.index_select(0, prev)).float(), dim=-1)
    return [lp[k, ids[p]].item() for k, p in enumerate(positions)]


@torch.no_grad()
def mlm_pll_at(sc, ids, positions, mask_batch):
    """encoder: log probability of the original token in copies of [CLS] ids [SEP] with each position masked in turn (Salazar PLL)"""
    base = [sc.tok.cls_token_id] + ids + [sc.tok.sep_token_id]
    if len(base) > sc.max_pos:
        raise ValueError(f"Input of {len(base)} tokens exceeds the maximum of {sc.max_pos}")
    out = []
    for b0 in range(0, len(positions), mask_batch):
        rows = positions[b0:b0 + mask_batch]
        batch = torch.tensor([base] * len(rows), device=sc.device)
        for r, p in enumerate(rows):
            batch[r, 1 + p] = sc.tok.mask_token_id
        h = sc.hidden(batch)
        sel = h[torch.arange(len(rows), device=sc.device), torch.tensor([1 + p for p in rows], device=sc.device)]
        lp = torch.log_softmax(sc.logits_from_hidden(sel).float(), dim=-1)
        out += [lp[r, ids[p]].item() for r, p in enumerate(rows)]
    return out


def measure(sc, prefix, choice, suffix, mask_batch, enc_full=False):
    """Range-wise scores (nat; larger = more plausible) and diagnostic information for the passage with one option inserted"""
    part = partition(sc.tok, prefix, choice, suffix)
    ids, gp, cp, bp = part["ids"], part["gap_pos"], part["choice_pos"], part["b_pos"]
    if not cp:
        raise ValueError(f"No option tokens: {choice!r}")
    sco = {}
    gap_sum = 0.0
    if sc.objective == "clm":
        lps = clm_logprobs_at(sc, ids, gp + cp + bp)
        g_lps, c_lps, b_lps = lps[:len(gp)], lps[len(gp):len(gp) + len(cp)], lps[len(gp) + len(cp):]
        gap_sum = float(sum(g_lps))
        sco["choice"] = word_score(c_lps, part["choice_wid"])       # same rule as v1_word (whitespace tokens excluded)
        sco["choice_sum"] = float(sum(c_lps))                       # log P(option | A)
        sco["context"] = float(sum(b_lps))
        sco["full"] = gap_sum + float(sum(c_lps) + sum(b_lps))      # whitespace tokens also counted on the option side
    else:
        c_lps = mlm_pll_at(sc, ids, cp, mask_batch)
        sco["choice"] = word_score(c_lps, part["choice_wid"])
        sco["choice_sum"] = float(sum(c_lps))                       # PLL sum ≈ log P(option | A, B)
        if enc_full:   # PLL of the whole passage (A + option + B). Footnote tokens are context only
            allp = part["a_pos"] + gp + cp + bp
            sco["full"] = float(sum(mlm_pll_at(sc, ids, allp, mask_batch)))
    return sco, len(cp), len(set(part["choice_wid"])), len(bp), part["b_has_word"], c_lps, gap_sum


# ══════════════════════════════════════════════════════════════════════════
# 5) Implementation check (--selftest)
# ══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_selftest(sc, dtype_name, mask_batch):
    ok = True
    tol = 1e-4 if dtype_name == "fp32" else 5e-2
    text = ("It was a hot morning. He opened the [MASK] and the milk was gone. "
            "He sighed. * sigh: 한숨 쉬다")
    alt = text.replace("한숨 쉬다", "전혀 다른 각주 내용입니다")
    choices = ["window", "fridge", "refrigerator door"]
    prefix, suffix = text.split("[MASK]")

    def line(t):
        print("\n" + "=" * 72 + f"\n  {t}\n" + "=" * 72)

    line("[Test A] option score = value computed by hand from the full logits (reproduces the v1_word rule)")
    for ch in choices:
        sco, T, W, nb, _, c_lps, _ = measure(sc, prefix, ch, suffix, 2)
        part = partition(sc.tok, prefix, ch, suffix)
        ids, cp = part["ids"], part["choice_pos"]
        if sc.objective == "clm":
            full = sc.model(input_ids=torch.tensor([sc.bos + ids], device=sc.device)).logits[0].float().log_softmax(-1)
            ref = [full[len(sc.bos) + p - 1, ids[p]].item() for p in cp]
        else:
            base = [sc.tok.cls_token_id] + ids + [sc.tok.sep_token_id]
            batch = torch.tensor([base] * len(cp), device=sc.device)
            for r, p in enumerate(cp):
                batch[r, 1 + p] = sc.tok.mask_token_id
            full = sc.model(input_ids=batch).logits.float().log_softmax(-1)
            ref = [full[r, 1 + p, ids[p]].item() for r, p in enumerate(cp)]
        diff = max(abs(a - b) for a, b in zip(ref, c_lps))
        same = diff < tol and W == len(ch.split())
        ok &= same
        print(f"  '{ch}' pieces={sc.tok.convert_ids_to_tokens([ids[p] for p in cp])} T={T} W={W} "
              f"|Δ|={diff:.1e} {'✅' if same else '❌'}")

    if sc.objective == "clm":
        line("[Test B] the sum of A-token log probabilities is identical across options (so excluding it leaves the ranking unchanged)")
        a_sums = []
        for ch in choices:
            part = partition(sc.tok, prefix, ch, suffix)
            a_sums.append(sum(clm_logprobs_at(sc, part["ids"], part["a_pos"])))
        spread = max(a_sums) - min(a_sums)
        ok &= spread < 1e-3
        print(f"  A sums: {[round(x, 5) for x in a_sums]} (range {spread:.1e}) {'✅' if spread < 1e-3 else '❌'}")

        line("[Test C] context·full unchanged when the footnote content is changed")
        p2, s2 = alt.split("[MASK]")
        c_ok = True
        for ch in choices:
            a = measure(sc, prefix, ch, suffix, 2)[0]
            b = measure(sc, p2, ch, s2, 2)[0]
            c_ok &= abs(a["context"] - b["context"]) < 1e-5 and abs(a["full"] - b["full"]) < 1e-5
        ok &= c_ok
        print(f"  → {'✅' if c_ok else '❌'}")

        line("[Test D] full = Σ option log values (+ isolated whitespace tokens) + context")
        d_ok = True
        for ch in choices:
            sco, _, _, _, _, c_lps, gs = measure(sc, prefix, ch, suffix, 2)
            d_ok &= abs(sco["full"] - (sum(c_lps) + gs + sco["context"])) < 1e-6
        ok &= d_ok
        print(f"  → {'✅' if d_ok else '❌'}")

    line("[Test E] whether footnotes are excluded from the B range")
    bt = partition(sc.tok, prefix, "fridge", suffix)["b_text"]
    e_ok = bt.strip().endswith("He sighed.") and "*" not in bt
    ok &= e_ok
    print(f"  B range = {bt!r} {'✅' if e_ok else '❌'}")

    line("[Test F] pre-training knowledge check — does the natural option win (full / choice for the encoder)")
    rng = "full" if sc.objective == "clm" else "choice"
    for pre_, good, bad, suf_ in [("She drank a cup of hot ", "coffee", "gravel", " this morning."),
                                  ("He opened the ", "fridge", "theorem", " and took out the milk.")]:
        g = measure(sc, pre_, good, suf_, 2)[0][rng]
        b = measure(sc, pre_, bad, suf_, 2)[0][rng]
        ok &= g > b
        print(f"  {good}={g:+.3f} vs {bad}={b:+.3f} {'✅' if g > b else '❌'}")

    print("\n" + "=" * 72 + ("\n  ✅ All checks passed" if ok else "\n  ❌ Check failed") + "\n" + "=" * 72 + "\n")
    return ok


# ══════════════════════════════════════════════════════════════════════════
# 6) Reading items · alignment with the student distribution
# ══════════════════════════════════════════════════════════════════════════
def iter_items(path, limit=0):
    xls = pd.ExcelFile(path)
    n = 0
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        text_col, choice_cols = detect_text_col(df), detect_choice_cols(df)
        if text_col is None or len(choice_cols) < 5:
            print(f"[warn] Sheet '{sheet}' skipped", file=sys.stderr)
            continue
        cols = {"gold": find_col(df, "Answer", "No") or find_col(df, "Answer"),
                "grade": find_col(df, "Grade"), "year": find_col(df, "Year"), "month": find_col(df, "Month"),
                "qno": find_col(df, "Item_No") or find_col(df, "Item", "No"), "err": find_col(df, "Error")}
        for _, row in df.dropna(subset=[text_col]).iterrows():
            text = str(row[text_col])
            if text.count("[MASK]") != 1:
                continue
            prefix, suffix = text.split("[MASK]")
            yield sheet, row, cols, prefix, suffix, parse_choices(row, choice_cols)
            n += 1
            if limit and n >= limit:
                return


KEY = ["Grade", "Year", "Month", "Item_No"]


def norm_keys(d):
    d = d.copy()
    d["Grade"] = d["Grade"].astype(str).str.extract(r"(\d)")[0]
    for k in ["Year", "Month", "Item_No"]:
        d[k] = pd.to_numeric(d[k], errors="coerce").astype("Int64").astype(str)
    return d


def alignment_report(df_range, students_path, label):
    """Mean item-level Kendall τ_b between the students' distractor distribution (four, key excluded) and the model pct"""
    meta = pd.concat(pd.read_excel(students_path, sheet_name=None).values(), ignore_index=True)
    S = [f"Student Choice {j}" for j in range(1, 6)]
    if not all(c in meta.columns for c in S):
        print(f"[warn] {students_path} has no 'Student Choice 1' ~ 'Student Choice 5' columns; skipping the alignment computation",
              file=sys.stderr)
        return
    for c in S:
        meta[c] = pd.to_numeric(meta[c], errors="coerce")
    mm = norm_keys(df_range).merge(norm_keys(meta[KEY + S]), on=KEY, how="inner")
    taus, top = [], []
    for _, r in mm.iterrows():
        if pd.isna(r["Answer_No"]):
            continue
        g = int(r["Answer_No"]) - 1
        dis = [j for j in range(5) if j != g]
        s = np.array([r[S[j]] for j in dis], float)
        # Rank by surprisal (bits, 4 decimal places) instead of the rounded pct (lower = preferred → sign reversed).
        # Because with summed scores the softmax becomes extreme, some items have all pct tied at 0.000000.
        p = -np.array([r[f"surprisal{j+1}"] for j in dis], float)
        if np.isnan(s).any():
            continue
        t = kendall_tau_b(s, p)
        if not np.isnan(t):
            taus.append(t)
            top.append(int(np.argmax(s) == np.argmax(p)))
    if not taus:
        print(f"[warn] {label}: no items matched to the student distribution (check Grade·Year·Month·Item_No)", file=sys.stderr)
        return
    t = np.array(taus)
    se = t.std(ddof=1) / math.sqrt(len(t)) if len(t) > 1 else float("nan")
    print(f"[align] {label:<8} n={len(t)}  τ̄_b={t.mean():+.4f}  95%CI[{t.mean()-1.96*se:+.4f}, "
          f"{t.mean()+1.96*se:+.4f}]  match with students' most popular distractor={np.mean(top):.1%} (chance 25%)", file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════
# 7) Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--arch", default="dec", choices=["enc", "dec"])
    ap.add_argument("--size", default="1b", choices=ETTIN_SIZES)
    ap.add_argument("--model", default=None)
    ap.add_argument("--outprefix", default=None, help="if empty, ettin_{arch}{size}_P")
    ap.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mask-batch", type=int, default=None, help="encoder copy batch (if empty, 1b=4, otherwise=16)")
    ap.add_argument("--enc-full", action="store_true", help="also compute the whole-passage PLL for the encoder (very slow)")
    ap.add_argument("--students", default=None, help="metadata Excel with student selection rates (alignment τ_b computed immediately)")
    ap.add_argument("--check-against", default=None,
                    help="existing v1_word --agg word result (same model). regression check that the choice range gives the same values")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    name = resolve_model_name(args.arch, args.size, args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and args.dtype != "fp32":
        args.dtype = "fp32"
    mb = args.mask_batch or (4 if re.search(r"-1b$", name) else 16)
    print(f"[info] model={name}, device={device}, dtype={args.dtype}, mask_batch={mb}", file=sys.stderr)
    sc = EttinScorer(name, device, DTYPES[args.dtype])
    if args.model is None and (sc.objective == "mlm") != (args.arch == "enc"):
        sys.exit("[error] --arch does not match the actual model type")
    if args.selftest:
        sys.exit(0 if run_selftest(sc, args.dtype, mb) else 1)
    if not args.input:
        ap.error("--input is required (or --selftest)")

    ranges = ["choice", "choice_sum", "context", "full"] if sc.objective == "clm" else \
        (["choice", "choice_sum", "full"] if args.enc_full else ["choice", "choice_sum"])
    outprefix = args.outprefix or f"ettin_{short_tag(name)}_P"
    records = {r: [] for r in ranges}
    n_noword = 0
    for k, (sheet, row, c, pre, suf, choices) in enumerate(iter_items(args.input, args.limit), 1):
        per = {r: [] for r in ranges}
        ntok, nword, nb, b_word = [], [], [], True
        for ch in choices:
            sco, T, W, B, bw, _, _ = measure(sc, pre, ch, suf, mb, args.enc_full)
            for r in ranges:
                per[r].append(sco[r])
            ntok.append(T)
            nword.append(W)
            nb.append(B)
            b_word &= bw
        n_noword += int(not b_word)
        gold = row[c["gold"]] if c["gold"] else None
        try:
            gold = int(gold)
        except (ValueError, TypeError):
            gold = None
        for r in ranges:
            s = np.array(per[r], float)
            w = np.exp((s - s.max()) / args.temperature)
            pct = w / w.sum() * 100
            pick = int(np.argmax(s)) + 1
            rec = {"Grade": row[c["grade"]] if c["grade"] else sheet,
                   "Year": row[c["year"]] if c["year"] else None,
                   "Month": row[c["month"]] if c["month"] else None,
                   "Item_No": row[c["qno"]] if c["qno"] else None,
                   "Error_Rate": row[c["err"]] if c["err"] else None,
                   "Answer_No": gold, "Model_Choice": pick, "Correct": (pick == gold) if gold else None,
                   "Aggregation": "word" if r == "choice" else "sum", "Scoring_Range": r,
                   "Model": name, "Objective": sc.objective.upper(), "BOS": "yes" if sc.bos else "no",
                   "B_Tokens": nb[0], "B_No_Words": not b_word, "context_Indistinguishable": max(nb) == 0}
            for i in range(5):
                rec[f"Option{i+1}"] = choices[i]
                rec[f"surprisal{i+1}"] = round(-s[i] / LN2, 4)
                rec[f"pct{i+1}"] = round(pct[i], PCT_DEC[r])
                rec[f"ntok{i+1}"] = ntok[i]
                rec[f"nword{i+1}"] = nword[i]
            if gold and 1 <= gold <= 5:
                rec["Correct_Pct"] = round(pct[gold - 1], PCT_DEC[r])
            records[r].append(rec)
        if k % 25 == 0:
            print(f"[info] {k} items processed", file=sys.stderr)

    print("", file=sys.stderr)
    for r in ranges:
        out = f"{outprefix}_{r}.xlsx"
        d = pd.DataFrame(records[r])
        d.to_excel(out, index=False)
        v = d.dropna(subset=["Correct"])
        if r == "context":
            v = v[~v["context_Indistinguishable"]]
        pk = v["Model_Choice"].values - 1
        nt = v[[f"ntok{i}" for i in range(1, 6)]].values
        var = nt.max(1) > nt.min(1)
        longest = (nt[np.arange(len(v)), pk] == nt.max(1))[var].mean() if var.any() else float("nan")
        shortest = (nt[np.arange(len(v)), pk] == nt.min(1))[var].mean() if var.any() else float("nan")
        print(f"[result] {r:<7} N={len(v)}  accuracy={v['Correct'].mean():.1%}  "
              f"chose longest option={longest:.1%} / shortest={shortest:.1%}  "
              f"choice distribution A~E={np.bincount(pk, minlength=5).tolist()}  → {out}", file=sys.stderr)
        if args.students:
            alignment_report(d, args.students, r)
    if sc.objective == "clm":
        print(f"[info] no words after the blank (B = punctuation only): {n_noword} items → the context range carries almost no information for these items",
              file=sys.stderr)
    if args.check_against:
        ref = norm_keys(pd.read_excel(args.check_against))
        new = norm_keys(pd.DataFrame(records["choice"]))
        mg = ref.merge(new, on=KEY, suffixes=("_ref", "_new"))
        if len(mg):
            diff = max((mg[f"surprisal{i}_ref"] - mg[f"surprisal{i}_new"]).abs().max() for i in range(1, 6))
            print(f"[check] choice range vs {args.check_against}: {len(mg)} items matched, max|Δsurprisal|={diff:.5f} bits "
                  f"→ {'✅ v1_word reproduced' if diff < 1e-3 else '⚠ needs checking (may reflect a different data version or answer corrections)'}",
                  file=sys.stderr)
        else:
            print("[check] no matched items (check Grade·Year·Month·Item_No)", file=sys.stderr)


if __name__ == "__main__":
    main()
