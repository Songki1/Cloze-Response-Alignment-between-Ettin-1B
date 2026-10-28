# -*- coding: utf-8 -*-
r"""
compute_cloze_surprisal_ettin_v4_cross.py  (Ettin passage-level scoring — version supporting cross-objective models)
=========================================================================
■ Changes from v3
  The two Ettin 'cross-objective' models can now be scored → 2×2 design

                         │ scored now by blank filling (bidir.)│ scored now by next-token prediction (causal)
  ───────────────────────┼────────────────────────────────────┼──────────────────────────────────
  pre-trained with MLM     │ --arch enc          (original encoder)│ --arch dec-from-enc (encoder→decoder)
  pre-trained with CLM     │ --arch enc-from-dec (decoder→encoder)│ --arch dec          (original decoder)

  · dec-from-enc : the encoder further trained with CLM for 50B tokens → scored exactly like the original decoder (chain rule)
  · enc-from-dec : the decoder further trained with MNTP for 50B tokens (Weller et al., Ettin paper § cross-objective training)
      MNTP (masked next token prediction, LLM2Vec style) = token i is masked with [MASK] and
      predicted from the output at 'the immediately preceding position i−1'. Reading position i as with an ordinary encoder therefore gives wrong values.
      → This script (1) switches attention to bidirectional and (2) reads at position i−1.
      → On every run, a 'calibration check' confirms from data that this setting is right (see ■ Safeguards below).

■ Within the same column, the scoring formula is the same
  left column (enc, enc-from-dec) : sum of PLL from masking every passage token in turn  (--enc-full)
  right column (dec, dec-from-enc): chain-rule sum of log P(option, B | A)                (full)
  → For comparisons within a column, the objection "because the scoring formulas differ" does not apply.

■ Safeguards (enc-from-dec)
  1) Attention-direction check: whether earlier-position outputs change when later tokens are changed → confirms bidirectionality
  2) Position-shift check: on short English sentences, the mean log probability of masked tokens is compared between
     'reading at the same position (shift 0)' and 'reading at the preceding position (shift 1)' → the higher one is selected automatically
     (with the correct setting the difference is large; a warning is given if it is small)
  dec and dec-from-enc are checked for causal attention, and enc for bidirectional attention, in the same way.

■ Passage = A (before the blank) + option + B (after the blank, excluding vocabulary footnotes and option list at the end of the passage)
  decoder(CLM: dec, dec-from-enc)
    choice  option tokens, 2-stage word aggregation / choice_sum sum over option tokens / context sum over B tokens / full sum over option+B ★
  encoder(MLM: enc, MNTP: enc-from-dec)
    choice  option-token PLL, 2-stage word / choice_sum sum of option-token PLL / full sum of whole-passage PLL (--enc-full) ★

■ Output: {prefix}_{range}.xlsx — same column structure as v3. The 'Objective' column records CLM / MLM / MNTP(shift=1, bidirectional).

Run (PowerShell) — the two new models
  python compute_cloze_surprisal_ettin_v4_cross.py --arch enc-from-dec --size 1b --selftest
  python compute_cloze_surprisal_ettin_v4_cross.py --arch dec-from-enc --size 1b --selftest
  python compute_cloze_surprisal_ettin_v4_cross.py --arch enc-from-dec --size 1b --enc-full --input 20261018_csat_data_original.xlsx --students Metadata_AI_Reviewed_GPT2_Clark.xlsx
  python compute_cloze_surprisal_ettin_v4_cross.py --arch dec-from-enc --size 1b --input 20261018_csat_data_original.xlsx --students Metadata_AI_Reviewed_GPT2_Clark.xlsx
  (the original enc and dec also run as before: --arch enc / --arch dec)
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
ARCHS = {"enc": "encoder", "dec": "decoder",
         "enc-from-dec": "encoder-from-decoder", "dec-from-enc": "decoder-from-encoder"}
EXPECTED_OBJ = {"enc": "mlm", "dec": "clm", "enc-from-dec": "mntp", "dec-from-enc": "clm"}

# Short English sentences for the position-shift and attention-direction checks (common expressions, independent of model knowledge)
CALIB_SENTENCES = [
    "The children walked to school together on a cold winter morning.",
    "Scientists have found that regular exercise improves both memory and mood.",
    "She opened the window because the room was too hot to sleep in.",
    "Many people believe that reading books is the best way to learn new words.",
    "After the meeting, the manager sent an email to everyone in the office.",
    "The price of fresh vegetables usually rises when the weather is bad.",
]


# ══════════════════════════════════════════════════════════════════════════
# 0) Model name · finding Excel columns · parsing options
# ══════════════════════════════════════════════════════════════════════════
# For the cross-objective models, the name in the model-card table (ettin-encoder-from-decoder-1b) differs from the actual repository name
# (a shortened form such as ettin-dec-from-enc-32m) → candidates are tried in turn
SHORT = {"enc-from-dec": "enc-from-dec", "dec-from-enc": "dec-from-enc"}


def candidate_names(arch, size, model=None):
    if model:
        return [model]
    names = [f"jhu-clsp/ettin-{ARCHS[arch]}-{size}"]
    if arch in SHORT:
        names.insert(0, f"jhu-clsp/ettin-{SHORT[arch]}-{size}")
    return names


def resolve_model_name(arch, size, model=None):
    """The first candidate name whose config can actually be fetched. If all fail, print guidance and exit"""
    cands = candidate_names(arch, size, model)
    errs = []
    for n in cands:
        try:
            AutoConfig.from_pretrained(n)
            return n
        except OSError as e:
            errs.append(f"{n}: {str(e).splitlines()[0][:120]}")
    sys.exit("[error] Model repository not found. Names tried:\n  " + "\n  ".join(errs) +
             "\nCheck the actual name: python -c \"from huggingface_hub import list_models; "
             "print([m.id for m in list_models(author='jhu-clsp', search='from')])\"\n"
             "→ pass the name found with --model (keep --arch as is)")


def short_tag(model_name):
    m = re.search(r"ettin-(encoder|decoder|enc|dec)(?:-from-(encoder|decoder|enc|dec))?-(\w+)$", model_name)
    if m:
        ab = {"encoder": "enc", "decoder": "dec", "enc": "enc", "dec": "dec"}
        return ab[m.group(1)] + (f"from{ab[m.group(2)]}" if m.group(2) else "") + m.group(3)
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
# 1) Position of vocabulary footnotes and option list at the end of the passage  (same as v3)
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
# 2) Splitting token ranges (same as v3)
# ══════════════════════════════════════════════════════════════════════════
def partition(tok, prefix, choice, suffix):
    """Tokenise the original text in one pass and classify each token as A / option / B / tail (footnotes)."""
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

    a_end = len(prefix.rstrip())
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
                gap_pos.append(p)
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
# 3) Loading the model
# ══════════════════════════════════════════════════════════════════════════
def detect_objective(config, name):
    """clm (next-token prediction) / mlm (blank filling) / mntp (decoder→encoder cross-objective training)"""
    if getattr(config, "masked_prediction", False) or re.search(r"enc(oder)?-from-dec", name):
        return "mntp"
    archs = " ".join(getattr(config, "architectures", None) or [])
    if getattr(config, "is_causal", False) or "CausalLM" in archs or config.model_type.endswith("decoder"):
        return "clm"
    return "mlm"


def _load(cls, name, dtype, config, extra=None):
    kw = dict(config=config, output_loading_info=True, **(extra or {}))
    try:
        try:
            return cls.from_pretrained(name, dtype=dtype, **kw)
        except TypeError as e:
            if "dtype" not in str(e):
                raise
            return cls.from_pretrained(name, torch_dtype=dtype, **kw)
    except (KeyError, ValueError) as e:
        sys.exit(f"[error] Model format not recognised ({e}). pip install -U transformers")


class EttinScorer:
    def __init__(self, name, device, dtype, objective=None, verbose=True):
        self.name, self.device = name, device
        config = AutoConfig.from_pretrained(name)
        self.objective = objective or detect_objective(config, name)
        # If the MNTP checkpoint is in decoder format (ModernBertDecoder), a bidirectional attention mask must be passed explicitly
        self.decoder_body = config.model_type.endswith("decoder")
        self.force_bi = self.objective == "mntp" and self.decoder_body
        if self.objective == "clm" or self.decoder_body:
            cls = AutoModelForCausalLM
        else:
            cls = AutoModelForMaskedLM
        if hasattr(config, "reference_compile"):
            config.reference_compile = False
        extra = {"attn_implementation": "sdpa"} if self.force_bi else None
        model, info = _load(cls, name, dtype, config, extra)
        tied = set(getattr(model, "_tied_weights_keys", None) or [])
        missing = [k for k in info.get("missing_keys", []) if k not in tied]
        if missing:
            sys.exit(f"[error] Missing weights (randomly initialised): {missing[:10]}")
        self.model = model.to(device).eval()
        self.head = model.head if hasattr(model, "head") else model.lm_head
        self.tok = AutoTokenizer.from_pretrained(name)
        if not self.tok.is_fast:
            sys.exit("[error] Offset mapping requires a fast tokenizer.")
        with_sp = self.tok("a")["input_ids"]
        without = self.tok("a", add_special_tokens=False)["input_ids"]
        k = with_sp.index(without[0]) if without and without[0] in with_sp else 0
        pre_sp, post_sp = with_sp[:k], with_sp[k + len(without):]
        if self.objective == "clm":
            self.bos, self.pre, self.post = pre_sp, [], []
        else:
            # encoder input = [CLS] ids [SEP] (same as v1_word). Added manually if the tokeniser does not add them
            cls_id = self.tok.cls_token_id if self.tok.cls_token_id is not None else self.tok.bos_token_id
            sep_id = self.tok.sep_token_id if self.tok.sep_token_id is not None else self.tok.eos_token_id
            self.bos = []
            self.pre = pre_sp or ([cls_id] if cls_id is not None else [])
            self.post = post_sp or ([sep_id] if sep_id is not None else [])
            if self.tok.mask_token_id is None:
                sys.exit("[error] The tokenizer has no [MASK] token.")
            if self.objective == "mntp" and not self.pre:
                sys.exit("[error] MNTP predicts the first token from the preceding position, so a leading special token is required.")
        self.shift = 0                          # for mntp, set by calibrate_mntp()
        self.max_pos = getattr(config, "max_position_embeddings", 512)
        self.window = getattr(config, "sliding_window", None)
        if self.window is None or self.window < 0:
            la = getattr(config, "local_attention", None)
            self.window = la // 2 if la else None
        if verbose:
            print(f"[check] objective={self.objective.upper()} / body={'decoder-type' if self.decoder_body else 'encoder-type'} / "
                  f"missing weights 0 ✅ / max positions {self.max_pos} / "
                  f"leading special token={self.tok.convert_ids_to_tokens(self.bos or self.pre) or 'none'}", file=sys.stderr)

    # ── Bidirectional attention mask (decoder-type MNTP only) ─────────────────────────────
    def _bi_masks(self, n, L):
        dt = next(self.model.parameters()).dtype
        full = torch.zeros((n, 1, L, L), dtype=dt, device=self.device)
        pos = torch.arange(L, device=self.device)
        far = (pos[None, :] - pos[:, None]).abs() > self.window if self.window else torch.zeros((L, L), dtype=torch.bool, device=self.device)
        sl = torch.zeros((L, L), dtype=dt, device=self.device).masked_fill(far, torch.finfo(dt).min)
        return {"full_attention": full, "sliding_attention": sl.expand(n, 1, L, L)}

    @torch.no_grad()
    def hidden(self, ids, bidirectional=None):
        """ids: (n, L). bidirectional=None uses the model's own scheme; True gives a decoder-type model a bidirectional mask"""
        bi = self.force_bi if bidirectional is None else bidirectional
        if bi and self.decoder_body:
            return self.model.model(input_ids=ids, attention_mask=self._bi_masks(ids.shape[0], ids.shape[1])).last_hidden_state
        return self.model.model(input_ids=ids).last_hidden_state

    @torch.no_grad()
    def logits_from_hidden(self, h):
        return self.model.decoder(self.head(h))


# ══════════════════════════════════════════════════════════════════════════
# 4) Measurement
# ══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def clm_logprobs_at(sc, ids, positions):
    """decoder: one forward pass over BOS + ids; log probability of the token at each position p of ids (= prediction from one position earlier)"""
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
def masked_pll_at(sc, ids, positions, mask_batch, shift=None, bidirectional=None):
    """encoder family: log probability of the original token in copies of [CLS] ids [SEP] with each position masked in turn.
    shift=0 → read at the masked position (MLM, Salazar PLL)
    shift=1 → read at the position immediately before the masked one (MNTP)"""
    shift = sc.shift if shift is None else shift
    base = sc.pre + ids + sc.post
    off = len(sc.pre)
    if len(base) > sc.max_pos:
        raise ValueError(f"Input of {len(base)} tokens exceeds the maximum of {sc.max_pos}")
    if positions and off + min(positions) - shift < 0:
        raise ValueError("Without a leading special token the first token cannot be scored")
    out = []
    for b0 in range(0, len(positions), mask_batch):
        rows = positions[b0:b0 + mask_batch]
        batch = torch.tensor([base] * len(rows), device=sc.device)
        for r, p in enumerate(rows):
            batch[r, off + p] = sc.tok.mask_token_id
        h = sc.hidden(batch, bidirectional)
        read = torch.tensor([off + p - shift for p in rows], device=sc.device)
        sel = h[torch.arange(len(rows), device=sc.device), read]
        lp = torch.log_softmax(sc.logits_from_hidden(sel).float(), dim=-1)
        out += [lp[r, ids[p]].item() for r, p in enumerate(rows)]
    return out


# v3-compatible names
def mlm_pll_at(sc, ids, positions, mask_batch):
    return masked_pll_at(sc, ids, positions, mask_batch, shift=0)


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
        sco["choice"] = word_score(c_lps, part["choice_wid"])
        sco["choice_sum"] = float(sum(c_lps))
        sco["context"] = float(sum(b_lps))
        sco["full"] = gap_sum + float(sum(c_lps) + sum(b_lps))
    else:                                    # mlm or mntp — same range; only the read-out position differs (sc.shift)
        c_lps = masked_pll_at(sc, ids, cp, mask_batch)
        sco["choice"] = word_score(c_lps, part["choice_wid"])
        sco["choice_sum"] = float(sum(c_lps))
        if enc_full:
            allp = part["a_pos"] + gp + cp + bp
            sco["full"] = float(sum(masked_pll_at(sc, ids, allp, mask_batch)))
    return sco, len(cp), len(set(part["choice_wid"])), len(bp), part["b_has_word"], c_lps, gap_sum


# ══════════════════════════════════════════════════════════════════════════
# 5) Safeguards: attention-direction and position-shift checks
# ══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def probe_direction(sc, bidirectional=None):
    """Bidirectional if the output at the first position changes when only the last token is changed. Returns: 'bi' / 'causal'"""
    ids = sc.tok(CALIB_SENTENCES[0], add_special_tokens=False)["input_ids"]
    other = sc.tok("zebra", add_special_tokens=False)["input_ids"][0]
    a = (sc.bos or sc.pre) + ids + sc.post
    b = list(a)
    last = len(a) - 1 - len(sc.post)
    b[last] = other if b[last] != other else (other + 1)
    ha = sc.hidden(torch.tensor([a], device=sc.device), bidirectional)[0, 1].float()
    hb = sc.hidden(torch.tensor([b], device=sc.device), bidirectional)[0, 1].float()
    return "bi" if (ha - hb).abs().max().item() > 1e-4 else "causal"


@torch.no_grad()
def mean_masked_lp(sc, shift, bidirectional=None, mask_batch=8):
    """Mean log probability (nat) of the original tokens when every token of the check sentences is masked in turn"""
    vals = []
    for s in CALIB_SENTENCES:
        ids = sc.tok(s, add_special_tokens=False)["input_ids"]
        vals += masked_pll_at(sc, ids, list(range(len(ids))), mask_batch, shift=shift, bidirectional=bidirectional)
    return float(np.mean(vals))


def calibrate(sc, forced_shift=None, mask_batch=8):
    """Print the check results for each model type and, for MNTP, set the read-out position (shift). False if there is a problem"""
    ok = True
    if sc.objective == "clm":
        d = probe_direction(sc)
        ok = d == "causal"
        print(f"[calib] attention direction = {'causal' if d == 'causal' else 'bidirectional'} "
              f"{'✅' if ok else '❌ (a next-token model but bidirectional → scoring is meaningless)'}", file=sys.stderr)
        return ok
    d = probe_direction(sc)
    print(f"[calib] attention direction = {'bidirectional' if d == 'bi' else 'causal'}"
          f"{' (bidirectional mask applied to a decoder-type body)' if sc.force_bi else ''} "
          f"{'✅' if d == 'bi' else '❌ (a blank-filling model but causal)'}", file=sys.stderr)
    ok &= d == "bi"
    m0, m1 = mean_masked_lp(sc, 0, mask_batch=mask_batch), mean_masked_lp(sc, 1, mask_batch=mask_batch)
    print(f"[calib] mean log probability of masked tokens: reading at the same position (shift 0) {m0:+.3f} / "
          f"reading at the preceding position (shift 1) {m1:+.3f} nat", file=sys.stderr)
    if sc.force_bi:
        mc = mean_masked_lp(sc, 1, bidirectional=False, mask_batch=mask_batch)
        print(f"[calib]   (reference) reading at the preceding position with causal attention = ordinary next-token prediction {mc:+.3f} nat", file=sys.stderr)
    best = 1 if m1 > m0 else 0
    if sc.objective == "mlm" and best != 0:
        print("[calib] ❌ an original encoder but shift 1 is higher → check the model and tokenizer", file=sys.stderr)
        ok = False
    if sc.objective == "mntp" and best != 1:
        print("[calib] ⚠ an MNTP model but shift 0 is higher → differs from the paper's description. Following the data, shift 0 is used, but interpret the results with care",
              file=sys.stderr)
    if abs(m1 - m0) < 0.5:
        print("[calib] ⚠ the two settings differ by less than 0.5 nat → the data do not clearly show which is right", file=sys.stderr)
    sc.shift = forced_shift if forced_shift is not None else (best if sc.objective == "mntp" else 0)
    print(f"[calib] read-out position used for scoring: shift {sc.shift}"
          f"{' (user-specified)' if forced_shift is not None else ''}", file=sys.stderr)
    return ok


# ══════════════════════════════════════════════════════════════════════════
# 6) Implementation check (--selftest)
# ══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_selftest(sc, dtype_name, mask_batch, check_knowledge=True):
    ok = True
    tol = 1e-4 if dtype_name == "fp32" else 5e-2
    text = ("It was a hot morning. He opened the [MASK] and the milk was gone. "
            "He sighed. * sigh: 한숨 쉬다")
    alt = text.replace("한숨 쉬다", "전혀 다른 각주 내용입니다")
    choices = ["window", "fridge", "refrigerator door"]
    prefix, suffix = text.split("[MASK]")

    def line(t):
        print("\n" + "=" * 72 + f"\n  {t}\n" + "=" * 72)

    line("[Test A] option score = value computed by hand from the full logits")
    for ch in choices:
        sco, T, W, nb, _, c_lps, _ = measure(sc, prefix, ch, suffix, 2)
        part = partition(sc.tok, prefix, ch, suffix)
        ids, cp = part["ids"], part["choice_pos"]
        if sc.objective == "clm":
            full = sc.model(input_ids=torch.tensor([sc.bos + ids], device=sc.device)).logits[0].float().log_softmax(-1)
            ref = [full[len(sc.bos) + p - 1, ids[p]].item() for p in cp]
        else:
            base = sc.pre + ids + sc.post
            off = len(sc.pre)
            batch = torch.tensor([base] * len(cp), device=sc.device)
            for r, p in enumerate(cp):
                batch[r, off + p] = sc.tok.mask_token_id
            if sc.force_bi:   # the hand calculation also computes the full logits with the bidirectional mask
                hfull = sc.model.model(input_ids=batch, attention_mask=sc._bi_masks(len(cp), batch.shape[1])).last_hidden_state
                full = sc.logits_from_hidden(hfull).float().log_softmax(-1)
            else:
                full = sc.model(input_ids=batch).logits.float().log_softmax(-1)
            ref = [full[r, off + p - sc.shift, ids[p]].item() for r, p in enumerate(cp)]
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
    else:
        line("[Test B'] whether the whole-passage PLL is the same when the batch size (mask_batch) is changed")
        s1 = measure(sc, prefix, "fridge", suffix, 1, enc_full=True)[0]["full"]
        s3 = measure(sc, prefix, "fridge", suffix, 3, enc_full=True)[0]["full"]
        b_ok = abs(s1 - s3) < tol * 10
        ok &= b_ok
        print(f"  batch 1: {s1:+.5f} / batch 3: {s3:+.5f} {'✅' if b_ok else '❌'}")

    line("[Test E] whether footnotes are excluded from the B range")
    bt = partition(sc.tok, prefix, "fridge", suffix)["b_text"]
    e_ok = bt.strip().endswith("He sighed.") and "*" not in bt
    ok &= e_ok
    print(f"  B range = {bt!r} {'✅' if e_ok else '❌'}")

    if check_knowledge:
        line("[Test F] pre-training knowledge check — does the natural option win (decoder: full / encoder family: choice)")
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
# 7) Reading items · alignment with the student distribution  (same as v3)
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
    """Mean item-level Kendall τ_b between the students' distractor distribution (four, key excluded) and the model scores"""
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
# 8) Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--arch", default="dec", choices=list(ARCHS))
    ap.add_argument("--size", default="1b", choices=ETTIN_SIZES)
    ap.add_argument("--model", default=None, help="model path/name specified directly (--arch is used to check the expected type)")
    ap.add_argument("--outprefix", default=None, help="if empty, ettin_{type}{size}_P")
    ap.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mask-batch", type=int, default=None, help="encoder-family copy batch (if empty, 1b=4, otherwise=16)")
    ap.add_argument("--enc-full", action="store_true", help="also compute the whole-passage PLL for the encoder family (very slow; needed for the 2×2 comparison)")
    ap.add_argument("--mntp-shift", type=int, default=None, choices=[0, 1],
                    help="force the enc-from-dec read-out position (if empty, chosen automatically from the check results)")
    ap.add_argument("--students", default=None, help="metadata Excel with student selection rates (alignment τ_b computed immediately)")
    ap.add_argument("--check-against", default=None,
                    help="existing result (same model). regression check that the choice range gives the same values")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-knowledge-test", action="store_true", help=argparse.SUPPRESS)  # for development checks with random weights
    args = ap.parse_args()

    name = resolve_model_name(args.arch, args.size, args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and args.dtype != "fp32":
        args.dtype = "fp32"
    mb = args.mask_batch or (4 if re.search(r"-1b$", name) else 16)
    print(f"[info] model={name}, device={device}, dtype={args.dtype}, mask_batch={mb}", file=sys.stderr)
    sc = EttinScorer(name, device, DTYPES[args.dtype])
    if sc.objective != EXPECTED_OBJ[args.arch]:
        sys.exit(f"[error] --arch {args.arch} should be a {EXPECTED_OBJ[args.arch].upper()} model, but "
                 f"the loaded model is {sc.objective.upper()}")
    calib_ok = calibrate(sc, args.mntp_shift, mask_batch=mb)
    if args.selftest:
        ok = run_selftest(sc, args.dtype, mb, check_knowledge=not args.no_knowledge_test)
        sys.exit(0 if (ok and calib_ok) else 1)
    if not calib_ok:
        sys.exit("[error] check failed → scoring aborted (see the [calib] lines above)")
    if not args.input:
        ap.error("--input is required (or --selftest)")

    ranges = ["choice", "choice_sum", "context", "full"] if sc.objective == "clm" else \
        (["choice", "choice_sum", "full"] if args.enc_full else ["choice", "choice_sum"])
    outprefix = args.outprefix or f"ettin_{short_tag(name)}_P"
    obj_label = {"clm": "CLM", "mlm": "MLM"}.get(sc.objective, f"MNTP(shift={sc.shift},bidirectional)")
    first_tok = sc.bos or sc.pre
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
                   "Model": name, "Objective": obj_label, "BOS": "yes" if first_tok else "no",
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
                  f"→ {'✅ reproduced' if diff < 1e-3 else '⚠ needs checking (may reflect a different data version or answer corrections)'}",
                  file=sys.stderr)
        else:
            print("[check] no matched items (check Grade·Year·Month·Item_No)", file=sys.stderr)


if __name__ == "__main__":
    main()
