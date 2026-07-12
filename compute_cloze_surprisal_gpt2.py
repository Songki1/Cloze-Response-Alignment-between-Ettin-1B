# -*- coding: utf-8 -*-
"""
compute_cloze_surprisal_gpt2.py  (GPT-2판: Causal LM, 선지 구간 autoregressive)
==============================================================================
수능/모의고사 빈칸([MASK]) 문항의 각 선지를 GPT-2(openai-community/gpt2)에 끼워
넣고, '선지 구간'의 평균 autoregressive surprisal(bits)을 계산한 뒤 softmax로
백분율을 배분한다. CLI·열 자동감지·출력 형식은 BERT/RoBERTa판과 동일.

BERT/RoBERTa판과의 차이 (★중요)
--------------------------------
GPT-2는 MLM이 아니라 Causal LM(다음 단어 예측, 왼쪽 문맥만)이다.
  - 마스크 토큰이 없다. 선지를 끼운 완성 문장을 그대로 한 번 forward.
  - 위치 t의 logits는 '토큰 t+1'을 예측한다(shift). 따라서 선지 토큰 k의
    log-prob는 logits[k-1]에서 읽는다.
  - 선지 구간 토큰들의 log-prob만 평균 → 길이 정규화(BERT판과 동일 기준).
  - GPT-2도 BPE라 어두 공백(Ġ)을 토큰에 부호화 → RoBERTa판처럼 '완성 문장
    일괄 토큰화 + 오프셋'으로 선지 구간을 안전하게 식별한다.

실행:
  python compute_cloze_surprisal_gpt2.py \
      --input "20260623_csat_data_original.xlsx" \
      --output "cloze_surprisal_gpt2.xlsx"
"""

import argparse
import math
import re
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

LN2 = math.log(2.0)


# ──────────────────────────────────────────────────────────────────────────
# 0) 열 이름 자동 감지 (BERT/RoBERTa판과 동일)
# ──────────────────────────────────────────────────────────────────────────
def find_col(df, *subs):
    for c in df.columns:
        if all(sub in str(c) for sub in subs):
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
    cols = [c for c in df.columns if ("선택지" in str(c)) or ("선지" in str(c))]

    def keyf(c):
        m = re.search(r"(\d)", str(c))
        return int(m.group(1)) if m else 99

    return sorted(cols, key=keyf)[:5]


_CIRC = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5}


def parse_choices(row, choice_cols):
    slots = {}
    for col_idx, col in enumerate(choice_cols, start=1):
        c = row[col]
        if c is None or str(c) == "nan":
            continue
        s = str(c).strip()
        ms = [(m.start(), _CIRC[m.group()]) for m in re.finditer(r"[①②③④⑤]", s)]
        if not ms:
            slots.setdefault(col_idx, s)
            continue
        for i, (pos, num) in enumerate(ms):
            end = ms[i + 1][0] if i + 1 < len(ms) else len(s)
            slots.setdefault(num, s[pos + 1:end].strip())
    return [slots.get(i, "").strip() for i in range(1, 6)]


# ──────────────────────────────────────────────────────────────────────────
# 1) 한 선지의 평균 autoregressive surprisal — GPT-2(Causal LM)
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def choice_pll(model, tok, prefix, choice, suffix, device, max_len=1024):
    """[MASK]에 choice를 끼운 완성 문장을 forward하고, '선지 구간' 토큰들의
    다음단어예측 log-prob(nat)만 평균낸다. (shift: logits[k-1]가 토큰 k를 예측)"""
    full = prefix + choice + suffix
    c_start = len(prefix)
    c_end = len(prefix) + len(choice)

    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offs = enc["offset_mapping"]

    # 선지 문자구간에 걸치는 토큰 인덱스
    cpos = [i for i, (a, b) in enumerate(offs) if b > a and a < c_end and b > c_start]
    if not cpos:
        return float("-inf"), 0

    # 선지의 첫 토큰이 위치 0이면 그 앞 예측이 없어 surprisal을 못 구함.
    # 이런 경우는 prefix가 비었다는 뜻 → 거의 없으나 방어적으로 제외.
    cpos = [i for i in cpos if i >= 1]
    if not cpos:
        return float("-inf"), 0

    # 1024 초과 시: 선지 구간을 포함하도록 왼쪽 문맥을 자른다(왼쪽 모델이라
    # 선지 앞 문맥은 가까운 쪽을 보존; 선지 이후는 surprisal에 안 쓰므로 잘라도 됨)
    hi = max(cpos)
    if hi + 1 > max_len:
        # 끝(hi)을 기준으로 max_len 토큰만 남김
        new_start = hi - max_len + 1
        ids = ids[new_start:hi + 1]
        cpos = [i - new_start for i in cpos]

    input_ids = torch.tensor([ids], device=device)
    logits = model(input_ids).logits[0]          # (seq_len, vocab)
    logprobs = torch.log_softmax(logits, dim=-1)

    total = 0.0
    for p in cpos:
        # 토큰 p의 확률은 '바로 앞 위치(p-1)'의 예측 분포에서 읽는다(shift)
        total += logprobs[p - 1, ids[p]].item()
    return total / len(cpos), len(cpos)


# ──────────────────────────────────────────────────────────────────────────
# 2) softmax → 백분율 (동일)
# ──────────────────────────────────────────────────────────────────────────
def softmax_pct(mean_logprobs_nat, temperature=1.0):
    x = np.array(mean_logprobs_nat, dtype=float) / temperature
    x = x - np.max(x)
    w = np.exp(x)
    w = w / w.sum()
    return (w * 100.0).tolist()


# ──────────────────────────────────────────────────────────────────────────
# 3) 메인 (동일, 모델 기본값만 gpt2)
# ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="cloze_surprisal_gpt2.xlsx")
    ap.add_argument("--model", default="openai-community/gpt2")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}, model={args.model}", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(args.model)
    if not tok.is_fast:
        sys.exit("[error] 오프셋 매핑에는 fast 토크나이저가 필요합니다.")
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()

    xls = pd.ExcelFile(args.input)
    records = []
    seen = 0
    printed = False
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)

        text_col = detect_text_col(df)
        choice_cols = detect_choice_cols(df)
        gold_col = find_col(df, "정답", "숫자") or find_col(df, "정답")
        grade_col = find_col(df, "학년")
        year_col = find_col(df, "연도")
        month_col = find_col(df, "월")
        qno_col = find_col(df, "문항번호") or find_col(df, "문항", "번호")
        err_col = find_col(df, "오답")

        if not printed:
            print(f"[detect] 문항열={text_col!r}", file=sys.stderr)
            print(f"[detect] 선지열={choice_cols}", file=sys.stderr)
            print(f"[detect] 정답열={gold_col!r}", file=sys.stderr)
            printed = True

        if text_col is None or len(choice_cols) < 5:
            print(f"[warn] '{sheet}' 시트에서 문항/선지 열을 못 찾음 → 건너뜀",
                  file=sys.stderr)
            continue

        df = df.dropna(subset=[text_col])
        for _, row in df.iterrows():
            text = str(row[text_col])
            if text.count("[MASK]") != 1:
                continue
            prefix, suffix = text.split("[MASK]")
            choices = parse_choices(row, choice_cols)

            mean_lps, surps, ntoks = [], [], []
            for ch in choices:
                mlp, nt = choice_pll(model, tok, prefix, ch, suffix, device)
                mean_lps.append(mlp)
                surps.append(-mlp / LN2 if nt > 0 else float("inf"))
                ntoks.append(nt)

            pct = softmax_pct(mean_lps, args.temperature)
            model_pick = int(np.argmin(surps)) + 1
            gold = row[gold_col] if gold_col else None
            try:
                gold = int(gold)
            except (ValueError, TypeError):
                gold = None

            rec = {
                "학년": row[grade_col] if grade_col else sheet,
                "연도": row[year_col] if year_col else None,
                "월": row[month_col] if month_col else None,
                "문항번호": row[qno_col] if qno_col else None,
                "오답률": row[err_col] if err_col else None,
                "정답숫자": gold, "모델선택": model_pick,
                "정답여부": (model_pick == gold) if gold else None,
            }
            for i in range(5):
                rec[f"선지{i+1}"] = choices[i]
                rec[f"surprisal{i+1}"] = round(surps[i], 4)
                rec[f"pct{i+1}"] = round(pct[i], 2)
                rec[f"ntok{i+1}"] = ntoks[i]
            if gold and 1 <= gold <= 5:
                rec["정답백분율"] = round(pct[gold - 1], 2)
            records.append(rec)

            seen += 1
            if seen % 50 == 0:
                print(f"[info] processed {seen} items", file=sys.stderr)
            if args.limit and seen >= args.limit:
                break
        if args.limit and seen >= args.limit:
            break

    out = pd.DataFrame(records)
    out.to_excel(args.output, index=False)

    valid = out.dropna(subset=["정답여부"])
    if len(valid):
        acc = valid["정답여부"].mean()
        print(f"[result] N={len(valid)}  모델정확도(최저 surprisal=정답): {acc:.1%}",
              file=sys.stderr)
    print(f"[done] saved -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
