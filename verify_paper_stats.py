#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_paper_stats.py
=====================
Reproduces every figure reported in the paper (Tables 1 and 2, and the
appendix statistics) from the item-level Excel files, following the formulas
in Appendix A. Equation numbers below refer to the appendix.

논문 표 1·표 2와 부록 통계 전부를 문항 단위 엑셀 원자료에서 재계산한다.
아래 함수의 식 번호는 부록 A의 수식 번호와 대응한다.

Usage:
    python verify_paper_stats.py --bert 메타정보_AI_검수_완료_BERT.xlsx \
                                 --gpt2 메타정보_AI_검수_완료_GPT2.xlsx

Requires: pandas, numpy, scipy, openpyxl
"""

import argparse
import itertools

import numpy as np
import pandas as pd
from scipy import stats

STUDENT_COLS = [f"학생 {i}번 선택" for i in range(1, 6)]
KEY_COL = "정답 숫자"


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────
def load(path: str, model_cols: list[str]) -> pd.DataFrame:
    """Concatenate the Grade 1/2/3 sheets and keep the 715 analyzable items
    (student distribution, model distribution and key all present)."""
    xs = pd.ExcelFile(path)
    df = pd.concat(
        [xs.parse(sh).assign(sheet=sh) for sh in xs.sheet_names],
        ignore_index=True,
    )
    for c in STUDENT_COLS + model_cols + [KEY_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    ok = (
        df[STUDENT_COLS].notna().all(axis=1)
        & df[model_cols].notna().all(axis=1)
        & df[KEY_COL].notna()
    )
    return df[ok].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────
# Correlation primitives (Eqs. A.4.1–A.4.3)
# ──────────────────────────────────────────────────────────────────────
def pearson(a, b):
    """Eq. (A.4.1). Returns NaN when either vector has zero variance."""
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    da, db = np.sqrt((a**2).sum()), np.sqrt((b**2).sum())
    if da == 0 or db == 0:
        return np.nan
    return float((a * b).sum() / (da * db))


def average_rank(v):
    """Average ranks (ties share the mean rank) — equivalent to Excel
    RANK.AVG. Used for Spearman, Eq. (A.4.2)."""
    v = np.asarray(v, float)
    return np.array(
        [1 + np.sum(v < x) + (np.sum(v == x) - 1) / 2 for x in v]
    )


def spearman(a, b):
    """Eq. (A.4.2): Pearson correlation of the average-rank vectors.
    Coincides with the general (tie-robust) definition of Spearman's rho."""
    return pearson(average_rank(a), average_rank(b))


def kendall_tau_a(a, b):
    """Eq. (A.4.3): tau_a, i.e. WITHOUT tie correction. Tied pairs count
    towards neither n_c nor n_d; the denominator stays |D|(|D|-1)."""
    n = len(a)
    nc = nd = 0
    for i, j in itertools.combinations(range(n), 2):
        s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
        if s > 0:
            nc += 1
        elif s < 0:
            nd += 1
    return 2 * (nc - nd) / (n * (n - 1))


# ──────────────────────────────────────────────────────────────────────
# Table 1  (RQ1): distractor-only correlations
# ──────────────────────────────────────────────────────────────────────
def table1(df: pd.DataFrame, model_cols: list[str], label: str):
    P, S, K, dropped = [], [], [], []
    for _, r in df.iterrows():
        key = int(r[KEY_COL])
        dis = [j for j in range(1, 6) if j != key]
        s = np.array([r[f"학생 {j}번 선택"] for j in dis], float)
        m = np.array([r[model_cols[j - 1]] for j in dis], float)
        p = pearson(s, m)
        if np.isnan(p):  # zero-variance model distribution → undefined
            dropped.append((r["sheet"], r["연도"], r["월"], r["문항번호"]))
            continue
        P.append(p)
        S.append(spearman(s, m))
        K.append(kendall_tau_a(s, m))
    print(f"[Table 1] {label}: n = {len(P)}"
          + (f"  (excluded: {dropped})" if dropped else ""))
    print(f"   Pearson  r̄  = {np.mean(P):.6f}  → paper reports {np.mean(P):.3f}")
    print(f"   Spearman ρ̄  = {np.mean(S):.6f}  → paper reports {np.mean(S):.3f}")
    print(f"   Kendall  τ̄a = {np.mean(K):.6f}  → paper reports {np.mean(K):.3f}")


# ──────────────────────────────────────────────────────────────────────
# Table 2  (RQ2): error alignment score and rank match rates
# ──────────────────────────────────────────────────────────────────────
def table2(df: pd.DataFrame, model_cols: list[str], label: str):
    alphas, rand_all = [], []
    rank_counts = [0, 0, 0, 0]  # k = 1..4, competition (min) ranking
    n_correct = 0
    for _, r in df.iterrows():
        key = int(r[KEY_COL])
        dis = [j for j in range(1, 6) if j != key]
        s = np.array([r[f"학생 {j}번 선택"] for j in dis], float)
        m_all = np.array([r[model_cols[j - 1]] for j in range(1, 6)], float)

        # Random baseline, Eq. (A.6.4): computed over ALL items
        rand_all.append(s.mean() / s.max())

        # Model's selection o_i = argmax of the softmax distribution
        o = int(np.argmax(m_all)) + 1
        if o == key:  # e_i = 0, Eq. (A.6.1)
            n_correct += 1
            continue

        s_o = float(r[f"학생 {o}번 선택"])
        alphas.append(s_o / s.max())  # Eq. (A.6.2)

        # Competition (minimum) rank of the chosen distractor among
        # student selection rates: 1 + (number strictly greater).
        # Tied distractors share the higher rank, consistent with the
        # max-VALUE denominator of Eq. (A.6.2).
        k = 1 + int(np.sum(s > s_o))
        rank_counts[min(k, 4) - 1] += 1

    n_err = len(alphas)
    alpha_bar = np.mean(alphas)                    # Eq. (A.6.3)
    baseline = np.mean(rand_all)                   # Eq. (A.6.4)
    binom = stats.binomtest(rank_counts[0], n_err, 0.25, alternative="greater")

    print(f"[Table 2] {label}: n_err = {n_err} "
          f"(model correct on {n_correct}/{len(df)})")
    print(f"   ᾱ         = {alpha_bar:.6f}   (paper: {alpha_bar:.3f})")
    print(f"   ᾱ_random  = {baseline:.6f}   (paper: {baseline:.3f})")
    pct = [c / n_err * 100 for c in rank_counts]
    print("   rank match k=1..4: "
          + "  ".join(f"{p:.1f}% ({c})" for p, c in zip(pct, rank_counts))
          + f"   Σ = {sum(pct):.1f}%")
    print(f"   binomial test (k=1 vs 25%): p = {binom.pvalue:.4f}")


# ──────────────────────────────────────────────────────────────────────
# Appendix A.5: item difficulty vs model probability of the key
# ──────────────────────────────────────────────────────────────────────
def appendix_a5(df: pd.DataFrame, model_cols: list[str], label: str):
    u, p_key = [], []
    for _, r in df.iterrows():
        key = int(r[KEY_COL])
        u.append(100.0 - float(r[f"학생 {key}번 선택"]))   # error rate
        p_key.append(float(r[model_cols[key - 1]]))
    R = np.corrcoef(u, p_key)[0, 1]
    print(f"[A.5] {label}: R = corr(error rate, model P(key)) = {R:+.4f}")


# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bert", required=True, help="BERT metadata xlsx")
    ap.add_argument("--gpt2", required=True, help="GPT-2 metadata xlsx")
    args = ap.parse_args()

    bert_cols = [f"BERT {i}번 선택" for i in range(1, 6)]
    gpt_cols = [f"GPT {i}번" for i in range(1, 6)]

    for label, path, cols in [
        ("BERT-base", args.bert, bert_cols),
        ("GPT-2", args.gpt2, gpt_cols),
    ]:
        df = load(path, cols)
        print("=" * 64)
        table1(df, cols, label)
        table2(df, cols, label)
        appendix_a5(df, cols, label)


if __name__ == "__main__":
    main()
