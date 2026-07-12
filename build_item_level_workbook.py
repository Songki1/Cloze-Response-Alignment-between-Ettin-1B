#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_item_level_workbook.py
============================
논문 부록 A.7이 약속하는 "수식 기반 스프레드시트"를 생성한다.
모든 통계량이 Excel 수식으로 들어가므로 리뷰어가 셀 단위로 검증할 수 있다.
하드코딩된 결과값은 하나도 없다.

Sheets
------
Data     : 715문항 × 문항 단위 전 계산 (전부 수식)
Summary  : 표 1 · 표 2 · 부록 A.5 집계 (전부 수식)
Notes    : 정의·관행·제외 문항 문서화
"""

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

BERT_XLSX = "/mnt/project/메타정보_AI_검수_완료_BERT.xlsx"
GPT_XLSX = "/mnt/project/메타정보_AI_검수_완료_GPT2.xlsx"
OUT = "/home/claude/repo/문항별_정합성_결과_논문기준.xlsx"

STU = [f"학생 {i}번 선택" for i in range(1, 6)]
BC_ = [f"BERT {i}번 선택" for i in range(1, 6)]
GC_ = [f"GPT {i}번" for i in range(1, 6)]


def load(path, mcols):
    xs = pd.ExcelFile(path)
    d = pd.concat([xs.parse(s).assign(학년구분=s) for s in xs.sheet_names],
                  ignore_index=True)
    for c in STU + mcols + ["정답 숫자"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


b = load(BERT_XLSX, BC_)
g = load(GPT_XLSX, GC_)
assert len(b) == len(g) == 761

# 두 파일의 문항 정렬이 동일한지 확인 (인덱스 기준 병합)
# 두 파일은 행 순서가 동일하다. 문항번호 셀 1건('31??????')만 문자열로 깨져 있어
# 학년/연도/월로 정렬을 확인하고, 학생 응답은 수치 동일성으로 검증한다.
key = ["학년구분", "연도", "월"]
assert (b[key].values == g[key].values).all(), "BERT/GPT 파일의 문항 순서가 다릅니다"
_sb = b[STU].apply(pd.to_numeric, errors="coerce")
_sg = g[STU].apply(pd.to_numeric, errors="coerce")
assert _sb.equals(_sg), "두 파일의 학생 선택률이 다릅니다"
b["문항번호"] = pd.to_numeric(b["문항번호"], errors="coerce").fillna(
    g["문항번호"]).astype(int)

ok = (b[STU].notna().all(axis=1) & b[BC_].notna().all(axis=1)
      & g[GC_].notna().all(axis=1) & b["정답 숫자"].notna())
b, g = b[ok].reset_index(drop=True), g[ok].reset_index(drop=True)
N = len(b)
assert N == 715, N

wb = Workbook()

# ══════════════════════════════════════════════════════════════════════
# Sheet: Data
# ══════════════════════════════════════════════════════════════════════
ws = wb.active
ws.title = "Data"

HEAD = (
    ["학년", "연도", "월", "문항번호", "정답 c_i"]                       # A-E
    + [f"학생 s_i{j}" for j in range(1, 6)]                              # F-J
    + [f"BERT P_i{j}" for j in range(1, 6)]                              # K-O
    + [f"GPT2 P_i{j}" for j in range(1, 6)]                              # P-T
    + [f"학생 오답 D{k}" for k in range(1, 5)]                            # U-X
    + [f"BERT 오답 D{k}" for k in range(1, 5)]                            # Y-AB
    + [f"GPT2 오답 D{k}" for k in range(1, 5)]                            # AC-AF
    + [f"순위 학생 D{k}" for k in range(1, 5)]                            # AG-AJ
    + [f"순위 BERT D{k}" for k in range(1, 5)]                            # AK-AN
    + [f"순위 GPT2 D{k}" for k in range(1, 5)]                            # AO-AR
    + ["Pearson BERT", "Spearman BERT", "Kendall BERT"]                  # AS-AU
    + ["Pearson GPT2", "Spearman GPT2", "Kendall GPT2"]                  # AV-AX
    + ["o_i BERT", "e_i BERT", "alpha_i BERT", "순위_i BERT"]             # AY-BB
    + ["o_i GPT2", "e_i GPT2", "alpha_i GPT2", "순위_i GPT2"]             # BC-BF
    + ["alpha_random_i"]                                                 # BG
)
ws.append(HEAD)

L = {n: get_column_letter(i + 1) for i, n in enumerate(
    ["gr", "yr", "mo", "no", "key",
     "s1", "s2", "s3", "s4", "s5",
     "b1", "b2", "b3", "b4", "b5",
     "g1", "g2", "g3", "g4", "g5",
     "sD1", "sD2", "sD3", "sD4",
     "bD1", "bD2", "bD3", "bD4",
     "gD1", "gD2", "gD3", "gD4",
     "rs1", "rs2", "rs3", "rs4",
     "rb1", "rb2", "rb3", "rb4",
     "rg1", "rg2", "rg3", "rg4",
     "pB", "spB", "kB", "pG", "spG", "kG",
     "oB", "eB", "aB", "rkB",
     "oG", "eG", "aG", "rkG", "arnd"])}


def tau(sd, md, r):
    """tau_a = (n_c - n_d)/6.  SIGN 곱의 합 = n_c - n_d (동점쌍은 0)."""
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    terms = [f"SIGN({sd[i]}{r}-{sd[j]}{r})*SIGN({md[i]}{r}-{md[j]}{r})"
             for i, j in pairs]
    return "=(" + "+".join(terms) + ")/6"


def avgrank(cell, rng, r):
    """RANK.AVG 등가식 (COUNTIF 기반, 동점에 평균순위 부여)."""
    return (f"=1+COUNTIF({rng},\">\"&{cell}{r})"
            f"+(COUNTIF({rng},\"=\"&{cell}{r})-1)/2")


for i in range(N):
    r = i + 2
    rb, rg = b.iloc[i], g.iloc[i]
    row = [rb["학년구분"], rb["연도"], rb["월"], rb["문항번호"],
           int(rb["정답 숫자"])]
    row += [float(rb[c]) for c in STU]
    row += [float(rb[c]) for c in BC_]
    row += [float(rg[c]) for c in GC_]
    ws.append(row)

    S = f"${L['s1']}{r}:${L['s5']}{r}"
    B = f"${L['b1']}{r}:${L['b5']}{r}"
    G = f"${L['g1']}{r}:${L['g5']}{r}"
    K = f"${L['key']}{r}"

    # 오답 4개 추출: k번째 비정답 선지의 인덱스 = IF(k < c_i, k, k+1)
    for k in range(1, 5):
        idx = f"IF({k}<{K},{k},{k+1})"
        ws[f"{L[f'sD{k}']}{r}"] = f"=INDEX({S},{idx})"
        ws[f"{L[f'bD{k}']}{r}"] = f"=INDEX({B},{idx})"
        ws[f"{L[f'gD{k}']}{r}"] = f"=INDEX({G},{idx})"

    SD = f"${L['sD1']}{r}:${L['sD4']}{r}"
    BD = f"${L['bD1']}{r}:${L['bD4']}{r}"
    GD = f"${L['gD1']}{r}:${L['gD4']}{r}"

    # 평균순위 (식 A.4.2 구현 주석과 일치)
    for k in range(1, 5):
        ws[f"{L[f'rs{k}']}{r}"] = avgrank(L[f"sD{k}"], SD, r)
        ws[f"{L[f'rb{k}']}{r}"] = avgrank(L[f"bD{k}"], BD, r)
        ws[f"{L[f'rg{k}']}{r}"] = avgrank(L[f"gD{k}"], GD, r)

    RS = f"${L['rs1']}{r}:${L['rs4']}{r}"
    RB = f"${L['rb1']}{r}:${L['rb4']}{r}"
    RG = f"${L['rg1']}{r}:${L['rg4']}{r}"

    # A.4.1 Pearson / A.4.2 Spearman / A.4.3 Kendall tau_a
    ws[f"{L['pB']}{r}"] = f'=IFERROR(CORREL({SD},{BD}),"")'
    ws[f"{L['spB']}{r}"] = f'=IFERROR(CORREL({RS},{RB}),"")'
    ws[f"{L['kB']}{r}"] = tau([L[f"sD{k}"] for k in range(1, 5)],
                              [L[f"bD{k}"] for k in range(1, 5)], r)
    ws[f"{L['pG']}{r}"] = f'=IFERROR(CORREL({SD},{GD}),"")'
    ws[f"{L['spG']}{r}"] = f'=IFERROR(CORREL({RS},{RG}),"")'
    ws[f"{L['kG']}{r}"] = tau([L[f"sD{k}"] for k in range(1, 5)],
                              [L[f"gD{k}"] for k in range(1, 5)], r)

    # 모델 선택 o_i / 오류지표 e_i / 정렬점수 alpha_i / 최소순위
    for tag, rng in [("B", B), ("G", G)]:
        o = f"{L['o'+tag]}{r}"
        ws[o] = f"=MATCH(MAX({rng}),{rng},0)"
        ws[f"{L['e'+tag]}{r}"] = f"=IF({o}={K},0,1)"          # A.6.1
        so = f"INDEX({S},{o})"
        ws[f"{L['a'+tag]}{r}"] = (                              # A.6.2
            f'=IF({L["e"+tag]}{r}=1,{so}/MAX({SD}),"")')
        ws[f"{L['rk'+tag]}{r}"] = (   # 경쟁(최소) 순위: 1 + 자기보다 큰 값의 수
            f'=IF({L["e"+tag]}{r}=1,1+COUNTIF({SD},">"&{so}),"")')

    ws[f"{L['arnd']}{r}"] = f"=AVERAGE({SD})/MAX({SD})"         # A.6.4

# ══════════════════════════════════════════════════════════════════════
# Sheet: Summary
# ══════════════════════════════════════════════════════════════════════
sm = wb.create_sheet("Summary")
last = N + 1
D = "Data!"


def col(c):
    return f"{D}${L[c]}$2:${L[c]}${last}"


sm.append(["표 1. 오답 분포 상관 (정답 제외, 오답 4개 기준)"])
sm.append(["모델", "Pearson r̄", "Spearman ρ̄", "Kendall τ̄_a", "n"])
sm.append(["BERT-base",
           f"=AVERAGE({col('pB')})", f"=AVERAGE({col('spB')})",
           f"=AVERAGE({col('kB')})", f"=COUNT({col('pB')})"])
sm.append(["GPT-2",
           f"=AVERAGE({col('pG')})", f"=AVERAGE({col('spG')})",
           f"=AVERAGE({col('kG')})", f"=COUNT({col('pG')})"])
sm.append([])

sm.append(["표 2. 오류 시 정렬점수 α 및 독자 오답 순위별 일치율"])
sm.append(["모델", "n_err", "ᾱ", "1순위(%)", "2순위(%)", "3순위(%)",
           "4순위(%)", "1순위 문항수", "합계(%)"])
for name, tag in [("BERT-base", "B"), ("GPT-2", "G")]:
    e, a, rk = col("e" + tag), col("a" + tag), col("rk" + tag)
    nerr = f"SUM({e})"
    pct = [f"=COUNTIF({rk},{k})/{nerr}*100" for k in range(1, 5)]
    sm.append([name, f"={nerr}", f"=AVERAGE({a})"] + pct
              + [f"=COUNTIF({rk},1)", None])
    r = sm.max_row
    sm.cell(r, 9).value = f"=SUM(D{r}:G{r})"
sm.append(["무작위 기준선", None, f"=AVERAGE({col('arnd')})",
           25, 25, 25, 25, None, 100])
sm.append([])

sm.append(["부록 A.5. 문항 난이도와 모델 정답확률의 상관 R"])
sm.append(["모델", "R"])
# 오답률 u_i = 100 - 정답 선택률 ;  모델 정답확률 P_i,c_i
for name, pre in [("BERT-base", "b"), ("GPT-2", "g")]:
    u = ",".join(
        f"100-INDEX({D}${L['s1']}${i+2}:${L['s5']}${i+2},{D}${L['key']}${i+2})"
        for i in range(0, 0))  # placeholder, 실제 값은 아래 헬퍼 열 사용
sm.append(["BERT-base", "=CORREL(Helper!$A$2:$A$716,Helper!$B$2:$B$716)"])
sm.append(["GPT-2", "=CORREL(Helper!$A$2:$A$716,Helper!$C$2:$C$716)"])
sm.append([])
sm.append(["검산: 표 2의 각 행은 합계(%)가 정확히 100.0이어야 한다."])
sm.append(["검산: 표 1의 n은 BERT 714(영분산 1문항 제외), GPT-2 715이어야 한다."])

# Helper sheet (A.5 전용, 역시 수식)
hp = wb.create_sheet("Helper")
hp.append(["오답률 u_i", "BERT P(정답)", "GPT2 P(정답)"])
for i in range(N):
    r = i + 2
    S = f"{D}${L['s1']}{r}:${L['s5']}{r}"
    B = f"{D}${L['b1']}{r}:${L['b5']}{r}"
    G = f"{D}${L['g1']}{r}:${L['g5']}{r}"
    K = f"{D}${L['key']}{r}"
    hp[f"A{r}"] = f"=100-INDEX({S},{K})"
    hp[f"B{r}"] = f"=INDEX({B},{K})"
    hp[f"C{r}"] = f"=INDEX({G},{K})"

# ══════════════════════════════════════════════════════════════════════
# Sheet: Notes
# ══════════════════════════════════════════════════════════════════════
nt = wb.create_sheet("Notes")
for line in [
    ["문항별 정합성 결과 — 논문 기준 (부록 A 수식 구현)"],
    [],
    ["■ 이 파일의 모든 통계량은 Excel 수식으로 계산되며 하드코딩된 값이 없다."],
    ["  Data 시트의 U열 이후는 전부 수식이므로 셀을 클릭하여 검증할 수 있다."],
    [],
    ["■ 산출 범위"],
    ["  표 1·표 2는 정답(c_i)을 제외한 오답 4개만을 대상으로 한다."],
    ["  이는 모델의 정답 식별력이 아니라 오답 간 상대적 매력도 구조를 보기 위함이다."],
    ["  U~X열(학생 오답), Y~AB열(BERT 오답), AC~AF열(GPT2 오답)이 그 4개이다."],
    ["  k번째 비정답 선지의 인덱스는 IF(k < c_i, k, k+1)로 얻는다."],
    [],
    ["■ 식 대응 (논문 부록 A)"],
    ["  A.4.1 Pearson   → AS/AV열: CORREL(오답 학생값, 오답 모델값)"],
    ["  A.4.2 Spearman  → AT/AW열: 평균순위(AG~AR열) 간 CORREL"],
    ["    평균순위는 RANK.AVG 등가식 1+COUNTIF(>x)+(COUNTIF(=x)-1)/2 로 구현"],
    ["  A.4.3 Kendall   → AU/AX열: tau_a = (n_c - n_d)/6"],
    ["    6개 쌍의 SIGN 곱의 합이 곧 n_c - n_d 이며, 동점쌍은 0으로 기여한다."],
    ["    동점 보정을 하지 않는 tau_a 이므로 분모는 |D_i|(|D_i|-1)=12 로 고정된다."],
    ["  A.6.1 e_i       → AZ/BD열"],
    ["  A.6.2 alpha_i   → BA/BE열: INDEX(학생,o_i) / MAX(학생 오답 4개)"],
    ["  A.6.4 무작위    → BG열: AVERAGE(오답 4개) / MAX(오답 4개)"],
    ["  A.5   R         → Summary 시트, Helper 시트 참조"],
    [],
    ["■ 동점 처리 관행"],
    ["  순위_i(BB/BF열)는 경쟁 순위(min-rank): 1 + COUNTIF(오답4개, \">\" & 선택값)"],
    ["  즉 학생 선택률이 동점인 오답은 상위 순위를 공유한다."],
    ["  이는 alpha의 분모가 max(값)이지 argmax(인덱스)가 아니라는 정의(A.6.2)와"],
    ["  정합적이며, 선지 배열 순서에 불변이다. 본 자료에 해당 동점 문항은 2건이다."],
    [],
    ["■ 제외 문항"],
    ["  고3 2015년 6월 31번은 BERT 확률분포가 [0,0,0,100,0]로 오답 4개의 분산이"],
    ["  0이어서 상관계수가 정의되지 않는다. IFERROR로 공란 처리되며 AVERAGE/COUNT"],
    ["  에서 자동 제외되므로 BERT의 n은 714, GPT-2는 715가 된다."],
    [],
    ["■ 학생 선택률"],
    ["  합이 평균 94.94%(최소 73.80%)로 100%가 아니다. 미기록 응답이 있기 때문이다."],
    ["  본 파일의 모든 지표는 양의 상수배에 불변이므로 원값을 그대로 사용한다."],
]:
    nt.append(line)

# ── 서식 ──
hdr_fill = PatternFill("solid", fgColor="1F3864")
hdr_font = Font(name="Arial", bold=True, color="FFFFFF", size=9)
thin = Side(style="thin", color="BFBFBF")
for c in range(1, len(HEAD) + 1):
    cell = ws.cell(1, c)
    cell.fill, cell.font = hdr_fill, hdr_font
    cell.alignment = Alignment(horizontal="center", wrap_text=True)
    ws.column_dimensions[get_column_letter(c)].width = 13
ws.freeze_panes = "F2"
for row in ws.iter_rows(min_row=2, max_row=last):
    for cell in row:
        cell.font = Font(name="Arial", size=9)
        cell.border = Border(bottom=thin)
        if cell.column >= 21:
            cell.number_format = "0.0000"

for sh in (sm, hp, nt):
    for row in sh.iter_rows():
        for cell in row:
            cell.font = Font(name="Arial", size=10)
for c in "ABCDEFGHI":
    sm.column_dimensions[c].width = 16
sm["A1"].font = Font(name="Arial", bold=True, size=11)
sm["A6"].font = Font(name="Arial", bold=True, size=11)
sm["A11"].font = Font(name="Arial", bold=True, size=11)
nt.column_dimensions["A"].width = 100
nt["A1"].font = Font(name="Arial", bold=True, size=12)

wb.save(OUT)
print("saved:", OUT, "| rows:", N)
