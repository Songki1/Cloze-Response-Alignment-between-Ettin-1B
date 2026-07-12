# -*- coding: utf-8 -*-
"""
compute_cloze_surprisal_bert.py  (v2: 열 이름 자동 감지)
========================================================
수능/모의고사 빈칸([MASK]) 문항의 각 선지를 BERT(bert-base-uncased)에 끼워 넣고,
선지별 평균 PLL 기반 surprisal(bits)을 계산한 뒤 softmax로 백분율(%)을 배분한다.

v2 변경점: 엑셀 열 이름을 하드코딩하지 않고 '자동 감지'한다.
- 문항 열  : 셀에 "[MASK]"가 가장 많이 든 열을 자동 선택
- 선지 열  : 이름에 "선택지" 또는 "선지"가 든 열을 숫자 순으로 5개
- 정답 열  : 이름에 "정답"+"숫자"가 든 열(없으면 "정답" 포함 열)
- 메타 열  : 연도/월/문항번호/오답률을 부분일치로 탐색
→ 파일을 새로 저장해 열 이름이 미세하게 바뀌어도 깨지지 않는다.
"""

import argparse #argparse는 명령행 옵션, 인수 및 하위 명령을 구문 분석하는 Python 모듈.
import math #수학 관련 함수와 상수를 제공하는 Python 모듈. 자연로그 등을 해석해줌.
import re #정규표현식 모듈. 문자열 검색, 치환, 패턴 매칭 등을 수행하는 기능을 제공.
import sys #출력 스트림, 명령행 인수, 예외 처리 등과 관련된 기능을 제공하는 Python 모듈.

import numpy as np #numpy는 수치 계산을 위한 Python 라이브러리로, 다차원 배열 객체와 다양한 수학 함수를 제공.
import pandas as pd #pandas는 데이터 조작 및 분석을 위한 Python 라이브러리로, 테이블 형식의 데이터를 다루는 데 유용.
import torch #PyTorch는 딥러닝 모델을 구축하고 학습시키기 위한 Python 라이브러리로, 텐서 연산과 자동 미분 기능을 제공.
from transformers import AutoTokenizer, AutoModelForMaskedLM #transformers 라이브러리에서 사전 학습된 토크나이저와 마스크 언어 모델을 자동으로 불러오는 기능을 제공.

LN2 = math.log(2.0)


# ──────────────────────────────────────────────────────────────────────────
# 0) 열 이름 자동 감지 도우미 
# ──────────────────────────────────────────────────────────────────────────
##엑셀 표에서 원하는 열을 자동으로 찾아주는 함수
#엑셀 파일을 일단 df라고 부른다. df는 pandas의 DataFrame 객체로, 엑셀 데이터를 테이블 형식으로 다룰 수 있게 해준다.
def find_col(df, *subs): #*subs는 찾고 싶은 열 이름 조각들 #*subs는 인자를 여러 개 받을 수 있다는 뜻
    """열 이름에 subs의 모든 조각이 들어간 첫 열을 반환(없으면 None)."""
    for c in df.columns: #df.columns는 DataFrame의 열 이름들을 반환하는 속성
        s = str(c) #현재 열 이름 c를 문자열로 바꿔서 s에 넣는 코드(엑셀 열 이름에 가끔 숫자나 날짜처럼 문자열이 아닌 값이 있을 수 있으므로 안전하게 문자열로 변환)
        if all(sub in s for sub in subs): #subs 안에 들어 있는 모든 문자열 조각이 현재 열 이름 s 안에 들어 있으면 True.
            return c
    return None #모든 열을 다 확인했는데도 찾지 못하면 None을 반환


def detect_text_col(df): #일단 함수 이름은 detect_text_col(df)로 지정. df는 pandas DataFrame 객체.
    """★★셀에 '[MASK]'가 가장 많이 든 열 = 문항 열.★★"""
    best, col = 0, None #best는 '[MASK]'가 가장 많이 든 횟수, col은 그 열 이름을 저장하는 변수
    for c in df.columns:
        cnt = df[c].astype(str).str.contains("[MASK]", regex=False, na=False).sum() #df[c]는 현재 열 ##현재 열 c에서 [MASK]가 들어 있는 셀의 개수를 세어서 cnt에 저장
        #c데이터를 가지고 오고 .astype(str)는 그 데이터를 문자열로 변환.(엑셀에는 숫자나 날짜나 빈칸처럼 문자열이 아닌 값이 있을 수 있으므로 안전하게 문자열로 변환)
        #.str.contains("[MASK]", regex=False, na=False)는 문자열 안에 '[MASK]'가 포함되어 있는지 확인. regex=False는 정규표현식이 아닌 단순 문자열 검색을 의미.
        #즉. [MASK]를 정규표현식으로 해석하지 말고, 그냥 문자 그대로 찾아라.
        # na=False는 결측값(NaN)을 False로 처리.
        #.sum()은 True의 개수를 세어서 '[MASK]'가 포함된 셀의 총 개수를 반환.
        if cnt > best: #만약 cnt가 지금까지 찾은 best보다 크면, 즉 '[MASK]'가 더 많이 들어 있는 열을 찾으면
            best, col = cnt, c #best를 cnt로 업데이트하고, col을 현재 열 이름 c로 업데이트
    return col #모든 열을 확인한 후, '[MASK]'가 가장 많이 들어 있는 열 이름 col을 반환


def detect_choice_cols(df):
    """★★이름에 '선택지'/'선지'가 든 열을 숫자 순으로 정렬해 최대 5개.★★"""
    cols = [c for c in df.columns if ("선택지" in str(c)) or ("선지" in str(c))] #열 이름에 "선택지"가 들어 있거나 "선지"가 들어 있으면 cols에 추가

    def keyf(c): #열 이름에서 숫자를 추출하여 정렬 키로 사용. 만약 숫자가 없으면 99를 반환.
        m = re.search(r"(\d)", str(c)) #r"(\d)"는 숫자 하나를 찾으려는 정규표현식. 
        return int(m.group(1)) if m else 99

    return sorted(cols, key=keyf)[:5] #cols를 keyf 기준으로 정렬 , [:5]로 앞에서 5개만 가져오라는 뜻.(오지선다)


_CIRC = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5}

#엑셀 한 줄에서 선지 1~5를 꺼내서 깨끗한 리스트로 만드는 함수
def parse_choices(row, choice_cols): #row = 엑셀에서 한 문항에 해당하는 한 줄, choice_cols = ["선지1", "선지2", "선지3", "선지4", "선지5"]
    """선지 5개를 깨끗한 리스트로 반환(원형 숫자 제거 + 뭉친 셀 자동 분리)."""
    slots = {}
    for col_idx, col in enumerate(choice_cols, start=1): #enumerate()는 반복하면서 순번과 값을 같이 꺼내는 함수
        # col_idx는 1부터 시작하는 순번, col은 현재 열 이름 = 1 선지1, 2 선지2, 3 선지3, 4 선지4, 5 선지5로 만든다.
        c = row[col] #현재 열에서 실제 선지 내용을 가져오는 줄(선지 안의 내용)
        if c is None or str(c) == "nan": #ㅡㅡ엑셀에서 빈칸이면 
            continue #넘어가라
        s = str(c).strip() #현재 열의 내용을 문자열로 변환하고 양쪽 공백 제거, strip()은 앞뒤 빈칸을 잘라내는 함수
        ms = [(m.start(), _CIRC[m.group()]) for m in re.finditer(r"[①②③④⑤]", s)]
        if not ms:
            slots.setdefault(col_idx, s)
            continue
        for i, (pos, num) in enumerate(ms):
            end = ms[i + 1][0] if i + 1 < len(ms) else len(s)
            slots.setdefault(num, s[pos + 1:end].strip())
    return [slots.get(i, "").strip() for i in range(1, 6)]


# ──────────────────────────────────────────────────────────────────────────
# 1) 512 토큰 윈도잉
# ──────────────────────────────────────────────────────────────────────────
def fit_window(prefix_ids, choice_ids, suffix_ids, max_len=510): #prefix_ids: [MASK] 앞 문장 토큰, choice_ids: 선지 토큰, suffix_ids: [MASK] 뒤 문장 토큰
    total = len(prefix_ids) + len(choice_ids) + len(suffix_ids) #전체 토큰 길이 계산 #510 + [CLS] + [SEP] = 512
    if total <= max_len: 
        return prefix_ids, suffix_ids
    budget = max_len - len(choice_ids)
    ##아래는 안 쓰임
    if budget <= 0:  ###현실적으로 말도 안되지만 선지 토큰 길이가 510보다 길면 그냥 빈 리스트 반환.
        return [], []
    half = budget // 2 #//는 나눗셈 후 소수점 버리기
    new_prefix = prefix_ids[-(budget - half):] if (budget - half) > 0 else [] 
    #prefix 전체를 다 쓰지 말고, [MASK] 바로 앞에 가까운 마지막 부분들만 쓰겠다.
    #budget - half가 0보다 크면, prefix_ids에서 뒤에서부터 budget - half개만큼 가져오고, 아니면 빈 리스트 반환.
    new_suffix = suffix_ids[:half] if half > 0 else []
    return new_prefix, new_suffix


# ──────────────────────────────────────────────────────────────────────────
# 2) 한 선지의 평균 PLL 계산 //선지 하나를 빈칸에 넣고, 그 선지의 토큰들을 하나씩 [MASK]로 가려서 BERT가 얼마나 자연스럽게 예측하는지 계산한다.
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad() #학습하지 않겠다, 이미 학습된 모델을 그대로 쓰겠다. #@torch.no_grad()를 붙이면 메모리도 덜 쓰고 속도도 조금 빨라진다.
def choice_pll(model, tok, prefix_text, choice_text, suffix_text, device): #prefix_text: [MASK] 앞 문장, choice_text: 선지, suffix_text: [MASK] 뒤 문장
    #choice_pll()은 선지 하나의 평균 PLL 점수를 계산하는 함수
    mask_id = tok.mask_token_id #BERT는 [MASK]라는 글자를 그대로 보는 게 아니라, 숫자로 봐야 한다. 그래서 토크나이저에서 [MASK] 토큰의 ID를 가져온다.
    
    ##영어 텍스트를 BERT가 이해할 수 있는 숫자 리스트로 바꾸는 부분
    prefix_ids = tok.encode(prefix_text.rstrip(), add_special_tokens=False) #rtrip()은 오른쪽 공백 제거, add_special_tokens=False는 [CLS]와 [SEP]를 붙이지 말라는 뜻
    choice_ids = tok.encode(choice_text.strip(), add_special_tokens=False) #strip()은 양쪽 공백 제거, add_special_tokens=False는 [CLS]와 [SEP]를 붙이지 말라는 뜻
    suffix_ids = tok.encode(suffix_text.lstrip(), add_special_tokens=False) #lstrip()은 왼쪽 공백 제거, add_special_tokens=False는 [CLS]와 [SEP]를 붙이지 말라는 뜻
    if len(choice_ids) == 0: #선지 토큰이 없으면 PLL 계산을 할 수 없으므로, -inf와 0을 반환
        return float("-inf"), 0 #즉, 사실상 이 선지는 PLL 계산에서 제외된다.(0% 확률)
    prefix_ids, suffix_ids = fit_window(prefix_ids, choice_ids, suffix_ids) #BERT 입력이 512토큰을 넘지 않게 prefix와 suffix를 조정

    #BERT 입력을 구성하는 부분. BERT 입력은 [CLS] 문장 [SEP]
    base = [tok.cls_token_id] + prefix_ids + choice_ids + suffix_ids + [tok.sep_token_id]
    start = 1 + len(prefix_ids) #base 안에서 선지가 시작되는 위치를 찾는 코드, 시작은 [CLS] + prefix_ids 이후이므로 1 + len(prefix_ids)
    positions = list(range(start, start + len(choice_ids))) 
    #선지 토큰들이 base 안에서 어느 위치에 있는지 리스트로 만드는 코드, start부터 시작해서 선지 토큰 길이만큼 위치를 나열
    #start부터 시작해서 len(choice_ids)만큼 증가시키면서 위치를 리스트로 만든다. 예를 들어, start=5이고 choice_ids 길이가 3이면 positions = [5, 6, 7]이 된다.
    
    ###BERT 모델에 입력을 넣고, 각 위치에서의 로그 확률을 계산하는 부분
    batch = torch.tensor([base] * len(positions), device=device) #batch = base를 len(positions)만큼 반복해서 만든 텐서, 즉 선지 토큰 개수만큼 같은 입력을 복제
    for i, pos in enumerate(positions):  #★★★ PLL의 핵심. 예를 들어, 선지 토큰이 3개라면, 첫 번째 토큰을 [MASK]로 바꾸고, 두 번째 토큰을 [MASK]로 바꾸고, 세 번째 토큰을 [MASK]로 바꾸는 식으로 반복
        batch[i, pos] = mask_id
        #batch[i, pos] = mask_id는 batch 텐서에서 i번째 샘플의 pos 위치를 [MASK] 토큰 ID로 바꾸는 코드. 즉, 선지 토큰 하나씩을 [MASK]로 바꾸면서 BERT가 예측하도록 한다.
    
    ##BERT가 실제로 점수를 내고, 선지 하나의 평균 PLL 값을 계산하는 마지막 단계
    logits = model(batch).logits #BERT 모델에 batch(선지를 넣고 문단 1,2,3,4....)를 넣고, logits를 가져오는 코드. logits는 각 위치에서의 단어 예측 점수(로짓)이다.
    logprobs = torch.log_softmax(logits, dim=-1) #logits를 log_softmax로 변환하여 각 위치에서의 단어 예측 확률을 로그 확률로 바꾸는 코드. 
    #dim=-1은 마지막 차원(단어 차원)에서 softmax를 적용하라는 뜻.
    total = 0.0
    for i, pos in enumerate(positions):  #예를 들어 operates in isolation이 3토큰이라면, 
        #1번째: operates의 log probability, 2번째: in의 log probability, 3번째: isolation의 log probability 를 하나씩 꺼내서 total에 더한다.
        total += logprobs[i, pos, choice_ids[i]].item() ###★★★선지의 pseudo-log-likelihood 합계
    return total / len(positions), len(positions)
    # 평균 log probability를 반환. 즉, 선지 하나의 평균 PLL 값과 선지 토큰 개수를 반환한다.

# ──────────────────────────────────────────────────────────────────────────
# 3) softmax → 백분율
# ──────────────────────────────────────────────────────────────────────────
def softmax_pct(mean_logprobs_nat, temperature=1.0):
    x = np.array(mean_logprobs_nat, dtype=float) / temperature
    x = x - np.max(x)
    w = np.exp(x)
    w = w / w.sum()
    return (w * 100.0).tolist()


# ──────────────────────────────────────────────────────────────────────────
# 4) 메인 
# ──────────────────────────────────────────────────────────────────────────
##터미널에서 입력한 옵션을 받는 부분
def main():
    ap = argparse.ArgumentParser() ##argparse는 명령행 옵션, 인수 및 하위 명령을 구문 분석하는 Python 모듈.
    ap.add_argument("--input", required=True) #True는 필수 옵션(반드시 파일을 넣어라), False는 선택 옵션
    ap.add_argument("--output", default="Surprisal_Result.xlsx")
    ap.add_argument("--model", default="bert-base-uncased") 
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    #GPU가 있으면 GPU를 쓰고, 없으면 CPU를 쓰는 코드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}, model={args.model}", file=sys.stderr)


    #BERT를 불러오는 부분. AutoTokenizer와 AutoModelForMaskedLM을 사용하여 사전 학습된 BERT 모델과 토크나이저를 불러온다.
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model).to(device).eval()

    xls = pd.ExcelFile(args.input) #args.input은 터미널에서 입력한 엑셀 파일 이름
    records = [] #처음에는 빈 리스트로 시작, 나중에 각 문항의 결과를 딕셔너리 형태로 저장
    seen = 0 #처음에는 0으로 시작, 나중에 처리한 문항 수를 카운트
    printed = False 
    for sheet in xls.sheet_names: #xls라는 ExcelFile 객체가 원래 가지고 있는 sheet_names라는 속성을 꺼내라
        df = pd.read_excel(xls, sheet_name=sheet)

        # ── 열 자동 감지 ──
        text_col = detect_text_col(df)  #문항 열 = 셀에 '[MASK]'가 가장 많이 든 열. 즉, 문항 열을 자동으로 감지하는 함수 detect_text_col(df)를 호출하여 text_col에 저장
        choice_cols = detect_choice_cols(df)
        gold_col = find_col(df, "정답", "숫자") or find_col(df, "정답")
        grade_col = find_col(df, "학년")
        year_col = find_col(df, "연도")
        month_col = find_col(df, "월")
        qno_col = find_col(df, "문항번호") or find_col(df, "문항", "번호")
        err_col = find_col(df, "오답")

        #열 자동 감지 결과를 처음 한 번만 출력하는 부분
        if not printed: #아직 출력 안 했으니까
            print(f"[detect] 문항열={text_col!r}", file=sys.stderr) #이건 코드가 자동으로 찾은 열 이름들을 화면에 보여주는 부분
            print(f"[detect] 선지열={choice_cols}", file=sys.stderr) #file=sys.stderr 는 진행 상황 / 경고 / 로그 메시지로 출력
            print(f"[detect] 정답열={gold_col!r}", file=sys.stderr)
            #text_col = "문항"
            #choice_cols = ["선지1", "선지2", "선지3", "선지4", "선지5"]
            #gold_col = "정답숫자"
            printed = True
        
        #분석할 수 없는 시트를 건너뛰는 부분. 문항 열이 없거나 선지 열이 5개 미만이면 건너뛰도록 한다.
        if text_col is None or len(choice_cols) < 5: #문항 열(text_col)을 못 찾았거나, 선지 열(choice_cols)이 5개보다 적으면,
            print(f"[warn] '{sheet}' 시트에서 문항/선지 열을 못 찾음 → 건너뜀",
                  file=sys.stderr)
            continue

        df = df.dropna(subset=[text_col]) #"문항" 열에 비어 있는 행은 버려라. #dropna()는 pandas에서 비어 있는 값, 즉 NaN이 있는 행을 제거하는 메서드
        for _, row in df.iterrows(): #DataFrame의 행을 하나씩 꺼내는 반복문
            text = str(row[text_col]) #현재 문항 행에서 지문을 꺼내는 코드
            if text.count("[MASK]") != 1: #text.count("[MASK]")는 지문 안에 [MASK]가 몇 번 나오는지 세어 1개이면 계속
                continue
            prefix, suffix = text.split("[MASK]") #지문을 [MASK] 기준으로 앞부분과 뒷부분으로 나누는 코드
            choices = parse_choices(row, choice_cols) #현재 문항 row에서 선지 1~5를 꺼내서 정리하는 코드

            mean_lps, surps, ntoks = [], [], []
            for ch in choices:  #선지 하나씩 꺼내서 [MASK] 자리에 넣고 PLL/surprisal을 계산
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
