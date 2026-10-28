# -*- coding: utf-8 -*-
"""
analysis_alignment_stats.py — alignment statistics for the four Ettin 1B models (whole-passage scoring)

Computes every statistic reported in the paper from the saved model outputs and the student metadata:
  1) Accuracy and item-level Pearson r, Spearman rho and Kendall tau_b between students' selection
     rates and model probabilities over the four distractors (key excluded), averaged over items
  2) Error-alignment score alpha on items the model answered incorrectly, its random baseline,
     rank-wise match rates and exact binomial tests against 25%
  3) Pairwise model comparisons: paired t tests over items with Holm correction across the three
     correlation indices, d_z, bootstrap CIs, alpha on items both models got wrong, exact McNemar test
  4) 2 x 2 contrasts (pre-training objective x scoring-time prediction mode) and item-level tau_b
     correlations between models
  5) Robustness: items with no context after the blank, tau ratio and gap-closure bootstrap CIs,
     sensitivity (minimum detectable d_z)

Inputs (same folder):
  ettin_enc1b_P_full.xlsx, ettin_dec1b_P_full.xlsx, ettin_decfromenc1b_P_full.xlsx,
  ettin_encfromdec1b_P_full.xlsx, Metadata_AI_Reviewed_GPT2_Clark.xlsx (student selection rates)
Outputs:
  alignment_stats.json (all numbers), alignment_items_{model}.csv (item-level indices)

Run:
  python analysis_alignment_stats.py
Requires: numpy, pandas, scipy, openpyxl (statsmodels optional, for the sensitivity analysis)
"""
import json
import re

import numpy as np
import pandas as pd
from scipy import stats

FILES = {"enc": "ettin_enc1b_P_full.xlsx", "dec": "ettin_dec1b_P_full.xlsx",
         "decfromenc": "ettin_decfromenc1b_P_full.xlsx", "encfromdec": "ettin_encfromdec1b_P_full.xlsx"}
STUDENTS = "Metadata_AI_Reviewed_GPT2_Clark.xlsx"
KEY = ["Grade", "Year", "Month", "Item_No"]
N_BOOT = 10000


def norm(x):
    """Normalise key values so that 2026, 2026.0 and '2026' match; 'Grade 3' and 3 both become '3'."""
    s = str(x).strip()
    return str(int(float(s))) if re.fullmatch(r"\d+(\.0+)?", s) else s


def item_key(row):
    return (re.sub(r"\D", "", str(row["Grade"])), norm(row["Year"]), norm(row["Month"]), norm(row["Item_No"]))


# ── 1) Student distributions: items with numeric selection rates for all five options (715 items) ──
sheets = pd.read_excel(STUDENTS, sheet_name=None)
meta = pd.concat(sheets.values(), ignore_index=True)
meta = meta[meta["Grade"].notna()]
S_COLS = [f"Student Choice {j}" for j in range(1, 6)]
STUD, GOLD = {}, {}
for _, r in meta.iterrows():
    s = pd.to_numeric(r[S_COLS], errors="coerce").to_numpy(float)
    if np.isnan(s).any():
        continue                                   # no response data for this item
    STUD[item_key(r)] = s
    GOLD[item_key(r)] = int(r["Answer No"])
ITEMS = sorted(STUD)
print(f"items with student response data: {len(ITEMS)} of {len(meta)}")

# ── 2) Model scores: probabilities recomputed from surprisal (bits) to avoid ties from rounding ──
RAW = {m: pd.read_excel(f) for m, f in FILES.items()}
NO_B = {item_key(r): bool(r["B_No_Words"]) for _, r in RAW["dec"].iterrows()}


def load_scores(df):
    out = {}
    for _, r in df.iterrows():
        k = item_key(r)
        if k not in STUD:
            continue
        assert int(r["Answer_No"]) == GOLD[k], f"answer key mismatch for {k}"
        sur = np.array([r[f"surprisal{j}"] for j in range(1, 6)], float)
        logp = -sur * np.log(2)                    # nat
        p = np.exp(logp - logp.max())
        out[k] = (logp, p / p.sum())
    return out


SCORES = {m: load_scores(df) for m, df in RAW.items()}


def item_indices(m):
    """Item-level indices over the four distractors (key excluded)."""
    rows = []
    for k in ITEMS:
        s, c = STUD[k], GOLD[k] - 1
        logp, p = SCORES[m][k]
        d = [j for j in range(5) if j != c]
        sd, pd_, ld = s[d], p[d], logp[d]
        top = d[int(np.argmax(ld))]                # distractor the model prefers most
        student_rank = stats.rankdata(-sd, method="min")
        rows.append(dict(item=k,
                         correct=int(np.argmax(logp)) == c,
                         r=stats.pearsonr(sd, pd_)[0] if np.ptp(pd_) > 0 else np.nan,
                         rho=stats.spearmanr(sd, ld)[0],
                         tau=stats.kendalltau(sd, ld, variant="b")[0],
                         alpha=s[top] / sd.max(),
                         base=sd.mean() / sd.max(),
                         rank_of_top=int(student_rank[d.index(top)]),
                         pmax=p.max(), noB=NO_B[k]))
    return pd.DataFrame(rows).set_index("item")


R = {m: item_indices(m) for m in FILES}

# ── 3) Per-model summaries ──
out = {}
for m, X in R.items():
    E = X[~X.correct]                              # items answered incorrectly by the model
    ranks = E.rank_of_top.value_counts(normalize=True).reindex([1, 2, 3, 4], fill_value=0) * 100
    n_top = int((E.rank_of_top == 1).sum())
    out[m] = dict(
        n_items=len(X), accuracy=X.correct.mean(), n_correct=int(X.correct.sum()), n_error=len(E),
        r=X.r.mean(), rho=X.rho.mean(), tau=X.tau.mean(),
        se_r=stats.sem(X.r, nan_policy="omit"), se_rho=stats.sem(X.rho), se_tau=stats.sem(X.tau),
        tau_ci95=stats.t.interval(.95, len(X) - 1, X.tau.mean(), stats.sem(X.tau)),
        alpha=E.alpha.mean(), alpha_baseline_own=E.base.mean(),
        alpha_vs_baseline_p=stats.ttest_rel(E.alpha, E.base).pvalue,
        rank_match_pct=ranks.round(2).tolist(),
        top1_match_n=n_top, top1_binom_p=stats.binomtest(n_top, len(E), .25, "greater").pvalue,
        pmax_median=X.pmax.median())
out["random_baseline_alpha"] = float(np.mean([STUD[k][[j for j in range(5) if j != GOLD[k] - 1]].mean()
                                              / STUD[k][[j for j in range(5) if j != GOLD[k] - 1]].max() for k in ITEMS]))
out["tied_top_distractor_items"] = int(sum((lambda d: (d == d.max()).sum() > 1)(np.delete(STUD[k], GOLD[k] - 1)) for k in ITEMS))

print("\n=== Per model (n = %d items) ===" % len(ITEMS))
for m in FILES:
    o = out[m]
    print(f"{m:11s} acc={o['accuracy']:.4f}  r={o['r']:.4f} ({o['se_r']:.3f})  rho={o['rho']:.4f} ({o['se_rho']:.3f})  "
          f"tau={o['tau']:.4f} ({o['se_tau']:.3f})  alpha={o['alpha']:.4f} (own baseline {o['alpha_baseline_own']:.4f}, "
          f"p={o['alpha_vs_baseline_p']:.3f})  rank match %={o['rank_match_pct']}  top-1 binomial p={o['top1_binom_p']:.3f}")
print(f"random baseline alpha (all items) = {out['random_baseline_alpha']:.6f}; items with tied top distractor = {out['tied_top_distractor_items']}")


# ── 4) Pairwise comparisons ──
def holm(pvals):
    order = np.argsort(pvals); n = len(pvals); adj = np.empty(n); running = 0
    for i, j in enumerate(order):
        running = max(running, (n - i) * pvals[j]); adj[j] = min(1, running)
    return adj


rng = np.random.default_rng(0)
cmp = {}
for a, b in [("dec", "enc"), ("decfromenc", "enc"), ("dec", "encfromdec"),
             ("encfromdec", "enc"), ("dec", "decfromenc"), ("decfromenc", "encfromdec")]:
    x, y = R[a], R[b]; res = {}; ps = []
    for c in ["r", "rho", "tau"]:
        d = (x[c] - y[c]).dropna(); t = stats.ttest_1samp(d, 0)
        boot = [rng.choice(d.values, len(d)).mean() for _ in range(N_BOOT)]
        res[c] = dict(diff=d.mean(), t=t.statistic, df=len(d) - 1, p=t.pvalue, dz=d.mean() / d.std(ddof=1),
                      t_ci95=stats.t.interval(.95, len(d) - 1, d.mean(), stats.sem(d)),
                      boot_ci95=(np.percentile(boot, 2.5), np.percentile(boot, 97.5)))
        ps.append(t.pvalue)
    for c, h in zip(["r", "rho", "tau"], holm(ps)):
        res[c]["p_holm"] = h                       # Holm correction across the three indices
    both = (~x.correct) & (~y.correct); d = (x.alpha - y.alpha)[both]
    res["alpha_both_wrong"] = dict(n=int(both.sum()), a=x.alpha[both].mean(), b=y.alpha[both].mean(),
                                   p=stats.ttest_1samp(d, 0).pvalue)
    only_a = int((x.correct & ~y.correct).sum()); only_b = int((~x.correct & y.correct).sum())
    res["mcnemar"] = dict(only_a_correct=only_a, only_b_correct=only_b,
                          p=stats.binomtest(only_a, only_a + only_b, .5).pvalue)
    cmp[f"{a}-{b}"] = res
    print(f"\n=== {a} − {b} ===")
    for c in ["r", "rho", "tau"]:
        q = res[c]
        print(f"  {c:4s} {q['diff']:+.4f}  t({q['df']})={q['t']:.2f}  p={q['p']:.4f}  Holm p={q['p_holm']:.4f}  "
              f"d_z={q['dz']:.3f}  95% CI [{q['t_ci95'][0]:+.3f}, {q['t_ci95'][1]:+.3f}]")
    print(f"  alpha on items both got wrong: {res['alpha_both_wrong']}  |  McNemar: {res['mcnemar']}")
out["pairwise"] = cmp

# ── 5) 2 x 2 contrasts: E = masked prediction, D = next-token prediction; superscript = pre-training, scoring ──
#   EE = encoder, DE = decoder→encoder (enc-from-dec), ED = encoder→decoder (dec-from-enc), DD = decoder
two = {}
print("\n=== 2 x 2 contrasts (one-sample t over items; equivalent to repeated-measures ANOVA, F = t^2) ===")
for c in ["r", "rho", "tau"]:
    EE, DE, ED, DD = R["enc"][c], R["encfromdec"][c], R["decfromenc"][c], R["dec"][c]
    contrasts = {"scoring_mode": ((ED + DD) - (EE + DE)) / 2,
                 "pretraining": ((DE + DD) - (EE + ED)) / 2,
                 "interaction": ((DD - DE) - (ED - EE)) / 2}
    two[c] = {}
    for name, z in contrasts.items():
        z = z.dropna(); t = stats.ttest_1samp(z, 0)
        two[c][name] = dict(est=z.mean(), t=t.statistic, df=len(z) - 1, p=t.pvalue, dz=z.mean() / z.std(ddof=1),
                            ci95=stats.t.interval(.95, len(z) - 1, z.mean(), stats.sem(z)))
    print(c, {k: f"{v['est']:+.4f} (t={v['t']:.2f}, p={v['p']:.4f})" for k, v in two[c].items()})
out["two_by_two"] = two
T = pd.DataFrame({m: R[m].tau for m in R})
out["item_tau_correlations"] = T.corr().round(3).to_dict()
print("\nItem-level tau_b correlations between models:\n", T.corr().round(3))

# ── 6) Robustness ──
sub = {}
for m in R:
    X = R[m][R[m].noB]; sub[m] = dict(n=len(X), r=X.r.mean(), rho=X.rho.mean(), tau=X.tau.mean())
d_noB = (R["dec"].tau - R["enc"].tau)[R["dec"].noB]
d_B = (R["dec"].tau - R["enc"].tau)[~R["dec"].noB]
sub["dec-enc_tau_noB"] = dict(n=len(d_noB), diff=d_noB.mean(), p=stats.ttest_1samp(d_noB, 0).pvalue)
sub["dec-enc_tau_withB"] = dict(n=len(d_B), diff=d_B.mean(), p=stats.ttest_1samp(d_B, 0).pvalue)
out["no_context_after_blank"] = sub
print("\nItems with no words after the blank:", {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in sub.items()})

idx = np.arange(len(ITEMS)); rng = np.random.default_rng(0)
tau = {m: R[m].tau.values for m in R}


def boot_ratio(a, b):
    bs = []
    for _ in range(N_BOOT):
        s = rng.choice(idx, len(idx)); bs.append(tau[a][s].mean() / tau[b][s].mean())
    return tau[a].mean() / tau[b].mean(), np.percentile(bs, [2.5, 97.5]).tolist()


def boot_closure(cross, base, top):
    bs = []
    for _ in range(N_BOOT):
        s = rng.choice(idx, len(idx))
        bs.append((tau[cross][s] - tau[base][s]).mean() / (tau[top][s] - tau[base][s]).mean())
    return (tau[cross] - tau[base]).mean() / (tau[top] - tau[base]).mean(), np.percentile(bs, [2.5, 97.5]).tolist()


out["tau_ratio_dec_over_enc"] = boot_ratio("dec", "enc")
out["tau_ratio_decfromenc_over_enc"] = boot_ratio("decfromenc", "enc")
out["tau_ratio_encfromdec_over_dec"] = boot_ratio("encfromdec", "dec")
out["gap_closed_by_decfromenc"] = boot_closure("decfromenc", "enc", "dec")       # share of the enc→dec gap gained
out["gap_lost_by_encfromdec"] = boot_closure("encfromdec", "dec", "enc")         # share of the gap lost
for k in ["tau_ratio_dec_over_enc", "tau_ratio_decfromenc_over_enc", "tau_ratio_encfromdec_over_dec",
          "gap_closed_by_decfromenc", "gap_lost_by_encfromdec"]:
    est, ci = out[k]; print(f"{k}: {est:.3f}  bootstrap 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
print("(bootstrap intervals vary slightly in the last digit with the random seed and number of resamples)")

try:
    from statsmodels.stats.power import TTestPower
    pw = TTestPower()
    out["sensitivity_dz_80pct_power"] = float(np.asarray(pw.solve_power(nobs=len(ITEMS), alpha=.05, power=.8, alternative="two-sided")).ravel()[0])
    out["sensitivity_dz_50pct_power"] = float(np.asarray(pw.solve_power(nobs=len(ITEMS), alpha=.05, power=.5, alternative="two-sided")).ravel()[0])
    print(f"sensitivity: d_z = {out['sensitivity_dz_80pct_power']:.3f} (80% power), "
          f"{out['sensitivity_dz_50pct_power']:.3f} (50% power), two-sided alpha = .05, n = {len(ITEMS)}")
except ImportError:
    print("statsmodels not installed: sensitivity analysis skipped")

# ── 7) Save ──
json.dump(out, open("alignment_stats.json", "w"), default=float, indent=1)
for m in R:
    R[m].reset_index().assign(item=lambda d: d["item"].map(lambda k: " ".join(k))).to_csv(
        f"alignment_items_{m}.csv", index=False)
print("\n[done] alignment_stats.json, alignment_items_{model}.csv saved")
