"""
Diagnostic: does the model know anything beyond "this provider bills a lot"?

Run this when ablation shows a single size/money column matching the full feature set.
It is the test that decides whether the project has a model or a sorted spreadsheet.

Four checks:

  1. CONDITIONAL LIFT (the decisive one). Split providers into strata of similar
     total_reimbursed. Inside a stratum every provider is roughly the same size, so a
     pure size-ranker has nothing left to say and scores ~0.5 ROC-AUC. Real behavioural
     signal survives.

  2. RANK AGREEMENT. Spearman correlation between the model score and total_reimbursed.
     Near 1.0 means the model IS the size ranking, whatever features it nominally uses.

  3. INCREMENTAL VALUE. Nested logistic models: size alone, versus size plus the other
     features. Compares out-of-fold PR-AUC so the gain is measured honestly.

  4. EXPOSURE DOUBLE-COUNTING. If P(fraud) is mostly size, then ranking by
     P(fraud) x amount is close to ranking by amount squared. Checks that directly,
     because it silently breaks the Top-K exposure strategy.

Usage:
    python src/size_confound.py                  # uses total_reimbursed
    python src/size_confound.py --size-col n_claims
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import (ALL_FEATURE_GROUPS, OUTPUTS, SEED, SIZE_FEATURES, TARGET, TRAIN_CSV,
                    build_pipeline, candidate_models, size_free_features)

N_STRATA = 10
MIN_POSITIVES_PER_STRATUM = 20   # below this, within-stratum ROC-AUC is noise, not evidence


def oof_scores(X, y, feats, model, n_splits=5):
    p = np.zeros(len(y))
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    for tr, va in cv.split(X, y):
        pipe = build_pipeline(feats, clone(model)).fit(X.iloc[tr][feats], y[tr])
        p[va] = pipe.predict_proba(X.iloc[va][feats])[:, 1]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size-col", default="total_reimbursed")
    ap.add_argument("--model", default=None, help="default: best available tree model")
    a = ap.parse_args()

    train = pd.read_csv(TRAIN_CSV, keep_default_na=False, na_values=[""])
    y = train[TARGET].to_numpy()
    # This diagnostic deliberately IGNORES config.INCLUDE_SIZE. Its whole purpose is to
    # compare a size-inclusive model against a size-free one, so it must build both
    # regardless of how the pipeline is currently configured. Using final_features() here
    # would make "full" collapse to "size-free" once the flag is turned off, and the
    # comparison would silently become size-free vs size-free.
    all_features = [c for g in ALL_FEATURE_GROUPS.values() for c in g]
    feats = [c for c in all_features if c in train.columns]
    other = [c for c in size_free_features() if c in train.columns]
    if a.size_col not in train.columns:
        raise SystemExit(f"--size-col '{a.size_col}' is not in {TRAIN_CSV.name}")
    size = train[a.size_col].to_numpy(dtype=float)

    models = candidate_models()
    name = a.model or ("XGBoost" if "XGBoost" in models else "RandomForest")
    model = models[name][0]
    print(f"Model: {name}   size column: {a.size_col}   n={len(train)}  fraud={y.mean():.1%}")

    print(f"  size-inclusive set: {len(feats)} features   size-free set: {len(other)} features")
    p_full = oof_scores(train, y, feats, model)
    p_size = oof_scores(train, y, [a.size_col], LogisticRegression(max_iter=2000))
    p_other = oof_scores(train, y, other, model)

    rep = {"model": name, "size_col": a.size_col, "n_providers": len(train),
           "fraud_rate": round(float(y.mean()), 4),
           "n_features_size_inclusive": len(feats), "n_features_size_free": len(other),
           "note": "This diagnostic ignores config.INCLUDE_SIZE by design."}

    # ---- 1. conditional lift within size strata --------------------------------
    strata = pd.qcut(pd.Series(size).rank(method="first"), N_STRATA, labels=False)
    rows = []
    for s in range(N_STRATA):
        m = strata == s
        ys, ps, pz = y[m.to_numpy()], p_full[m.to_numpy()], p_other[m.to_numpy()]
        if len(np.unique(ys)) < 2:
            rows.append({"stratum": s, "n": int(m.sum()), "n_fraud": int(ys.sum()),
                         "roc_auc_full": np.nan, "roc_auc_without_size": np.nan})
            continue
        rows.append({"stratum": s, "n": int(m.sum()), "n_fraud": int(ys.sum()),
                     "fraud_rate": round(float(ys.mean()), 3),
                     "roc_auc_full": round(float(roc_auc_score(ys, ps)), 3),
                     "roc_auc_without_size": round(float(roc_auc_score(ys, pz)), 3)})
    strat = pd.DataFrame(rows)
    print("1. CONDITIONAL LIFT - ranking quality INSIDE strata of similar size")
    print(strat.to_string(index=False))
    # A plain mean across strata treats a stratum with 2 fraud cases as equal evidence to one
    # with 254. It is not: ROC-AUC on a handful of positives is noise. Weight by positives,
    # and separately report only the strata large enough to carry evidence.
    ok = strat.dropna(subset=["roc_auc_full"])
    w = ok["n_fraud"].to_numpy(dtype=float)
    cond_unweighted = float(ok["roc_auc_full"].mean())
    cond = float(np.average(ok["roc_auc_full"], weights=w)) if w.sum() else np.nan
    cond_nosize = float(np.average(ok["roc_auc_without_size"], weights=w)) if w.sum() else np.nan
    solid = ok[ok["n_fraud"] >= MIN_POSITIVES_PER_STRATUM]
    cond_solid = float(solid["roc_auc_full"].mean()) if len(solid) else np.nan

    rep["within_stratum_roc_auc_weighted"] = round(cond, 4)
    rep["within_stratum_roc_auc_weighted_without_size"] = round(cond_nosize, 4)
    rep["within_stratum_roc_auc_unweighted"] = round(cond_unweighted, 4)
    rep[f"within_stratum_roc_auc_strata_with_ge{MIN_POSITIVES_PER_STRATUM}_positives"] = (
        round(cond_solid, 4) if not np.isnan(cond_solid) else None)
    rep["n_strata_with_enough_positives"] = int(len(solid))
    rep["pct_fraud_in_largest_size_decile"] = round(
        float(strat["n_fraud"].iloc[-1] / max(strat["n_fraud"].sum(), 1)), 4)

    print(f"\n   within-stratum ROC-AUC, weighted by fraud count: {cond:.3f}   <- use this one")
    print(f"     same, model trained WITHOUT {a.size_col}:        {cond_nosize:.3f}")
    print(f"     strata with >= {MIN_POSITIVES_PER_STRATUM} fraud cases ({len(solid)} of "
          f"{len(ok)}):        {cond_solid:.3f}")
    print(f"     unweighted mean (reported for reference only):  {cond_unweighted:.3f}")
    print(f"   (0.50 = no signal beyond size, 1.00 = perfect)")
    print(f"\n   {rep['pct_fraud_in_largest_size_decile']:.0%} of all fraud sits in the largest "
          f"size decile; small strata carry too few positives to score.")

    # ---- 2. rank agreement ------------------------------------------------------
    rho = float(spearmanr(p_full, size).statistic)
    rep["spearman_model_vs_size"] = round(rho, 4)
    print(f"\n2. RANK AGREEMENT   Spearman(model score, {a.size_col}) = {rho:.3f}")

    # ---- 3. incremental value -----------------------------------------------------
    ap_full = average_precision_score(y, p_full)
    ap_size = average_precision_score(y, p_size)
    ap_other = average_precision_score(y, p_other)
    rep |= {"pr_auc_size_only": round(float(ap_size), 4),
            "pr_auc_all_features": round(float(ap_full), 4),
            "pr_auc_without_size": round(float(ap_other), 4),
            "incremental_pr_auc_over_size": round(float(ap_full - ap_size), 4)}
    print(f"\n3. INCREMENTAL VALUE (out-of-fold PR-AUC)")
    print(f"   {a.size_col} alone             {ap_size:.4f}")
    print(f"   size-inclusive ({len(feats)} features) {ap_full:.4f}   ({ap_full - ap_size:+.4f})")
    print(f"   size-free ({len(other)} features)      {ap_other:.4f}")

    # ---- 4. exposure double-counting ------------------------------------------------
    exposure = train["exposure_amount"].to_numpy(dtype=float)
    rho_exp = float(spearmanr(p_full * exposure, exposure).statistic)
    rho_exp_sf = float(spearmanr(p_other * exposure, exposure).statistic)

    # A raw correlation here is NOT interpretable on its own. exposure_amount typically spans
    # 3-4 orders of magnitude, so multiplying it by ANY probability in [0,1] barely reorders it.
    # The null reference is the same correlation using random probabilities: anything at or
    # near that level means the model is contributing nothing to the exposure ranking.
    rng = np.random.default_rng(SEED)
    null = float(np.mean([spearmanr(rng.random(len(exposure)) * exposure, exposure).statistic
                          for _ in range(20)]))
    rep["spearman_expected_exposure_vs_exposure"] = round(rho_exp, 4)
    rep["spearman_expected_exposure_vs_exposure_size_free_model"] = round(rho_exp_sf, 4)
    rep["spearman_expected_exposure_null_random_p"] = round(null, 4)
    rep["exposure_log10_spread"] = round(float(np.log10(exposure.max() / max(exposure.min(), 1))), 2)

    print(f"\n4. EXPOSURE RANKING   Spearman(P(fraud) x amount, amount)")
    print(f"   null reference (RANDOM p): {null:.3f}   <- compare against this, not against 1.0")
    print(f"   size-inclusive model:      {rho_exp:.3f}")
    print(f"   size-free model:           {rho_exp_sf:.3f}")
    print(f"   amount spans {rep['exposure_log10_spread']:.1f} orders of magnitude, which is why "
          f"even random\n   probabilities score high here.")
    if rho_exp > null + 0.10:
        print("   The queue is ordered mainly by claim amount. A size-free model does NOT fix "
              "this:\n   the skew in amount does it, not the model. Use banded ranking "
              "(rank_topk.py) instead.")

    # ---- verdict ---------------------------------------------------------------------
    verdict = []
    if np.isnan(cond):
        verdict.append("Too few positives in every stratum to judge.")
    elif cond < 0.60:
        verdict.append(f"Within providers of similar {a.size_col}, the model barely ranks fraud "
                       f"above non-fraud (weighted ROC-AUC {cond:.2f}). It is a size ranker.")
    elif cond < 0.70:
        verdict.append(f"Weak but real signal beyond size (weighted within-stratum ROC-AUC "
                       f"{cond:.2f}).")
    else:
        verdict.append(f"Genuine behavioural signal beyond size (weighted within-stratum ROC-AUC "
                       f"{cond:.2f}). Size dominates the OVERALL ranking only because fraud is "
                       f"concentrated in large providers -- within a size band the model still "
                       f"discriminates, which is the operationally useful part.")
    if rho > 0.90:
        verdict.append(f"Model score is almost a monotone function of {a.size_col} (rho {rho:.2f}).")
    if ap_full - ap_size < 0.03:
        verdict.append(f"All 37 features add {ap_full - ap_size:+.3f} PR-AUC over one column. "
                       f"The extra complexity is not earning its place.")
    if rho_exp > null + 0.10:
        verdict.append(f"Exposure ranking is ordered mainly by claim amount "
                       f"(rho {rho_exp:.2f} vs random-p null {null:.2f}). A size-free model does "
                       f"not fix this -- the skew in amount does it. Use the banded strategy in "
                       f"rank_topk.py, and report the probability-ranked queue alongside.")
    rep["verdict"] = verdict

    print("\nVERDICT")
    for v in verdict:
        print(f"  - {v}")

    OUTPUTS.mkdir(exist_ok=True)
    strat.to_csv(OUTPUTS / "size_strata.csv", index=False)
    (OUTPUTS / "size_confound.json").write_text(json.dumps(rep, indent=2))
    print(f"\nSaved {OUTPUTS / 'size_confound.json'}")


if __name__ == "__main__":
    main()
