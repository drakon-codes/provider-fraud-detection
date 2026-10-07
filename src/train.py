"""
Step 5: compare models against rule baselines, select one, evaluate ONCE on the test set.

Principles enforced here:
  * Rule baselines go through exactly the same CV. A model that cannot beat
    "rank providers by total reimbursed" has not earned its complexity.
  * No resampling. Parts 6 and 7 multiply probabilities by money, so probabilities
    must stay calibrated; SMOTE inflates them.
  * Repeated stratified CV, reporting mean AND standard deviation.
  * Selection by the one-standard-error rule: the simplest model within 1 SE of the best.
  * Out-of-fold probabilities are saved, so the threshold and ranking steps never
    touch the test set.
  * Bootstrap confidence intervals on every test metric.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import precision_recall_curve

from config import (ID_COL, INCLUDE_VOLUME, MODELS, OUTPUTS, RULE_BASELINES, RULE_COLUMNS, SEED,
                    TARGET, TEST_CSV, TRAIN_CSV, build_pipeline, candidate_models, cv_splitter,
                    final_features, threshold_free_metrics, threshold_metrics)

N_SPLITS, N_REPEATS = 5, 3


def read(path):
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def spec(name, baselines, ml_models, default_feats):
    if name in baselines:
        return baselines[name], RULE_COLUMNS
    model, override = ml_models[name]
    return model, (override or default_feats)


def estimator(model, feats):
    return clone(model) if hasattr(model, "name") else build_pipeline(feats, clone(model))


def max_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[int(np.nanargmax(f1[:-1]))])


def bootstrap_ci(y, p, threshold, n=2000):
    rng, out = np.random.default_rng(SEED), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        out.append(threshold_free_metrics(y[i], p[i]) | threshold_metrics(y[i], p[i], threshold))
    b = pd.DataFrame(out)
    return {k: [round(float(b[k].quantile(.025)), 3), round(float(b[k].quantile(.975)), 3)]
            for k in ["pr_auc", "roc_auc", "precision", "recall", "f1"]}


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    train, test = read(TRAIN_CSV), read(TEST_CSV)
    ytr, yte = train[TARGET].to_numpy(), test[TARGET].to_numpy()
    feats = [c for c in final_features() if c in train.columns]
    print(f"include_volume={INCLUDE_VOLUME}  n_features={len(feats)}  "
          f"train={len(train)} ({ytr.mean():.1%} fraud)")

    baselines = {c.name: c() for c in RULE_BASELINES}
    ml_models = candidate_models()
    all_names = list(baselines) + list(ml_models)
    splits = list(cv_splitter(N_SPLITS, N_REPEATS).split(train, ytr))

    fold_rows, oof = [], {}
    for name in all_names:
        model, cols = spec(name, baselines, ml_models, feats)
        s, c = np.zeros(len(ytr)), np.zeros(len(ytr))
        for k, (tr, va) in enumerate(splits):
            est = estimator(model, cols).fit(train.iloc[tr][cols], ytr[tr])
            p = est.predict_proba(train.iloc[va][cols])[:, 1]
            s[va] += p
            c[va] += 1
            fold_rows.append({"model": name, "fold": k, **threshold_free_metrics(ytr[va], p),
                              **threshold_metrics(ytr[va], p, 0.5)})
        oof[name] = s / c
        f = pd.DataFrame([r for r in fold_rows if r["model"] == name])
        print(f"{name:24s} PR-AUC {f.pr_auc.mean():.3f} ± {f.pr_auc.std():.3f}   "
              f"ROC-AUC {f.roc_auc.mean():.3f}   Brier {f.brier.mean():.3f}")

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(OUTPUTS / "cv_fold_scores.csv", index=False)
    summary = folds.drop(columns="fold").groupby("model", sort=False).agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.round(4).to_csv(OUTPUTS / "cv_results.csv")

    # ---- one-standard-error selection among ML models (dict order = simplest first) ----
    ml = summary.loc[list(ml_models)]
    best = ml["pr_auc_mean"].idxmax()
    se = float(ml.loc[best, "pr_auc_std"] / np.sqrt(len(splits)))
    eligible = [m for m in ml_models if ml.loc[m, "pr_auc_mean"] >= ml.loc[best, "pr_auc_mean"] - se]
    selected = eligible[0]

    rule_tbl = summary.loc[list(baselines)]
    best_rule = rule_tbl["pr_auc_mean"].idxmax()
    gain = float(ml.loc[selected, "pr_auc_mean"] - rule_tbl.loc[best_rule, "pr_auc_mean"])

    thr = max_f1_threshold(ytr, oof[selected])

    # ---- calibration on out-of-fold probabilities ----
    cal = pd.DataFrame({"p": oof[selected], "y": ytr})
    cal["bin"] = pd.qcut(cal.p, 5, duplicates="drop")
    cal.groupby("bin", observed=True).agg(mean_predicted=("p", "mean"),
                                          observed_rate=("y", "mean"),
                                          n=("y", "size")).round(3).to_csv(OUTPUTS / "calibration_oof.csv")

    pd.DataFrame({ID_COL: train[ID_COL], TARGET: ytr,
                  "exposure_amount": train["exposure_amount"],
                  **{f"p_{k}": v for k, v in oof.items()}}
                 ).to_csv(OUTPUTS / "oof_probabilities.csv", index=False)

    sel_model, sel_feats = spec(selected, baselines, ml_models, feats)
    selection = {"include_volume": INCLUDE_VOLUME, "cv": f"{N_REPEATS}x{N_SPLITS} repeated stratified",
                 "best_by_mean_pr_auc": best, "one_se": round(se, 4),
                 "eligible_within_1se": eligible, "selected_model": selected,
                 "best_rule_baseline": best_rule,
                 "rule_pr_auc": round(float(rule_tbl.loc[best_rule, "pr_auc_mean"]), 4),
                 "gain_over_best_rule": round(gain, 4),
                 "model_justified": bool(gain > se),
                 "provisional_threshold_oof_max_f1": round(thr, 4),
                 "oof_mean_predicted": round(float(oof[selected].mean()), 4),
                 "train_fraud_rate": round(float(ytr.mean()), 4),
                 "selected_features": sel_feats}
    (OUTPUTS / "selection.json").write_text(json.dumps(selection, indent=2))
    print("\n" + json.dumps({k: v for k, v in selection.items() if k != "selected_features"},
                            indent=2))
    if not selection["model_justified"]:
        from config import INCLUDE_SIZE as _inc, SIZE_FEATURES as _sf
        rule_uses_excluded = (not _inc) and any(c in RULE_COLUMNS for c in _sf)
        if rule_uses_excluded:
            print("\nNOTE: model_justified is False, but the comparison is not like-for-like.\n"
                  "  INCLUDE_SIZE=False denies the model the size columns "
                  f"({', '.join(c for c in RULE_COLUMNS if c in _sf)}) that the\n"
                  "  rule baselines rank by. A size-free model is EXPECTED to lose on global\n"
                  "  PR-AUC; that is the trade accepted when turning the flag off.\n"
                  "  Judge it instead on:\n"
                  "    - within-stratum ROC-AUC from size_confound.py, and\n"
                  "    - precision@K vs 'rule_total_billed_only' in rank_topk.py.")
        else:
            print("\nWARNING: the selected model does not beat the best rule baseline by more "
                  "than one standard error. Do not proceed to the cost and ranking steps until "
                  "this is resolved -- a rule would do the same job.")

    final = build_pipeline(sel_feats, clone(sel_model)).fit(train[sel_feats], ytr)
    joblib.dump({"pipeline": final, "features": sel_feats, "threshold": thr,
                 "selection": selection}, MODELS / "provider_fraud_model.joblib")

    # ---- ONE final test evaluation ----
    rows = []
    p_sel = final.predict_proba(test[sel_feats])[:, 1]
    for label, t in [("0.50", 0.5), ("oof_max_f1", thr)]:
        rows.append({"model": selected, "threshold": label,
                     **threshold_free_metrics(yte, p_sel), **threshold_metrics(yte, p_sel, t)})
    for name in [best_rule, "Baseline_Majority"]:
        p = baselines[name].fit(train[RULE_COLUMNS], ytr).predict_proba(test[RULE_COLUMNS])[:, 1]
        rows.append({"model": name, "threshold": "rule", **threshold_free_metrics(yte, p),
                     **threshold_metrics(yte, p, 0.5)})
    pd.DataFrame(rows).round(3).to_csv(OUTPUTS / "final_test_results.csv", index=False)

    test_out = test[[ID_COL, TARGET, "exposure_amount"]].copy()
    test_out["p_fraud"] = p_sel
    test_out.to_csv(OUTPUTS / "test_probabilities.csv", index=False)

    ci = bootstrap_ci(yte, p_sel, thr)
    (OUTPUTS / "test_bootstrap_ci.json").write_text(json.dumps(ci, indent=2))

    print("\nFINAL TEST (evaluated once):")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"\nBootstrap 95% CI for {selected}: {json.dumps(ci)}")


if __name__ == "__main__":
    main()
