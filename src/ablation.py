"""
Step 4: feature-group ablation on the TRAINING split only.

The point is the question that sank the previous dataset: is the model learning
behaviour, or is it learning that big providers get investigated more?

Every result here uses repeated stratified CV on training data. The test set is untouched.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone

from config import (ALL_FEATURE_GROUPS, BEHAVIOUR_FEATURES, MONEY_FEATURES, OUTPUTS,
                    PATIENT_MIX_FEATURES, SIZE_FEATURES, TARGET, TRAIN_CSV, VOLUME_FEATURES,
                    build_pipeline, candidate_models, cv_splitter, size_free_features,
                    threshold_free_metrics)

ALL = VOLUME_FEATURES + MONEY_FEATURES + BEHAVIOUR_FEATURES + PATIENT_MIX_FEATURES


def without(cols):
    return [c for c in ALL if c not in set(cols)]


FEATURE_SETS = {
    "A_all_groups": ALL,
    "B_volume_only": VOLUME_FEATURES,
    "C_money_only": MONEY_FEATURES,
    "D_behaviour_only": BEHAVIOUR_FEATURES,
    "E_patient_mix_only": PATIENT_MIX_FEATURES,
    "F_no_volume": without(VOLUME_FEATURES),
    "G_no_patient_mix": without(PATIENT_MIX_FEATURES),
    "H_behaviour_plus_money": BEHAVIOUR_FEATURES + MONEY_FEATURES,
    "I_single_best_n_claims": ["n_claims"],
    "J_single_best_total_reimbursed": ["total_reimbursed"],
    "K_size_free_rates_and_ratios": size_free_features(),
    "L_size_only": SIZE_FEATURES,
}


def main():
    train = pd.read_csv(TRAIN_CSV, keep_default_na=False, na_values=[""])
    y = train[TARGET].to_numpy()
    models = candidate_models()
    chosen = {k: models[k][0] for k in
              ["LogReg_L2_C1", "XGBoost" if "XGBoost" in models else "RandomForest"]}
    splits = list(cv_splitter().split(train, y))

    rows = []
    for set_name, feats in FEATURE_SETS.items():
        feats = [c for c in feats if c in train.columns]
        for model_name, model in chosen.items():
            scores = []
            for tr, va in splits:
                pipe = build_pipeline(feats, clone(model)).fit(train.iloc[tr][feats], y[tr])
                p = pipe.predict_proba(train.iloc[va][feats])[:, 1]
                scores.append(threshold_free_metrics(y[va], p))
            s = pd.DataFrame(scores)
            rows.append({"feature_set": set_name, "model": model_name, "n_features": len(feats),
                         "pr_auc_mean": s.pr_auc.mean(), "pr_auc_std": s.pr_auc.std(),
                         "roc_auc_mean": s.roc_auc.mean()})
            print(f"{set_name:32s} {model_name:14s} n={len(feats):>2}  "
                  f"PR-AUC {s.pr_auc.mean():.3f} ± {s.pr_auc.std():.3f}  "
                  f"ROC-AUC {s.roc_auc.mean():.3f}")

    res = pd.DataFrame(rows)
    ref = res[res.feature_set == "A_all_groups"].set_index("model")["pr_auc_mean"]
    res["delta_vs_all"] = res.apply(lambda r: r.pr_auc_mean - ref[r.model], axis=1)
    res.round(4).to_csv(OUTPUTS / "ablation_results.csv", index=False)

    print(f"\nBase rate (random ranking PR-AUC): {y.mean():.3f}")
    print("(This script runs a FIXED set of experiments and ignores config.INCLUDE_SIZE -- "
          "its job is\n to compare size-inclusive and size-free sets, so its output does not "
          "change when you\n flip that flag. INCLUDE_SIZE affects train.py, rank_topk.py and "
          "explain.py.)")
    def best(name):
        return float(res[res.feature_set == name].pr_auc_mean.max())

    alln = best("A_all_groups")
    print(f"{'all groups':<34s} {alln:.3f}")
    for label, key in [("volume only", "B_volume_only"), ("size only (counts + totals)", "L_size_only"),
                       ("single column: total_reimbursed", "J_single_best_total_reimbursed"),
                       ("size-free (rates and ratios)", "K_size_free_rates_and_ratios")]:
        v = best(key)
        print(f"{label:<34s} {v:.3f}   gain of all-groups over this: {alln - v:+.3f}")

    worst_confounder = max(best("L_size_only"), best("J_single_best_total_reimbursed"))
    if alln - worst_confounder < 0.03:
        print("\nWARNING: the full feature set adds less than 0.03 PR-AUC over provider size "
              "alone.\n  The model may be ranking by how big a provider is, not how it behaves.\n"
              "  Run:  python src/size_confound.py\n"
              "  If that confirms it, set INCLUDE_SIZE = False in config.py and re-run.")
    print(f"Saved {OUTPUTS / 'ablation_results.csv'}")


if __name__ == "__main__":
    main()
