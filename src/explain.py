"""
Step 8 (Part 6): explain the scores, globally and per provider.

Uses SHAP when available, and falls back to model coefficients or impurity importances
when it is not, so the script never becomes a hard dependency.

Language rule, applied throughout: a feature CONTRIBUTES to a score. It does not cause
fraud. The output is a reason an analyst can check, not a finding.
"""
import json

import joblib
import numpy as np
import pandas as pd

from config import ID_COL, MODELS, OUTPUTS, TARGET, TEST_CSV

TOP_N_GLOBAL = 15
TOP_N_LOCAL = 5
N_EXAMPLES = 10

READABLE = {
    "n_claims": "number of claims filed",
    "total_reimbursed": "total amount reimbursed",
    "mean_reimbursed": "average reimbursement per claim",
    "median_reimbursed": "median reimbursement per claim",
    "max_reimbursed": "largest single reimbursement",
    "p90_reimbursed": "90th-percentile reimbursement",
    "std_reimbursed": "variability of reimbursement amounts",
    "reimbursed_per_beneficiary": "reimbursement per patient",
    "claims_per_beneficiary": "claims filed per patient",
    "claims_per_attending_physician": "claims per attending physician",
    "top_physician_concentration": "share of claims signed by a single physician",
    "top_diagnosis_concentration": "share of claims with the same primary diagnosis",
    "pct_duplicate_bene_claims_same_day": "share of patients billed twice on one day",
    "pct_claims_missing_attending": "share of claims with no attending physician recorded",
    "pct_claims_with_operating_physician": "share of claims listing an operating physician",
    "mean_n_diagnosis_codes": "average diagnosis codes per claim",
    "mean_n_procedure_codes": "average procedure codes per claim",
    "mean_claim_duration_days": "average claim duration",
    "mean_admit_days": "average length of stay",
    "inpatient_share": "share of claims that are inpatient",
    "n_unique_beneficiaries": "number of distinct patients",
}


def readable(f):
    return READABLE.get(f, f.replace("_", " "))


def global_importance(pipeline, feats, X):
    """Returns a DataFrame of feature, importance, direction. SHAP if available."""
    prep, model = pipeline.named_steps["prep"], pipeline.named_steps["model"]
    Xt = prep.transform(X)
    names = list(prep.get_feature_names_out())
    names = [n.split("__", 1)[-1] for n in names]
    try:
        import shap
        bg = shap.utils.sample(Xt, min(100, Xt.shape[0]), random_state=0)
        explainer = (shap.LinearExplainer(model, bg) if hasattr(model, "coef_")
                     else shap.TreeExplainer(model))
        vals = explainer.shap_values(Xt)
        if isinstance(vals, list):
            vals = vals[1]
        if getattr(vals, "ndim", 2) == 3:
            vals = vals[:, :, 1]
        imp = np.abs(vals).mean(axis=0)
        direction = np.sign((vals * (Xt - Xt.mean(axis=0))).mean(axis=0))
        method = "shap"
    except Exception as e:                                    # noqa: BLE001
        print(f"  (SHAP unavailable: {type(e).__name__}; using model-native importances)")
        vals = None
        if hasattr(model, "coef_"):
            imp = np.abs(model.coef_.ravel())
            direction = np.sign(model.coef_.ravel())
        else:
            imp = model.feature_importances_
            direction = np.zeros_like(imp)
        method = "coefficients" if hasattr(model, "coef_") else "impurity"
    df = pd.DataFrame({"feature": names, "importance": imp, "direction": direction})
    return df.sort_values("importance", ascending=False), vals, names, method


def main():
    bundle = joblib.load(MODELS / "provider_fraud_model.joblib")
    pipeline, feats = bundle["pipeline"], bundle["features"]
    threshold = bundle.get("cost_threshold", bundle["threshold"])

    test = pd.read_csv(TEST_CSV, keep_default_na=False, na_values=[""])
    X = test[feats]
    p = pipeline.predict_proba(X)[:, 1]

    print(f"Explaining {bundle['selection']['selected_model']} on {len(test)} held-out providers")
    imp, vals, names, method = global_importance(pipeline, feats, X)
    imp.head(TOP_N_GLOBAL).round(4).to_csv(OUTPUTS / "global_importance.csv", index=False)

    print(f"\nGlobal drivers ({method}):")
    for _, r in imp.head(TOP_N_GLOBAL).iterrows():
        arrow = "raises risk" if r.direction > 0 else ("lowers risk" if r.direction < 0 else "")
        print(f"  {r.importance:7.4f}  {readable(r.feature):<52s} {arrow}")

    # ---- per-provider reasons ---------------------------------------------------
    order = np.argsort(-p)[:N_EXAMPLES]
    cards = []
    for i in order:
        row = test.iloc[i]
        card = {"provider": row[ID_COL], "fraud_probability": round(float(p[i]), 4),
                "exposure_amount": float(row["exposure_amount"]),
                "expected_exposure": round(float(p[i] * row["exposure_amount"]), 2),
                "flagged_at_cost_threshold": bool(p[i] >= threshold),
                "actual_label": int(row[TARGET])}
        if vals is not None:
            contrib = pd.Series(vals[i], index=names).sort_values(key=np.abs, ascending=False)
            card["contributed_to_higher_risk"] = [
                f"{readable(f)} (value {row[f]:,.2f})" if f in row else readable(f)
                for f in contrib[contrib > 0].head(TOP_N_LOCAL).index]
            card["contributed_to_lower_risk"] = [
                f"{readable(f)} (value {row[f]:,.2f})" if f in row else readable(f)
                for f in contrib[contrib < 0].head(TOP_N_LOCAL).index]
        else:
            card["contributed_to_higher_risk"] = [readable(f) for f in
                                                  imp[imp.direction > 0].head(TOP_N_LOCAL).feature]
            card["contributed_to_lower_risk"] = [readable(f) for f in
                                                 imp[imp.direction < 0].head(TOP_N_LOCAL).feature]
        cards.append(card)

    (OUTPUTS / "explanations.json").write_text(json.dumps(
        {"method": method,
         "language_rule": ("Features CONTRIBUTE to a risk score. They do not cause or prove fraud. "
                           "Every flagged provider requires human investigation."),
         "providers": cards}, indent=2))

    print(f"\nTop {min(3, len(cards))} providers by risk:")
    for c in cards[:3]:
        print(f"\n  {c['provider']}   P(fraud) {c['fraud_probability']:.3f}   "
              f"exposure {c['exposure_amount']:,.0f}   "
              f"{'FLAGGED' if c['flagged_at_cost_threshold'] else 'below threshold'}")
        for r in c["contributed_to_higher_risk"][:3]:
            print(f"     + {r}")
        for r in c["contributed_to_lower_risk"][:2]:
            print(f"     - {r}")

    print(f"\nSaved {OUTPUTS / 'global_importance.csv'} and {OUTPUTS / 'explanations.json'}")
    print("\nReminder: these are contributions to a model score, not evidence of fraud.")


if __name__ == "__main__":
    main()
