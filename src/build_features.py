"""
Step 2: merge the four tables and aggregate to ONE ROW PER PROVIDER.

Design rules:
  * Only provider-internal statistics. No feature is computed using other providers'
    data (no target encoding, no cross-provider network features), so nothing can leak
    across the train/test split.
  * Protected beneficiary attributes (Gender, Race) are excluded: scoring a provider on
    who their patients are is not defensible in an investigation.
  * Diagnosis and procedure codes are used only as counts and concentration ratios.
    One-hot encoding 65,000 codes would produce a wide, noisy matrix and a model that
    memorises codes instead of behaviour.
  * Every feature answers "is this provider's behaviour unusual", not "is this provider big".
    Size features are kept, but isolated in the `volume` group so their contribution is
    always measurable (see ablation.py and the volume-only baseline).

Output: data/processed/provider_features.csv
"""
import numpy as np
import pandas as pd

from config import (ADMIT_DT, AMOUNT, BENE_ANNUAL_COLS, BENE_ID, CHRONIC_COLS, CLAIM_END, CLAIM_ID,
                    CLAIM_START, DEDUCTIBLE, DIAG_COLS, DISCHARGE_DT, FILE_PATTERNS, ID_COL,
                    POSITIVE_LABEL, PROC_COLS, PROCESSED, PROVIDERS_CSV, TARGET)
from audit import read


def _concentration(series):
    """Share of a provider's claims taken by its single most frequent value.
    High concentration = one physician or one diagnosis dominates everything."""
    s = series.dropna()
    return float(s.value_counts(normalize=True).iloc[0]) if len(s) else np.nan


def build_claims():
    ip = read(FILE_PATTERNS["inpatient"]).assign(is_inpatient=1)
    op = read(FILE_PATTERNS["outpatient"]).assign(is_inpatient=0)
    claims = pd.concat([ip, op], ignore_index=True)

    for c in [CLAIM_START, CLAIM_END, ADMIT_DT, DISCHARGE_DT]:
        if c in claims.columns:
            claims[c] = pd.to_datetime(claims[c], errors="coerce")
    claims[AMOUNT] = pd.to_numeric(claims[AMOUNT], errors="coerce")
    claims[DEDUCTIBLE] = pd.to_numeric(claims[DEDUCTIBLE], errors="coerce")

    claims["claim_duration_days"] = (claims[CLAIM_END] - claims[CLAIM_START]).dt.days
    if ADMIT_DT in claims.columns:
        claims["admit_days"] = (claims[DISCHARGE_DT] - claims[ADMIT_DT]).dt.days
    else:
        claims["admit_days"] = np.nan
    claims["n_diagnosis_codes"] = claims[DIAG_COLS].notna().sum(axis=1)
    claims["n_procedure_codes"] = claims[PROC_COLS].notna().sum(axis=1)
    return claims


def provider_claim_features(claims):
    g = claims.groupby(ID_COL)
    f = pd.DataFrame(index=g.size().index)

    # ---- volume ----------------------------------------------------------------
    f["n_claims"] = g.size()
    f["n_inpatient_claims"] = g["is_inpatient"].sum()
    f["n_outpatient_claims"] = f["n_claims"] - f["n_inpatient_claims"]
    f["inpatient_share"] = f["n_inpatient_claims"] / f["n_claims"]
    f["n_unique_beneficiaries"] = g[BENE_ID].nunique()
    f["n_unique_attending_physicians"] = g["AttendingPhysician"].nunique()
    f["claims_per_beneficiary"] = f["n_claims"] / f["n_unique_beneficiaries"].replace(0, np.nan)
    f["claims_per_attending_physician"] = (f["n_claims"] /
                                           f["n_unique_attending_physicians"].replace(0, np.nan))

    # ---- money ------------------------------------------------------------------
    f["total_reimbursed"] = g[AMOUNT].sum()
    f["mean_reimbursed"] = g[AMOUNT].mean()
    f["median_reimbursed"] = g[AMOUNT].median()
    f["std_reimbursed"] = g[AMOUNT].std()
    f["max_reimbursed"] = g[AMOUNT].max()
    f["p90_reimbursed"] = g[AMOUNT].quantile(0.90)
    f["mean_deductible"] = g[DEDUCTIBLE].mean()
    f["reimbursed_per_beneficiary"] = (f["total_reimbursed"] /
                                       f["n_unique_beneficiaries"].replace(0, np.nan))
    ip_only = claims[claims.is_inpatient == 1].groupby(ID_COL)[AMOUNT].mean()
    op_only = claims[claims.is_inpatient == 0].groupby(ID_COL)[AMOUNT].mean()
    f["mean_reimbursed_inpatient"] = ip_only
    f["mean_reimbursed_outpatient"] = op_only

    # ---- behaviour ----------------------------------------------------------------
    f["mean_claim_duration_days"] = g["claim_duration_days"].mean()
    f["mean_admit_days"] = g["admit_days"].mean()
    f["pct_claims_zero_duration"] = g["claim_duration_days"].apply(lambda s: float((s == 0).mean()))
    f["pct_claims_missing_attending"] = g["AttendingPhysician"].apply(lambda s: float(s.isna().mean()))
    f["pct_claims_with_operating_physician"] = g["OperatingPhysician"].apply(
        lambda s: float(s.notna().mean()))
    f["mean_n_diagnosis_codes"] = g["n_diagnosis_codes"].mean()
    f["mean_n_procedure_codes"] = g["n_procedure_codes"].mean()
    f["pct_claims_with_procedure"] = g["n_procedure_codes"].apply(lambda s: float((s > 0).mean()))
    f["top_diagnosis_concentration"] = g["ClmDiagnosisCode_1"].apply(_concentration)
    f["top_physician_concentration"] = g["AttendingPhysician"].apply(_concentration)

    # same beneficiary billed more than once on the same day
    dup = (claims.groupby([ID_COL, BENE_ID, CLAIM_START]).size().rename("n")
           .reset_index().groupby(ID_COL)
           .apply(lambda d: float((d["n"] > 1).sum() / max(d["n"].sum(), 1)), include_groups=False))
    f["pct_duplicate_bene_claims_same_day"] = dup
    return f


def provider_patient_mix(claims, bene):
    bene = bene.copy()
    bene["DOB"] = pd.to_datetime(bene["DOB"], errors="coerce")
    bene["DOD"] = pd.to_datetime(bene["DOD"], errors="coerce")
    ref = pd.Timestamp("2009-12-01")
    bene["patient_age"] = ((ref - bene["DOB"]).dt.days / 365.25).round(1)
    bene["is_deceased"] = bene["DOD"].notna().astype(int)
    # chronic columns are coded 1 = has condition, 2 = does not
    chronic = bene[CHRONIC_COLS].apply(pd.to_numeric, errors="coerce")
    bene["n_chronic_conditions"] = (chronic == 1).sum(axis=1)
    bene["has_renal_disease"] = (bene["RenalDiseaseIndicator"].astype(str) == "Y").astype(int)
    for c in BENE_ANNUAL_COLS:
        bene[c] = pd.to_numeric(bene[c], errors="coerce")

    keep = [BENE_ID, "patient_age", "is_deceased", "n_chronic_conditions",
            "has_renal_disease"] + BENE_ANNUAL_COLS + ["State", "County"]
    link = claims[[ID_COL, BENE_ID]].merge(bene[keep], on=BENE_ID, how="left")
    g = link.groupby(ID_COL)

    f = pd.DataFrame(index=g.size().index)
    f["mean_patient_age"] = g["patient_age"].mean()
    f["pct_patients_deceased"] = g["is_deceased"].mean()
    f["mean_chronic_conditions"] = g["n_chronic_conditions"].mean()
    f["pct_renal_disease"] = g["has_renal_disease"].mean()
    f["mean_ip_annual_reimbursement"] = g["IPAnnualReimbursementAmt"].mean()
    f["mean_op_annual_reimbursement"] = g["OPAnnualReimbursementAmt"].mean()
    f["n_states_served"] = g["State"].nunique()
    f["n_counties_served"] = g["County"].nunique()
    return f


def main():
    labels = read(FILE_PATTERNS["labels"])
    bene = read(FILE_PATTERNS["beneficiary"])
    claims = build_claims()

    feats = provider_claim_features(claims).join(provider_patient_mix(claims, bene))
    feats = feats.reset_index()

    df = labels[[ID_COL, TARGET]].merge(feats, on=ID_COL, how="inner")
    df[TARGET] = (df[TARGET] == POSITIVE_LABEL).astype(int)

    # exposure: total money this provider was reimbursed. Used by the cost model and Top-K ranking.
    df["exposure_amount"] = df["total_reimbursed"]

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROVIDERS_CSV, index=False)

    print(f"Provider table: {df.shape[0]} providers x {df.shape[1]} columns")
    print(f"Fraud: {int(df[TARGET].sum())} ({df[TARGET].mean():.1%})")
    print(f"Providers dropped (labelled but no claims): "
          f"{len(labels) - len(df)}")
    nulls = df.isna().sum()
    if nulls.any():
        print("\nNull counts (imputed inside CV folds, never globally):")
        print(nulls[nulls > 0].to_string())
    print(f"\nSaved {PROVIDERS_CSV}")


if __name__ == "__main__":
    main()
