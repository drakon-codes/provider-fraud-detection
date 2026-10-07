"""
Step 1: audit the raw files BEFORE building anything.

Checks the failure modes that cost time on the previous dataset:
  * are category strings being eaten by pandas' default NaN list?
  * which columns are genuinely missing vs structurally absent (outpatient has no AdmissionDt)?
  * do provider/claim/beneficiary IDs behave as expected?
  * does any provider appear with conflicting labels?
  * do beneficiaries or physicians span multiple providers? (affects how the split must work)
  * what is the date span, and is temporal validation possible?

Writes outputs/audit_raw.json. Reads nothing else; makes no modelling decisions.
"""
import json

import numpy as np
import pandas as pd

from config import (AMOUNT, BENE_ID, CLAIM_END, CLAIM_ID, CLAIM_START, DIAG_COLS,
                    EXCLUDE_PREFIXES, FILE_PATTERNS, ID_COL, OUTPUTS, PHYSICIAN_COLS,
                    POSITIVE_LABEL, PROC_COLS, RAW, TARGET)


def find(pattern):
    """Resolve exactly one file. Test_* files are excluded outright, and an ambiguous
    match is a hard error rather than a silent 'take the first one'."""
    hits = [p for p in sorted(RAW.glob(pattern))
            if not p.name.startswith(EXCLUDE_PREFIXES)]
    if not hits:
        raise FileNotFoundError(
            f"No file matching '{pattern}' in {RAW}.\n"
            f"Download the four Train files from Kaggle into that folder, or run:\n"
            f"    python src/make_sample_data.py")
    if len(hits) > 1:
        raise RuntimeError(
            f"'{pattern}' matched {len(hits)} files in {RAW}:\n  "
            + "\n  ".join(h.name for h in hits)
            + "\nLeave exactly one of each in place so the loaded file is unambiguous.")
    return hits[0]


def read(pattern):
    # keep_default_na=False stops pandas turning category strings such as 'None' or 'NA'
    # into NaN. Empty strings are the only true missing marker in these files.
    return pd.read_csv(find(pattern), keep_default_na=False, na_values=[""], low_memory=False)


def load_all():
    return (read(FILE_PATTERNS["labels"]), read(FILE_PATTERNS["beneficiary"]),
            read(FILE_PATTERNS["inpatient"]), read(FILE_PATTERNS["outpatient"]))


def main():
    labels, bene, ip, op = load_all()
    rep = {}

    rep["files"] = {k: find(v).name for k, v in FILE_PATTERNS.items()}
    rep["shapes"] = {"labels": list(labels.shape), "beneficiary": list(bene.shape),
                     "inpatient": list(ip.shape), "outpatient": list(op.shape)}

    # ---- labels -------------------------------------------------------------------
    rep["label_counts"] = labels[TARGET].value_counts().to_dict()
    rep["fraud_rate"] = round(float((labels[TARGET] == POSITIVE_LABEL).mean()), 4)
    rep["n_providers_labelled"] = int(labels[ID_COL].nunique())
    rep["duplicate_provider_labels"] = int(labels[ID_COL].duplicated().sum())
    conflict = labels.groupby(ID_COL)[TARGET].nunique()
    rep["providers_with_conflicting_labels"] = int((conflict > 1).sum())
    n_pos = int((labels[TARGET] == POSITIVE_LABEL).sum())
    rep["expected_test_positives"] = int(0.2 * n_pos)
    rep["size_verdict"] = ("TOO SMALL" if 0.2 * n_pos < 100 else
                           "workable" if 0.2 * n_pos < 300 else "good")

    # ---- claims -------------------------------------------------------------------
    claims = pd.concat([ip.assign(EncounterType="inpatient"),
                        op.assign(EncounterType="outpatient")], ignore_index=True)
    rep["n_claims"] = int(len(claims))
    rep["duplicate_claim_ids"] = int(claims[CLAIM_ID].duplicated().sum())
    prov_claims, prov_labels = set(claims[ID_COL]), set(labels[ID_COL])
    rep["providers_in_claims"] = len(prov_claims)
    rep["providers_with_claims_and_label"] = len(prov_claims & prov_labels)
    rep["providers_in_claims_not_labelled"] = len(prov_claims - prov_labels)
    rep["labelled_providers_with_no_claims"] = len(prov_labels - prov_claims)
    rep["pct_labelled_providers_with_claims"] = round(
        len(prov_claims & prov_labels) / max(len(prov_labels), 1), 4)

    # FATAL: labels and claims must describe the same providers. Near-zero overlap almost
    # always means the Test_* claim files were loaded against the Train labels.
    if rep["providers_with_claims_and_label"] < 0.5 * len(prov_labels):
        raise SystemExit(
            f"\nFATAL: only {rep['providers_with_claims_and_label']} of {len(prov_labels)} "
            f"labelled providers appear in the claim files.\n"
            f"Loaded claim files:\n"
            f"  inpatient  = {find(FILE_PATTERNS['inpatient']).name}\n"
            f"  outpatient = {find(FILE_PATTERNS['outpatient']).name}\n"
            f"  labels     = {find(FILE_PATTERNS['labels']).name}\n"
            f"These must all be the Train_* / Train- files. The Test_* files have no labels.")

    # true missing per column, split by encounter type so structural absence is visible
    miss = {}
    for name, part in [("inpatient", ip), ("outpatient", op)]:
        miss[name] = {c: int(part[c].isna().sum()) for c in part.columns if part[c].isna().any()}
    rep["missing_by_encounter"] = miss
    rep["note_on_missing"] = ("AdmissionDt/DischargeDt/DiagnosisGroupCode are absent from outpatient "
                              "claims by design, not missing data. Physician and diagnosis-code "
                              "columns are sparse by design: claims list fewer codes than slots.")

    # ---- amounts --------------------------------------------------------------------
    amt = pd.to_numeric(claims[AMOUNT], errors="coerce")
    rep["amount_stats"] = {"min": float(amt.min()), "median": float(amt.median()),
                           "mean": round(float(amt.mean()), 2), "max": float(amt.max()),
                           "n_negative": int((amt < 0).sum()), "n_zero": int((amt == 0).sum())}

    # ---- dates ----------------------------------------------------------------------
    start = pd.to_datetime(claims[CLAIM_START], errors="coerce")
    end = pd.to_datetime(claims[CLAIM_END], errors="coerce")
    rep["date_span"] = [str(start.min().date()), str(start.max().date())]
    rep["date_span_days"] = int((start.max() - start.min()).days)
    rep["temporal_validation_possible"] = bool(rep["date_span_days"] > 365)
    rep["claims_ending_before_start"] = int((end < start).sum())

    # ---- entity overlap (matters for how the split must be done) --------------------
    prov_per_bene = claims.groupby(BENE_ID)[ID_COL].nunique()
    rep["beneficiaries_seen_by_multiple_providers"] = int((prov_per_bene > 1).sum())
    rep["pct_beneficiaries_shared"] = round(float((prov_per_bene > 1).mean()), 4)
    att = claims.dropna(subset=["AttendingPhysician"])
    prov_per_doc = att.groupby("AttendingPhysician")[ID_COL].nunique()
    rep["physicians_working_for_multiple_providers"] = int((prov_per_doc > 1).sum())
    rep["split_note"] = ("The modelling unit is the provider and each provider is one row, so a "
                         "provider-level split cannot leak a provider across train and test. Shared "
                         "beneficiaries and physicians mean cross-provider network features WOULD "
                         "leak; none are used.")

    # ---- code columns -----------------------------------------------------------------
    rep["n_unique_diagnosis_codes"] = int(pd.unique(claims[DIAG_COLS].values.ravel()).size)
    rep["n_unique_procedure_codes"] = int(pd.unique(claims[PROC_COLS].values.ravel()).size)
    rep["n_unique_attending_physicians"] = int(claims["AttendingPhysician"].nunique())
    rep["code_note"] = ("Diagnosis/procedure codes are high-cardinality identifiers. They are used "
                        "only as counts and concentration ratios, never one-hot encoded.")

    # ---- beneficiaries ------------------------------------------------------------------
    rep["duplicate_bene_ids"] = int(bene[BENE_ID].duplicated().sum())
    rep["bene_ids_in_claims_missing_from_beneficiary_file"] = int(
        len(set(claims[BENE_ID]) - set(bene[BENE_ID])))

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "audit_raw.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))
    print(f"\nSaved {OUTPUTS / 'audit_raw.json'}")


if __name__ == "__main__":
    main()
