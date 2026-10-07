"""
Generates SYNTHETIC data with the exact schema of the Kaggle
"Healthcare Provider Fraud Detection Analysis" files.

Purpose: let you run and test the whole pipeline before (or without) downloading the
real data. NOTHING produced here is real, and no result from it belongs in your report.

Usage:
    python src/make_sample_data.py --providers 600 --out data/raw_sample

Then point config.RAW at that folder, or copy the files into data/raw/.

To use the REAL data instead, download these four files from
https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis
into data/raw/ and skip this script entirely:
    Train_Beneficiarydata-*.csv
    Train_Inpatientdata-*.csv
    Train_Outpatientdata-*.csv
    Train-*.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import (BENE_ANNUAL_COLS, CHRONIC_COLS, DIAG_COLS, PHYSICIAN_COLS, PROC_COLS, SEED)

STAMP = "1542865627584"   # matches the real filename suffix


def make(n_providers, n_benes, rng):
    """Providers get latent behaviour traits; fraud is a NOISY probabilistic function of
    those traits, not a deterministic flag. This is deliberate: a generator where fraud is
    obvious produces PR-AUC 1.0 and tells you nothing about whether the pipeline works."""
    providers = [f"PRV{51000 + i}" for i in range(n_providers)]

    # latent traits, standard normal, overlapping heavily between classes
    inflation = rng.normal(0, 1, n_providers)      # bills more per claim
    dup_rate = rng.normal(0, 1, n_providers)       # repeat-bills the same patient
    doc_conc = rng.normal(0, 1, n_providers)       # one physician signs everything
    size = rng.normal(0, 1, n_providers)           # confounder: sheer size

    # fraud depends mostly on behaviour, weakly on size, plus a large noise term.
    logit = (-2.6 + 1.15 * inflation + 0.95 * dup_rate + 0.75 * doc_conc
             + 0.25 * size + rng.normal(0, 0.55, n_providers))
    fraud = rng.random(n_providers) < 1 / (1 + np.exp(-logit))

    labels = pd.DataFrame({"Provider": providers,
                           "PotentialFraud": np.where(fraud, "Yes", "No")})

    # ---- beneficiaries ----
    benes = [f"BENE{11000 + i}" for i in range(n_benes)]
    dob = pd.to_datetime("1920-01-01") + pd.to_timedelta(rng.integers(0, 25000, n_benes), "D")
    dod = pd.Series(pd.NaT, index=range(n_benes))
    died = rng.random(n_benes) < 0.02
    dod[died] = pd.to_datetime("2009-01-01") + pd.to_timedelta(rng.integers(0, 365, died.sum()), "D")
    bene = pd.DataFrame({
        "BeneID": benes,
        "DOB": dob.strftime("%Y-%m-%d"),
        "DOD": pd.to_datetime(dod).dt.strftime("%Y-%m-%d"),
        "Gender": rng.integers(1, 3, n_benes),
        "Race": rng.choice([1, 2, 3, 5], n_benes, p=[.82, .1, .04, .04]),
        "RenalDiseaseIndicator": np.where(rng.random(n_benes) < .12, "Y", "0"),
        "State": rng.integers(1, 55, n_benes),
        "County": rng.integers(1, 999, n_benes),
        "NoOfMonths_PartACov": rng.choice([12, 11, 10], n_benes, p=[.94, .03, .03]),
        "NoOfMonths_PartBCov": rng.choice([12, 11, 10], n_benes, p=[.94, .03, .03]),
    })
    for c in CHRONIC_COLS:
        bene[c] = rng.choice([1, 2], n_benes, p=[.45, .55])   # 1 = yes, 2 = no, as in the real file
    bene["IPAnnualReimbursementAmt"] = (rng.gamma(1.4, 4000, n_benes)).round(0)
    bene["IPAnnualDeductibleAmt"] = (rng.gamma(1.2, 550, n_benes)).round(0)
    bene["OPAnnualReimbursementAmt"] = (rng.gamma(1.6, 900, n_benes)).round(0)
    bene["OPAnnualDeductibleAmt"] = (rng.gamma(1.3, 260, n_benes)).round(0)

    # ---- claims ----
    ip_rows, op_rows, cid = [], [], 0
    for j, prov in enumerate(providers):
        n_claims = int(np.clip(np.exp(4.55 + 0.55 * size[j] + rng.normal(0, .4)), 8, 1200))
        n_ip = int(n_claims * np.clip(rng.normal(.18, .10), .02, .5))
        amt_scale = float(np.exp(0.60 * inflation[j] + rng.normal(0, .10)))

        n_docs = max(1, int(n_claims / np.clip(np.exp(1.6 + 0.8 * doc_conc[j]), 1.5, 60)))
        docs = [f"PHY{rng.integers(300000, 460000)}" for _ in range(n_docs)]
        # weighted physician choice: high doc_conc means one physician dominates
        w = rng.dirichlet(np.full(n_docs, float(np.clip(np.exp(-0.9 * doc_conc[j]), .05, 8))))

        pool = max(2, int(n_claims / np.clip(np.exp(0.75 * dup_rate[j]) * rng.uniform(1.0, 1.3), .8, 12)))
        prov_benes = rng.choice(benes, size=min(len(benes), pool), replace=False)

        for k in range(n_claims):
            cid += 1
            inpatient = k < n_ip
            start = pd.Timestamp("2009-01-01") + pd.Timedelta(days=int(rng.integers(0, 700)))
            dur = int(rng.integers(0, 12)) if inpatient else int(rng.integers(0, 4))
            row = {
                "BeneID": rng.choice(prov_benes),
                "ClaimID": f"CLM{100000 + cid}",
                "ClaimStartDt": start.strftime("%Y-%m-%d"),
                "ClaimEndDt": (start + pd.Timedelta(days=dur)).strftime("%Y-%m-%d"),
                "Provider": prov,
                "InscClaimAmtReimbursed": float(round(
                    rng.gamma(2.0, (4200 if inpatient else 320) * amt_scale), 0)),
                "AttendingPhysician": rng.choice(docs, p=w) if rng.random() > .01 else np.nan,
                "OperatingPhysician": rng.choice(docs) if rng.random() < (.55 if inpatient else .08) else np.nan,
                "OtherPhysician": rng.choice(docs) if rng.random() < .3 else np.nan,
                "ClmAdmitDiagnosisCode": str(rng.integers(1000, 99999)) if inpatient else np.nan,
                "DeductibleAmtPaid": float(rng.choice([0, 1068])) if inpatient else float(rng.integers(0, 200)),
            }
            n_diag = rng.integers(1, 11 if inpatient else 6)
            for i, c in enumerate(DIAG_COLS):
                row[c] = str(rng.integers(1000, 99999)) if i < n_diag else np.nan
            n_proc = rng.integers(0, 4 if inpatient else 2)
            for i, c in enumerate(PROC_COLS):
                row[c] = float(rng.integers(1, 9999)) if i < n_proc else np.nan
            if inpatient:
                adm = start
                row["AdmissionDt"] = adm.strftime("%Y-%m-%d")
                row["DischargeDt"] = (adm + pd.Timedelta(days=max(1, dur))).strftime("%Y-%m-%d")
                row["DiagnosisGroupCode"] = str(rng.integers(1, 900))
                ip_rows.append(row)
            else:
                op_rows.append(row)

    ip_cols = ([ "BeneID", "ClaimID", "ClaimStartDt", "ClaimEndDt", "Provider",
                 "InscClaimAmtReimbursed"] + PHYSICIAN_COLS +
               ["AdmissionDt", "ClmAdmitDiagnosisCode", "DeductibleAmtPaid", "DischargeDt",
                "DiagnosisGroupCode"] + DIAG_COLS + PROC_COLS)
    op_cols = (["BeneID", "ClaimID", "ClaimStartDt", "ClaimEndDt", "Provider",
                "InscClaimAmtReimbursed"] + PHYSICIAN_COLS +
               ["ClmAdmitDiagnosisCode", "DeductibleAmtPaid"] + DIAG_COLS + PROC_COLS)
    return (labels, bene,
            pd.DataFrame(ip_rows).reindex(columns=ip_cols),
            pd.DataFrame(op_rows).reindex(columns=op_cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", type=int, default=600)
    ap.add_argument("--beneficiaries", type=int, default=8000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rng = np.random.default_rng(SEED)
    labels, bene, ip, op = make(a.providers, a.beneficiaries, rng)

    out = Path(a.out) if a.out else Path(__file__).resolve().parents[1] / "data" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out / f"Train-{STAMP}.csv", index=False)
    bene.to_csv(out / f"Train_Beneficiarydata-{STAMP}.csv", index=False)
    ip.to_csv(out / f"Train_Inpatientdata-{STAMP}.csv", index=False)
    op.to_csv(out / f"Train_Outpatientdata-{STAMP}.csv", index=False)

    print(f"SYNTHETIC data written to {out}")
    print(f"  providers   {len(labels):>7}  ({(labels.PotentialFraud == 'Yes').mean():.1%} fraud)")
    print(f"  inpatient   {len(ip):>7}")
    print(f"  outpatient  {len(op):>7}")
    print(f"  beneficiaries {len(bene):>5}")
    print("\nThis is NOT real data. Replace with the Kaggle files before reporting anything.")


if __name__ == "__main__":
    main()
