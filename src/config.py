"""
Central configuration. Every other module imports from here, so a decision
is changed in exactly one place.

Dataset: "Healthcare Provider Fraud Detection Analysis" (Kaggle, rohitrox).
Four files, all at data/raw/:
    Train_Beneficiarydata-*.csv
    Train_Inpatientdata-*.csv
    Train_Outpatientdata-*.csv
    Train-*.csv                  <- Provider, PotentialFraud (Yes/No)

The Test_* files on Kaggle have NO labels, so they are unusable for evaluation.
We ignore them and make our own split from the labelled training file.

UNIT OF ANALYSIS: the provider, not the claim. Labels are attached to providers,
so claims are aggregated up. This matches how insurers actually work -- they open
an investigation into a provider, not a single claim.
"""
from pathlib import Path
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", message=".*penalty.*")

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"

PROVIDERS_CSV = PROCESSED / "provider_features.csv"
TRAIN_CSV = PROCESSED / "train_providers.csv"
TEST_CSV = PROCESSED / "test_providers.csv"

SEED = 42
TEST_SIZE = 0.20
ID_COL = "Provider"
TARGET = "PotentialFraud"
POSITIVE_LABEL = "Yes"

# --------------------------------------------------------------------------- raw schema
# Filenames on Kaggle carry a timestamp suffix, so we glob on a prefix instead.
# IMPORTANT: these must be Train-specific. Kaggle ships Test_* copies of the same three
# files, and only the Train_* ones correspond to the labels in Train-*.csv. Matching both
# and taking the first hit silently loads the Test_* claims ("Test" sorts before "Train"),
# which produces a table where zero providers have both claims and a label.
FILE_PATTERNS = {
    "beneficiary": "Train_Beneficiarydata*.csv",
    "inpatient": "Train_Inpatientdata*.csv",
    "outpatient": "Train_Outpatientdata*.csv",
    "labels": "Train-*.csv",
}
# The Test_* files carry NO labels, so they cannot be used for evaluation. They are ignored.
EXCLUDE_PREFIXES = ("Test_", "Test-")

CLAIM_ID = "ClaimID"
BENE_ID = "BeneID"
AMOUNT = "InscClaimAmtReimbursed"
DEDUCTIBLE = "DeductibleAmtPaid"
CLAIM_START, CLAIM_END = "ClaimStartDt", "ClaimEndDt"
ADMIT_DT, DISCHARGE_DT = "AdmissionDt", "DischargeDt"

PHYSICIAN_COLS = ["AttendingPhysician", "OperatingPhysician", "OtherPhysician"]
DIAG_COLS = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
PROC_COLS = [f"ClmProcedureCode_{i}" for i in range(1, 7)]

CHRONIC_COLS = [
    "ChronicCond_Alzheimer", "ChronicCond_Heartfailure", "ChronicCond_KidneyDisease",
    "ChronicCond_Cancer", "ChronicCond_ObstrPulmonary", "ChronicCond_Depression",
    "ChronicCond_Diabetes", "ChronicCond_IschemicHeart", "ChronicCond_Osteoporasis",
    "ChronicCond_rheumatoidarthritis", "ChronicCond_stroke",
]
BENE_ANNUAL_COLS = ["IPAnnualReimbursementAmt", "IPAnnualDeductibleAmt",
                    "OPAnnualReimbursementAmt", "OPAnnualDeductibleAmt"]

# Beneficiary attributes that are protected or sensitive. Excluded from the model by
# default: aggregating a provider's patient demographics into a fraud score means
# penalising providers for who their patients are.
PROTECTED_BENE_COLS = ["Gender", "Race"]

# --------------------------------------------------------------------------- feature groups
# Filled in by build_features.py; the groups drive ablation.py and final_features().
VOLUME_FEATURES = [
    "n_claims", "n_inpatient_claims", "n_outpatient_claims", "inpatient_share",
    "n_unique_beneficiaries", "n_unique_attending_physicians", "claims_per_beneficiary",
    "claims_per_attending_physician",
]
MONEY_FEATURES = [
    "total_reimbursed", "mean_reimbursed", "median_reimbursed", "std_reimbursed",
    "max_reimbursed", "p90_reimbursed", "mean_deductible", "reimbursed_per_beneficiary",
    "mean_reimbursed_inpatient", "mean_reimbursed_outpatient",
]
BEHAVIOUR_FEATURES = [
    "mean_claim_duration_days", "mean_admit_days", "pct_claims_zero_duration",
    "pct_claims_missing_attending", "pct_claims_with_operating_physician",
    "mean_n_diagnosis_codes", "mean_n_procedure_codes", "pct_claims_with_procedure",
    "top_diagnosis_concentration", "top_physician_concentration",
    "pct_duplicate_bene_claims_same_day", "n_states_served", "n_counties_served",
]
PATIENT_MIX_FEATURES = [
    "mean_patient_age", "pct_patients_deceased", "mean_chronic_conditions",
    "pct_renal_disease", "mean_ip_annual_reimbursement", "mean_op_annual_reimbursement",
]

ALL_FEATURE_GROUPS = {
    "volume": VOLUME_FEATURES,
    "money": MONEY_FEATURES,
    "behaviour": BEHAVIOUR_FEATURES,
    "patient_mix": PATIENT_MIX_FEATURES,
}

# SIZE features are absolute counts and totals: they say how BIG a provider is, not how it
# BEHAVES. On this dataset they are the main confounder, because the PotentialFraud label
# may reflect who got investigated (large billers) rather than who defrauded.
# Everything not listed here is a rate, ratio, average or concentration, i.e. size-normalised.
SIZE_FEATURES = [
    "n_claims", "n_inpatient_claims", "n_outpatient_claims", "n_unique_beneficiaries",
    "n_unique_attending_physicians", "total_reimbursed", "n_states_served", "n_counties_served",
]

# Set INCLUDE_SIZE = False to force the model onto behaviour only. Do this when
# src/size_confound.py shows the model is effectively ranking by provider size.
INCLUDE_SIZE = True
INCLUDE_VOLUME = True     # kept for the volume-only baseline comparison


def size_free_features():
    """Every feature that is a rate, ratio, average or concentration."""
    all_f = [c for g in ALL_FEATURE_GROUPS.values() for c in g]
    return [c for c in all_f if c not in set(SIZE_FEATURES)]


def final_features():
    feats = []
    for name, cols in ALL_FEATURE_GROUPS.items():
        if name == "volume" and not INCLUDE_VOLUME:
            continue
        feats += cols
    if not INCLUDE_SIZE:
        feats = [c for c in feats if c not in set(SIZE_FEATURES)]
    return feats


CATEGORICAL_FEATURES = []          # provider-level table is all numeric after aggregation


# --------------------------------------------------------------------------- pipeline
def build_preprocessor(features):
    num = [c for c in features if c not in CATEGORICAL_FEATURES]
    cat = [c for c in features if c in CATEGORICAL_FEATURES]
    blocks = [("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                ("scale", StandardScaler())]), num)]
    if cat:
        blocks.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                        ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                                 min_frequency=20))]), cat))
    return ColumnTransformer(blocks)


def build_pipeline(features, model):
    return Pipeline([("prep", build_preprocessor(features)), ("model", model)])


def candidate_models():
    """{name: (estimator, feature_override_or_None)}, ordered simplest first.
    Order matters: the one-standard-error rule picks the FIRST model close enough to the best.
    No SMOTE anywhere -- it distorts probabilities, and Parts 4-5 multiply probabilities by money."""
    models = {
        "LogReg_L1_C0.1": (LogisticRegression(penalty="l1", solver="liblinear", C=0.1,
                                              random_state=SEED), None),
        "LogReg_L2_C1": (LogisticRegression(C=1.0, max_iter=5000, random_state=SEED), None),
        "DecisionTree_d4": (DecisionTreeClassifier(max_depth=4, min_samples_leaf=25,
                                                   random_state=SEED), None),
        "RandomForest": (RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                                n_jobs=-1, random_state=SEED), None),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = (XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                           subsample=0.8, colsample_bytree=0.8,
                                           eval_metric="logloss", random_state=SEED), None)
    except ImportError:
        pass
    return models


def cv_splitter(n_splits=5, n_repeats=3):
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)


# --------------------------------------------------------------------------- metrics
def threshold_free_metrics(y, p):
    return {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
            "brier": brier_score_loss(y, p)}


def threshold_metrics(y, p, threshold):
    yhat = (np.asarray(p) >= threshold).astype(int)
    return {"accuracy": accuracy_score(y, yhat),
            "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat, zero_division=0),
            "f1": f1_score(y, yhat, zero_division=0)}


# --------------------------------------------------------------------------- rule baselines
class _Rule(BaseEstimator):
    """Ranks providers by one column. Any ML model must beat these to justify itself."""
    col = None

    def fit(self, X, y=None):
        v = X[self.col].astype(float)
        self.lo_, self.hi_ = float(v.min()), float(v.max())
        return self

    def predict_proba(self, X):
        v = X[self.col].astype(float).to_numpy()
        rng = max(self.hi_ - self.lo_, 1e-9)
        s = np.clip((v - self.lo_) / rng, 0, 1)
        return np.column_stack([1 - s, s])


class VolumeRule(_Rule):
    """Rank by claim count. The single most dangerous confounder in this dataset."""
    name, col = "Rule_ClaimVolume", "n_claims"


class TotalAmountRule(_Rule):
    name, col = "Rule_TotalReimbursed", "total_reimbursed"


class MeanAmountRule(_Rule):
    name, col = "Rule_MeanReimbursed", "mean_reimbursed"


class MajorityClass(BaseEstimator):
    name = "Baseline_Majority"

    def fit(self, X, y):
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.rate_)
        return np.column_stack([1 - p, p])


RULE_BASELINES = [MajorityClass, VolumeRule, TotalAmountRule, MeanAmountRule]
RULE_COLUMNS = ["n_claims", "total_reimbursed", "mean_reimbursed"]


# --------------------------------------------------------------------------- cost model
# Relative units, NOT real insurer costs. Sensitivity across ratios is reported.
INVESTIGATION_COST = 1_500.0        # cost of investigating one provider
RECOVERY_FRACTION = 0.10            # fraction of a fraudulent provider's reimbursements recoverable
COST_RATIOS_TO_TEST = [0.02, 0.05, 0.10, 0.20, 0.40]
INVESTIGATOR_CAPACITY_PCT = [0.01, 0.05, 0.10, 0.20]
