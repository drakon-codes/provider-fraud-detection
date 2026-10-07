# Provider Fraud Investigation Prioritisation

An explainable, cost-aware system that ranks healthcare providers for investigation,
built on the Kaggle **Healthcare Provider Fraud Detection Analysis** dataset.

The system does not decide that fraud occurred. It decides **who a limited investigation
team should look at first**, and shows its reasoning. A human makes every call.

---

## Quick start

```bash
pip install -r requirements.txt

# Option A: real data (recommended)
#   Download these four files from
#   https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis
#   into data/raw/ :
#     Train_Beneficiarydata-*.csv
#     Train_Inpatientdata-*.csv
#     Train_Outpatientdata-*.csv
#     Train-*.csv
#
# Option B: synthetic sample, to test the pipeline first
python src/make_sample_data.py

./run_all.sh          # runs all ten stages
```

Or run stages individually:

```bash
export PYTHONPATH=src
python src/audit.py             # 1. raw-file audit, leakage and integrity checks
python src/build_features.py    # 2. merge 4 tables -> one row per provider
python src/split.py             # 3. freeze the train/test split
python src/ablation.py          # 4. which feature groups actually earn their place
python src/train.py             # 5. baselines, models, selection, ONE test evaluation
python src/size_confound.py     # 6. size-confound diagnostic
python src/threshold_cost.py    # 7. cost-aware threshold from out-of-fold probabilities
python src/rank_topk.py         # 8. Top-K queue, precision@K, recall@K, exposure
python src/queue_significance.py # 9. paired bootstrap vs total-billing rule
python src/explain.py           # 10. global and per-provider reasons
```

### About the synthetic generator

`make_sample_data.py` writes files with the **exact schema** of the real ones, so the
pipeline can be tested without the download. Fraud is a noisy probabilistic function of
latent provider behaviours, not a deterministic flag, so the difficulty is realistic.
**No number produced from synthetic data belongs in your report.** Delete `data/raw/` and
drop in the Kaggle files before generating results you intend to present.

---

## The three decisions that shape everything

**1. The unit of analysis is the provider, not the claim.**
Labels in this dataset attach to providers, so claims are aggregated upward. This matches
reality: an insurer opens a case on a provider, not on a single claim. It also makes the
split trivially safe, since each provider is exactly one row and cannot appear on both sides.

**2. Nothing is judged against a threshold of 0.50.**
0.50 is an artifact of how classifiers are usually written up, not a business decision.
The operating point comes from expected cost (Stage 6), and the queue comes from ranking
(Stage 7).

**3. Every model is measured against rule baselines.**
`Rule_ClaimVolume`, `Rule_TotalReimbursed` and `Rule_MeanReimbursed` run through identical
cross-validation. `train.py` reports `model_justified`, and prints a warning if the selected
model fails to beat the best rule by more than one standard error. If that warning appears,
**stop** — the machine learning is not doing anything a sorted spreadsheet could not.

---

## Stages

### 1. `audit.py` — raw file audit
All files are read with `keep_default_na=False`, so category strings such as `None` are
never silently converted to NaN. Reports label integrity, structurally-absent versus truly
missing columns (outpatient claims have no `AdmissionDt` by design), amount sanity, date
span, and how far beneficiaries and physicians span multiple providers. Writes
`outputs/audit_raw.json`.

### 2. `build_features.py` — provider feature table
Aggregates claims and beneficiary data into four groups:

| Group | What it captures | Examples |
|---|---|---|
| `volume` | how big the provider is | claim count, distinct patients |
| `money` | how much and how variably they bill | mean and p90 reimbursement, reimbursement per patient |
| `behaviour` | how they operate | physician concentration, same-day duplicate billing, codes per claim |
| `patient_mix` | who their patients are | mean age, chronic-condition load |

Rules applied:
- Only provider-internal statistics. No target encoding, no cross-provider network features,
  so nothing can leak across the split.
- **Gender and Race are excluded.** Scoring a provider on their patients' demographics is
  not defensible in an investigation.
- Diagnosis and procedure codes become counts and concentration ratios, never one-hot
  columns. One-hot encoding 65,000 codes produces memorisation, not behaviour detection.

### 3. `split.py` — frozen split
Stratified by default. Use `--mode temporal` once you have a `first_claim_date` column;
the real data spans about two years, which makes temporal validation genuinely possible.
Warns when the test set holds under 100 fraud cases.

### 4. `ablation.py` — do the features earn their place?
Runs each feature group alone and each with one group removed. The key comparison is
**all groups versus volume only**: if the gain is under 0.03 PR-AUC, the model is ranking by
provider size rather than provider behaviour, and the script says so.

### 5. `train.py` — selection and one test evaluation
- No resampling anywhere. SMOTE inflates probabilities, and Stages 7 and 8 multiply
  probabilities by money.
- Repeated stratified CV reporting mean **and** standard deviation.
- Selection by the one-standard-error rule: the simplest model within 1 SE of the best,
  rather than declaring a winner on a gap smaller than fold noise.
- Calibration check on out-of-fold probabilities (`outputs/calibration_oof.csv`).
- Test set is touched exactly once; every metric comes with a bootstrap 95% CI.

### 6. `size_confound.py` — check whether the model adds signal beyond provider size`nCompares size-inclusive and size-free models, measures ranking quality within provider-size strata,`n and checks whether expected-exposure ranking adds information beyond total billed. Writes`n`outputs/size_confound.json` and `outputs/size_strata.csv`.`n`n### 7. `threshold_cost.py` — cost-aware operating point
```
cost = (TP + FP) x investigation_cost
     + sum over missed frauds of exposure_amount x recovery_fraction
```
This charges for investigating true frauds (an investigation costs the same whether it finds
something or not) and weights each missed fraud by the money actually at stake. The threshold
is chosen on out-of-fold training probabilities, never on test.

The sensitivity table is the real output. The optimal threshold moves substantially with the
recovery assumption, and you do not have real insurer figures. **Report the range, not a
single number.**

### 8. `rank_topk.py` — the investigation queue
Compares two strategies at realistic capacities (1%, 5%, 10%, 20% of providers):

- **probability**: rank by P(fraud) — catches more providers
- **exposure**: rank by P(fraud) × total reimbursed — recovers more money

Random ordering is included as a floor. Writes `outputs/investigation_queue.csv` with
priority, risk band, and expected exposure per provider.

### 9. `queue_significance.py` — compare the queue with sorting by total billed`nUses paired bootstrap resampling and Holm-Bonferroni correction to test whether model rankings`nbeat the no-model billing rule. Writes `outputs/queue_significance.csv` and `outputs/queue_significance.json`.`n`n### 10. `explain.py` — reasons a human can check
SHAP when available, with automatic fallback to coefficients or impurity importances.
Feature names are translated into plain English (`top_physician_concentration` becomes
"share of claims signed by a single physician"). Language rule enforced throughout: a
feature **contributes to a risk score**; it does not cause or prove fraud.

---

## Outputs

| File | Contents |
|---|---|
| `audit_raw.json` | raw-file integrity and leakage audit |
| `ablation_results.csv` | feature-group evidence |
| `cv_results.csv`, `cv_fold_scores.csv` | per-model CV means, stds, per-fold scores |
| `selection.json` | chosen model, 1-SE reasoning, whether it beats the rules |
| `calibration_oof.csv` | predicted vs observed fraud rate by probability bin |
| `final_test_results.csv`, `test_bootstrap_ci.json` | the single test evaluation |
| `size_confound.json`, `size_strata.csv` | size-confound diagnostics |`n| `threshold_cost_curve.csv`, `threshold_cost_sensitivity.csv` | cost by threshold, and by assumption |
| `topk_ranking.csv`, `investigation_queue.csv` | precision@K, recall@K, the ranked queue |`n| `queue_significance.csv`, `queue_significance.json` | paired significance test against the billing rule |
| `global_importance.csv`, `explanations.json` | global drivers and per-provider reasons |

---

## Limitations to state in the report

1. **The label is `PotentialFraud`, not adjudicated fraud.** Every result predicts a flag
   someone already applied, which may encode the insurer's own investigation biases. Say this
   plainly; it is the single most important caveat.
2. **Provider-level, not claim-level.** The system says which provider to examine, not which
   claim is fraudulent.
3. **Costs are relative units.** Real investigation costs and recovery rates are not
   available, which is why sensitivity analysis replaces a single headline number.
4. **The Kaggle test files carry no labels** and are unusable for evaluation, so the split is
   made from the labelled training file only.
5. **Historical Medicare data.** Not current claims, and not transferable to another
   insurer, country or line of business without revalidation.
6. **Decision support only.** The model ranks; investigators decide. No claim is denied and
   no provider is sanctioned by a model score.

---

## What is deliberately not here

Drift monitoring, retraining automation and a production API are not built, because none
of them can be validated on a static historical extract. The repository includes a local
decision-support dashboard; its generated JSON snapshots are intentionally excluded from
Git because they contain provider-level records. Run the pipeline and then
`python dashboard/prepare_data.py` to generate local dashboard data. Build monitoring and
retraining only when there is a live claim stream to validate them against.

## Repository data policy

Raw claims, processed provider tables, model files, evaluation outputs and generated
dashboard snapshots are ignored by Git. Download the source data separately and keep the
resulting artifacts local. The repository contains code and documentation only.
