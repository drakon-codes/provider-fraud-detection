#!/usr/bin/env bash
# Full pipeline, in order. Stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src"

step () { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

if [ ! -e data/raw/Train-*.csv ] 2>/dev/null; then
  step "No raw data found - generating SYNTHETIC sample (not for reporting)"
  python src/make_sample_data.py
fi

step "1. Audit raw files";            python src/audit.py
step "2. Build provider features";    python src/build_features.py
step "3. Freeze train/test split";    python src/split.py
step "4. Feature-group ablation";     python src/ablation.py
step "5. Train, select, test once";   python src/train.py
step "6. Size-confound diagnostic";   python src/size_confound.py
step "7. Cost-aware threshold";       python src/threshold_cost.py
step "8. Top-K ranking";              python src/rank_topk.py
step "9. Queue significance test";    python src/queue_significance.py
step "10. Explanations";              python src/explain.py

printf '\n\033[1mDone.\033[0m Results in outputs/\n'
