"""
Step 6 (Part 4): choose an operating threshold from EXPECTED COST, not from 0.50.

Critical rule: the threshold is chosen on out-of-fold TRAINING probabilities
(outputs/oof_probabilities.csv), then applied once to the frozen test set. Tuning a
threshold on test would make the test set part of model fitting.

Cost model
----------
    cost = (TP + FP) x investigation_cost          # every provider you open a case on
         + sum over missed frauds of
             exposure_amount x recovery_fraction   # money you fail to claw back

Note what this fixes: the naive "FP x cost + FN x cost" version charges nothing for
investigating a true fraud, which is wrong -- an investigation costs the same whether it
finds something or not. It also treats all missed frauds as equally expensive, when
missing a provider reimbursed $4m matters far more than missing one reimbursed $40k.

Costs are RELATIVE UNITS, not real insurer figures. That is why the sensitivity table
matters more than any single "optimal" number.
"""
import json

import joblib
import numpy as np
import pandas as pd

from config import (COST_RATIOS_TO_TEST, INVESTIGATION_COST, MODELS, OUTPUTS, RECOVERY_FRACTION,
                    TARGET)


def expected_cost(y, p, exposure, threshold, inv_cost, recovery):
    flag = p >= threshold
    n_investigated = int(flag.sum())
    missed = (~flag) & (y == 1)
    return {"threshold": round(float(threshold), 3),
            "n_investigated": n_investigated,
            "investigation_rate": round(float(flag.mean()), 4),
            "tp": int((flag & (y == 1)).sum()),
            "fp": int((flag & (y == 0)).sum()),
            "fn": int(missed.sum()),
            "tn": int(((~flag) & (y == 0)).sum()),
            "precision": round(float((flag & (y == 1)).sum() / max(n_investigated, 1)), 4),
            "recall": round(float((flag & (y == 1)).sum() / max((y == 1).sum(), 1)), 4),
            "investigation_cost": round(n_investigated * inv_cost, 2),
            "missed_fraud_cost": round(float(exposure[missed].sum() * recovery), 2),
            "total_cost": round(n_investigated * inv_cost
                                + float(exposure[missed].sum() * recovery), 2)}


def sweep(y, p, exposure, inv_cost, recovery, grid=None):
    grid = grid if grid is not None else np.round(np.arange(0.02, 0.96, 0.01), 3)
    return pd.DataFrame([expected_cost(y, p, exposure, t, inv_cost, recovery) for t in grid])


def main():
    bundle = joblib.load(MODELS / "provider_fraud_model.joblib")
    selected = bundle["selection"]["selected_model"]

    oof = pd.read_csv(OUTPUTS / "oof_probabilities.csv")
    y = oof[TARGET].to_numpy()
    p = oof[f"p_{selected}"].to_numpy()
    exposure = oof["exposure_amount"].to_numpy()

    # ---- main sweep at the configured cost assumptions --------------------------
    curve = sweep(y, p, exposure, INVESTIGATION_COST, RECOVERY_FRACTION)
    curve.to_csv(OUTPUTS / "threshold_cost_curve.csv", index=False)
    best = curve.loc[curve.total_cost.idxmin()]
    always = expected_cost(y, p, exposure, 0.0, INVESTIGATION_COST, RECOVERY_FRACTION)
    never = expected_cost(y, p, exposure, 1.01, INVESTIGATION_COST, RECOVERY_FRACTION)
    at_half = expected_cost(y, p, exposure, 0.5, INVESTIGATION_COST, RECOVERY_FRACTION)

    print(f"Model: {selected}   investigation_cost={INVESTIGATION_COST:,.0f}  "
          f"recovery_fraction={RECOVERY_FRACTION}")
    print("\nReference points (out-of-fold training data):")
    for label, row in [("investigate everyone", always), ("investigate nobody", never),
                       ("threshold 0.50", at_half)]:
        print(f"  {label:22s} investigated {row['n_investigated']:>4}  "
              f"recall {row['recall']:.3f}  total cost {row['total_cost']:>14,.0f}")
    print(f"  {'COST-OPTIMAL':22s} investigated {int(best.n_investigated):>4}  "
          f"recall {best.recall:.3f}  total cost {best.total_cost:>14,.0f}  "
          f"at threshold {best.threshold}")
    saving = never["total_cost"] - best.total_cost
    print(f"\nSaving vs investigating nobody: {saving:,.0f} "
          f"({saving / max(never['total_cost'], 1):.1%})")

    # ---- sensitivity: the threshold moves a lot with the cost assumption ---------
    sens = []
    for r in COST_RATIOS_TO_TEST:
        c = sweep(y, p, exposure, INVESTIGATION_COST, r)
        b = c.loc[c.total_cost.idxmin()]
        sens.append({"recovery_fraction": r, "optimal_threshold": b.threshold,
                     "investigation_rate": b.investigation_rate, "recall": b.recall,
                     "precision": b.precision, "n_investigated": int(b.n_investigated),
                     "total_cost": b.total_cost})
    sens = pd.DataFrame(sens)
    sens.to_csv(OUTPUTS / "threshold_cost_sensitivity.csv", index=False)
    print("\nSensitivity to the recovery assumption:")
    print(sens.to_string(index=False))
    print("\nRead this as: the 'optimal' threshold is a function of an assumption you do not "
          "have real data for. Report the range, not one number.")

    chosen = float(best.threshold)
    bundle["cost_threshold"] = chosen
    bundle["cost_assumptions"] = {"investigation_cost": INVESTIGATION_COST,
                                  "recovery_fraction": RECOVERY_FRACTION}
    joblib.dump(bundle, MODELS / "provider_fraud_model.joblib")
    (OUTPUTS / "threshold_choice.json").write_text(json.dumps(
        {"selected_model": selected, "chosen_threshold": chosen,
         "chosen_on": "out-of-fold training probabilities",
         "assumptions": bundle["cost_assumptions"],
         "optimal_threshold_range_across_assumptions":
             [float(sens.optimal_threshold.min()), float(sens.optimal_threshold.max())]}, indent=2))
    print(f"\nSaved threshold {chosen} -> models/provider_fraud_model.joblib")


if __name__ == "__main__":
    main()
