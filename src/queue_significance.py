"""
Final check: is the model's advantage over a sorted spreadsheet real, or noise?

rank_topk.py reports that the model catches a few more fraudulent providers than simply
sorting by total billed. "A few more" out of several hundred needs a significance test
before it goes in a report.

Method: paired bootstrap. Resample providers with replacement; on each resample, rebuild
BOTH queues and take the difference in precision@K. Pairing matters -- the two queues share
most of their members, so an unpaired comparison badly overstates the uncertainty.

Reads outputs/oof_probabilities.csv (out-of-fold, so no test-set contamination).

Usage:
    python src/queue_significance.py
    python src/queue_significance.py --n-boot 5000
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from config import INVESTIGATOR_CAPACITY_PCT, MODELS, OUTPUTS, SEED, TARGET

RULE_COL = "exposure_amount"       # total reimbursed: the no-model baseline


def precision_at_k(y, score, k):
    idx = np.argsort(-score, kind="stable")[:k]
    return float(y[idx].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    selected = joblib.load(MODELS / "provider_fraud_model.joblib")["selection"]["selected_model"]
    df = pd.read_csv(OUTPUTS / "oof_probabilities.csv")
    y = df[TARGET].to_numpy()
    p = df[f"p_{selected}"].to_numpy(dtype=float)
    amount = df[RULE_COL].to_numpy(dtype=float)

    strategies = {"probability": p, "exposure": p * amount}
    rng = np.random.default_rng(SEED)
    n = len(y)
    rows = []

    print(f"Model: {selected}   providers: {n}   fraud: {y.sum()} ({y.mean():.1%})")
    print(f"Paired bootstrap, {a.n_boot} resamples\n")

    # pre-draw resample indices so every comparison uses identical resamples
    boots = [rng.integers(0, n, n) for _ in range(a.n_boot)]

    for pct in INVESTIGATOR_CAPACITY_PCT:
        k = max(1, int(round(pct * n)))
        base_rule = precision_at_k(y, amount, k)
        for name, score in strategies.items():
            base_model = precision_at_k(y, score, k)
            diffs = np.empty(a.n_boot)
            for i, idx in enumerate(boots):
                yb, kb = y[idx], k
                diffs[i] = precision_at_k(yb, score[idx], kb) - precision_at_k(yb, amount[idx], kb)
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            # Ties (diff exactly 0) are common: the two queues often select the same
            # providers. Counting ties in BOTH tails makes 2*min(...) exceed 1, so split
            # the tied mass evenly between the tails.
            frac_below = (diffs < 0).mean() + 0.5 * (diffs == 0).mean()
            p_two_sided = float(min(1.0, 2 * min(frac_below, 1 - frac_below)))
            rows.append({"capacity_pct": pct, "k": k, "strategy": name,
                         "precision_model": round(base_model, 4),
                         "precision_rule": round(base_rule, 4),
                         "lift": round(base_model - base_rule, 4),
                         "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
                         "p_value": round(p_two_sided, 4),
                         "pct_resamples_tied": round(float((diffs == 0).mean()), 3)})
            print(f"  capacity {pct:>5.0%} (k={k:>4})  {name:<12s} "
                  f"lift {base_model - base_rule:+.4f}   "
                  f"95% CI [{lo:+.4f}, {hi:+.4f}]   p={p_two_sided:.3f}")

    res = pd.DataFrame(rows)

    # Every capacity x strategy cell is a separate hypothesis test. Testing 8 of them at
    # alpha=0.05 produces roughly 0.4 false positives by chance, so an uncorrected "one cell
    # was significant" is not evidence. Holm-Bonferroni controls the family-wise error rate.
    m = len(res)
    order = res.p_value.to_numpy().argsort()
    holm = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * res.p_value.iloc[i])
        holm[i] = min(1.0, running)
    res["p_holm"] = holm.round(4)
    res["beats_rule"] = np.where(res.p_holm < 0.05,
                                 np.where(res.lift > 0, "YES", "WORSE"), "no")
    res.to_csv(OUTPUTS / "queue_significance.csv", index=False)

    print(f"\nAfter Holm-Bonferroni correction for {m} comparisons:")
    for _, r in res.iterrows():
        print(f"  capacity {r.capacity_pct:>5.0%}  {r.strategy:<12s} "
              f"lift {r.lift:+.4f}   raw p={r.p_value:.3f}   corrected p={r.p_holm:.3f}   "
              f"beats rule: {r.beats_rule}")

    any_sig = (res.beats_rule == "YES").any()
    verdict = (
        "At least one capacity shows an improvement over sorting by total billed that "
        "survives correction for multiple comparisons." if any_sig else
        "NO capacity shows a reliable improvement over sorting providers by total billed. "
        "The apparent lift does not survive correction for multiple comparisons and is within "
        "sampling noise. Report this plainly: on this dataset the GLOBAL investigation queue "
        "does not need a model. The model's measurable value is within-size-band "
        "discrimination (see size_confound.py), not the overall ranking.")
    (OUTPUTS / "queue_significance.json").write_text(json.dumps(
        {"model": selected, "n_boot": a.n_boot, "n_comparisons": m,
         "correction": "Holm-Bonferroni, family-wise alpha = 0.05",
         "verdict": verdict, "results": res.to_dict("records")}, indent=2, default=str))
    print(f"\nVERDICT: {verdict}")
    print(f"\nSaved {OUTPUTS / 'queue_significance.csv'}")


if __name__ == "__main__":
    main()
