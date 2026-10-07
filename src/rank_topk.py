"""
Step 7 (Part 5): rank providers for a limited investigation queue.

The operational question is not "who is above 0.5" but "we can open 20 cases this month,
which 20?". Two ranking strategies are compared:

    probability     rank by P(fraud)
    exposure        rank by P(fraud) x total reimbursed
    banded          split capacity across size bands, rank by P(fraud) within each band

Exposure ranking will usually recover more money at the same capacity, while catching
slightly fewer providers -- it prefers one large suspicious provider over three small ones.

BANDED ranking exists because of a problem src/size_confound.py detects: when fraud is
concentrated in large providers, P(fraud) becomes largely a function of size, and
P(fraud) x amount degenerates towards ranking by amount alone. A queue built that way only
ever audits big billers, which reproduces whatever audit policy created the labels and can
never discover fraud among smaller providers. Banded ranking allocates capacity across size
bands and ranks within each, so the model competes against like-for-like.

All three are reported so the trade-off is explicit rather than assumed.

Everything is measured on out-of-fold training probabilities first. The test set is used
once, at the end, for confirmation.
"""
import json

import joblib
import numpy as np
import pandas as pd

from config import ID_COL, INVESTIGATOR_CAPACITY_PCT, MODELS, OUTPUTS, TARGET

N_SIZE_BANDS = 5


def rank_metrics(df, score_col, k):
    top = df.nlargest(k, score_col)
    n_fraud_total = int(df[TARGET].sum())
    caught = int(top[TARGET].sum())
    return {"k": k,
            "capacity_pct": round(k / len(df), 4),
            "precision_at_k": round(caught / max(k, 1), 4),
            "recall_at_k": round(caught / max(n_fraud_total, 1), 4),
            "fraud_caught": caught,
            "exposure_captured": round(float(top.loc[top[TARGET] == 1, "exposure_amount"].sum()), 2),
            "pct_fraud_exposure_captured": round(
                float(top.loc[top[TARGET] == 1, "exposure_amount"].sum()
                      / max(df.loc[df[TARGET] == 1, "exposure_amount"].sum(), 1)), 4)}


def banded_selection(df, k, n_bands=N_SIZE_BANDS):
    """Split capacity evenly across size bands; take the top P(fraud) inside each."""
    bands = pd.qcut(df["exposure_amount"].rank(method="first"), n_bands, labels=False)
    per_band = max(1, k // n_bands)
    picks = [g.nlargest(per_band, "p_fraud") for _, g in df.groupby(bands)]
    out = pd.concat(picks)
    if len(out) < k:                      # top up with the best remaining overall
        rest = df.drop(out.index).nlargest(k - len(out), "p_fraud")
        out = pd.concat([out, rest])
    return out.head(k)


def banded_metrics(df, k):
    top = banded_selection(df, k)
    n_fraud_total = int(df[TARGET].sum())
    caught = int(top[TARGET].sum())
    fraud_exposure = float(df.loc[df[TARGET] == 1, "exposure_amount"].sum())
    return {"k": k, "capacity_pct": round(k / len(df), 4),
            "precision_at_k": round(caught / max(k, 1), 4),
            "recall_at_k": round(caught / max(n_fraud_total, 1), 4),
            "fraud_caught": caught,
            "exposure_captured": round(float(top.loc[top[TARGET] == 1, "exposure_amount"].sum()), 2),
            "pct_fraud_exposure_captured": round(
                float(top.loc[top[TARGET] == 1, "exposure_amount"].sum() / max(fraud_exposure, 1)), 4)}


def compare(df, label):
    df = df.copy()
    df["score_probability"] = df["p_fraud"]
    df["score_exposure"] = df["p_fraud"] * df["exposure_amount"]
    rows = []
    for pct in INVESTIGATOR_CAPACITY_PCT:
        k = max(1, int(round(pct * len(df))))
        for strategy, col in [("probability", "score_probability"), ("exposure", "score_exposure")]:
            rows.append({"dataset": label, "strategy": strategy, **rank_metrics(df, col, k)})
        rows.append({"dataset": label, "strategy": "banded", **banded_metrics(df, k)})
        # THE decisive operational baseline: no model at all, just sort by money billed.
        # If the model cannot beat this at realistic capacity, it is not earning its place.
        rows.append({"dataset": label, "strategy": "rule_total_billed_only",
                     **rank_metrics(df, "exposure_amount", k)})
        # random-ordering reference at the same capacity
        rng = np.random.default_rng(0)
        rand = [rank_metrics(df.assign(r=rng.random(len(df))), "r", k) for _ in range(200)]
        rows.append({"dataset": label, "strategy": "random",
                     **{kk: round(float(np.mean([r[kk] for r in rand])), 4)
                        for kk in rand[0]}})
    return pd.DataFrame(rows)


def main():
    bundle = joblib.load(MODELS / "provider_fraud_model.joblib")
    selected = bundle["selection"]["selected_model"]

    oof = pd.read_csv(OUTPUTS / "oof_probabilities.csv").rename(columns={f"p_{selected}": "p_fraud"})
    test = pd.read_csv(OUTPUTS / "test_probabilities.csv")

    res = pd.concat([compare(oof, "out_of_fold_train"), compare(test, "test")], ignore_index=True)
    res["k"] = res["k"].astype(int)
    res.to_csv(OUTPUTS / "topk_ranking.csv", index=False)

    show = res[["dataset", "strategy", "capacity_pct", "k", "precision_at_k", "recall_at_k",
                "pct_fraud_exposure_captured"]]
    print("Top-K investigation ranking\n")
    print(show.to_string(index=False))

    # headline comparison at each capacity, on out-of-fold data
    print("\nAt each capacity (out-of-fold): recall vs share of fraudulent exposure captured")
    o = res[res.dataset == "out_of_fold_train"]
    for pct in INVESTIGATOR_CAPACITY_PCT:
        sub = o[o.capacity_pct.round(3) == round(pct, 3)]
        if sub.empty:
            continue
        pr = sub[sub.strategy == "probability"].iloc[0]
        ex = sub[sub.strategy == "exposure"].iloc[0]
        bd = sub[sub.strategy == "banded"].iloc[0]
        rl = sub[sub.strategy == "rule_total_billed_only"].iloc[0]
        rn = sub[sub.strategy == "random"].iloc[0]
        print(f"  capacity {pct:>5.0%} (k={int(pr.k):>4}):  "
              f"prob {pr.recall_at_k:.2f}/{pr.pct_fraud_exposure_captured:.2f}  "
              f"exposure {ex.recall_at_k:.2f}/{ex.pct_fraud_exposure_captured:.2f}  "
              f"banded {bd.recall_at_k:.2f}/{bd.pct_fraud_exposure_captured:.2f}  "
              f"NO-MODEL-rule {rl.recall_at_k:.2f}/{rl.pct_fraud_exposure_captured:.2f}  "
              f"random {rn.recall_at_k:.2f}")
    print("  (each pair is recall@K / share of fraudulent exposure captured)")

    # ---- did the model beat the no-model rule? ------------------------------------
    print("\nDOES THE MODEL BEAT A SORTED SPREADSHEET? (precision@K, out-of-fold)")
    verdicts = []
    for pct in INVESTIGATOR_CAPACITY_PCT:
        sub = o[o.capacity_pct.round(3) == round(pct, 3)]
        if sub.empty:
            continue
        best_model = sub[sub.strategy.isin(["probability", "exposure", "banded"])]
        bm = best_model.loc[best_model.precision_at_k.idxmax()]
        rl = sub[sub.strategy == "rule_total_billed_only"].iloc[0]
        lift = bm.precision_at_k - rl.precision_at_k
        verdicts.append(lift)
        print(f"  capacity {pct:>5.0%}:  best model strategy ({bm.strategy}) {bm.precision_at_k:.3f}"
              f"   vs sort-by-money {rl.precision_at_k:.3f}   lift {lift:+.3f}")
    if verdicts and max(verdicts) < 0.03:
        print("\n  The model does not beat sorting providers by total billed. Report that "
              "plainly:\n  on this dataset the operational value is in the size-band analysis, "
              "not the global queue.")

    # the investigator-facing queue
    queue = test.copy()
    queue["expected_exposure"] = queue["p_fraud"] * queue["exposure_amount"]
    queue = queue.sort_values("expected_exposure", ascending=False)
    queue["priority"] = range(1, len(queue) + 1)
    queue["risk_band"] = pd.cut(queue["p_fraud"], [-.01, .10, .30, 1.01],
                                labels=["LOW", "MEDIUM", "HIGH"])
    queue[[ "priority", ID_COL, "p_fraud", "exposure_amount", "expected_exposure",
            "risk_band", TARGET]].round(4).to_csv(OUTPUTS / "investigation_queue.csv", index=False)

    (OUTPUTS / "ranking_summary.json").write_text(json.dumps({
        "selected_model": selected,
        "strategies": ["probability", "exposure = P(fraud) x total reimbursed"],
        "note": ("Exposure ranking prioritises money at risk over case count. Report both; the "
                 "right choice depends on whether the team is measured on cases closed or on "
                 "recoveries."),
    }, indent=2))
    print(f"\nSaved {OUTPUTS / 'topk_ranking.csv'} and {OUTPUTS / 'investigation_queue.csv'}")


if __name__ == "__main__":
    main()
