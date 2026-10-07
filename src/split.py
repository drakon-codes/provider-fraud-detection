"""
Step 3: one frozen train/test split at PROVIDER level.

Each provider is exactly one row, so a random split cannot place the same provider on
both sides. No preprocessing happens here: imputation and scaling are fitted inside
each CV fold, so there is only one place where that can go wrong.

After this runs, do not look at test metrics again until every decision is final.
"""
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from config import PROVIDERS_CSV, SEED, TARGET, TEST_CSV, TEST_SIZE, TRAIN_CSV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["stratified", "temporal"], default="stratified",
                    help="temporal requires a first_claim_date column; see README")
    a = ap.parse_args()

    df = pd.read_csv(PROVIDERS_CSV, keep_default_na=False, na_values=[""])

    if a.mode == "temporal":
        if "first_claim_date" not in df.columns:
            raise SystemExit("temporal split needs a 'first_claim_date' column in the provider table")
        df = df.sort_values("first_claim_date")
        cut = int(len(df) * (1 - TEST_SIZE))
        train, test = df.iloc[:cut], df.iloc[cut:]
    else:
        train, test = train_test_split(df, test_size=TEST_SIZE, stratify=df[TARGET],
                                       random_state=SEED)

    train.to_csv(TRAIN_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)

    for name, part in [("train", train), ("test", test)]:
        print(f"{name}: {len(part)} providers, fraud = {int(part[TARGET].sum())} "
              f"({part[TARGET].mean():.1%})")

    n_test_pos = int(test[TARGET].sum())
    if n_test_pos < 100:
        print(f"\nWARNING: only {n_test_pos} fraud cases in the test set. Test metrics will be "
              f"noisy; treat the repeated-CV results as the primary evidence.")
    print("\nTest split is now frozen.")


if __name__ == "__main__":
    main()
