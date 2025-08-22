import sys
import argparse
from pathlib import Path

import pandas as pd

if not __package__:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from challenge.model import DelayModel


def main():
    parser = argparse.ArgumentParser(
        description="Train DelayModel and persist artifact."
    )
    parser.add_argument("--data", default="data/data.csv", help="Training CSV path")
    parser.add_argument(
        "--output", default="models/delay_model.pkl", help="Where to save the model"
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    df = pd.read_csv(data_path)
    model = DelayModel()

    X, y = model.preprocess(df, target_column="delay")
    model.fit(X, y)

    out = Path(args.output)
    model.save(out)

    print(f"[train_model] Saved model to {out.resolve()}")


if __name__ == "__main__":
    main()
