"""Inspect, score, and prepare the student stress dataset."""

from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "dataset" / "raw" / "Stress Prediction Form.csv"
PROCESSED_FILE = ROOT / "dataset" / "processed" / "cleaned_stress_dataset.csv"
PSS_COLUMNS = [f"PSS_Q{i}" for i in range(1, 11)]
REVERSE_ITEMS = ["PSS_Q4", "PSS_Q5", "PSS_Q7", "PSS_Q8"]


def score_pss10(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with standard PSS-10 total and specified stress classes.

    Positively worded items Q4, Q5, Q7, and Q8 are reversed as 4 - response.
    The specified research categories are Low (0-13), Moderate (14-26), and
    High (27-40). Original item responses remain unchanged in ``data``.
    """
    missing = sorted(set(PSS_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Required PSS-10 columns are missing: {missing}")

    scored = data.copy()
    pss = scored[PSS_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if pss.isna().any().any():
        raise ValueError("PSS-10 responses contain missing or non-numeric values")
    if not pss.isin(range(5)).all().all():
        raise ValueError("PSS-10 responses must be integer values from 0 through 4")

    for column in REVERSE_ITEMS:
        pss[column] = 4 - pss[column]
    scored["PSS10_Total"] = pss.sum(axis=1).astype(int)
    if not scored["PSS10_Total"].between(0, 40).all():
        raise ValueError("Calculated PSS-10 total is outside the valid 0-40 range")
    scored["Stress_Level"] = pd.cut(
        scored["PSS10_Total"],
        bins=[-1, 13, 26, 40],
        labels=["Low Stress", "Moderate Stress", "High Stress"],
    ).astype(str)
    return scored


def inspect_dataset(data: pd.DataFrame) -> None:
    """Print concise descriptive checks before preprocessing decisions."""
    print(f"Dataset shape: {data.shape[0]} rows x {data.shape[1]} columns")
    print("Columns and dtypes:")
    print(data.dtypes.to_string())
    print(f"Missing values: {int(data.isna().sum().sum())}")
    print(f"Duplicate rows: {int(data.duplicated().sum())}")
    print("Categorical cardinalities:")
    categorical = data.select_dtypes(include=["object", "category"])
    print(categorical.nunique(dropna=False).to_string())


def main() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_FILE}")
    raw = pd.read_csv(RAW_FILE)
    inspect_dataset(raw)

    clean = raw.drop_duplicates().copy()
    # The inspected source currently has no missing values. Retain rows and
    # let model pipelines impute any future predictor missingness in-fold.
    clean = score_pss10(clean)
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(PROCESSED_FILE, index=False)
    print(f"Scored rows: {len(clean)}")
    print(f"PSS10_Total range: {clean['PSS10_Total'].min()}-{clean['PSS10_Total'].max()}")
    print("Stress-level counts:")
    class_distribution = clean["Stress_Level"].value_counts().sort_index()
    print(class_distribution.to_string())
    print(f"Saved processed dataset: {PROCESSED_FILE.relative_to(ROOT)}")
    metrics_directory = ROOT / "results" / "metrics"
    metrics_directory.mkdir(parents=True, exist_ok=True)
    distribution_table = class_distribution.rename("count").to_frame()
    distribution_table["proportion"] = distribution_table["count"] / len(clean)
    distribution_table.to_csv(metrics_directory / "class_distribution.csv")
    figure_directory = ROOT / "results" / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    clean["Stress_Level"].value_counts().reindex(
        ["Low Stress", "Moderate Stress", "High Stress"]
    ).plot(kind="bar", rot=0, color="#4472C4")
    plt.ylabel("Responses")
    plt.xlabel("Stress level")
    plt.title("Stress level class distribution")
    plt.tight_layout()
    plt.savefig(figure_directory / "class_distribution.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
