import os
import urllib.request
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CSV_PATH = os.path.join(DATA_DIR, "creditcard.csv")
TF_MIRROR_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"


def ensure_dataset(csv_path: str = CSV_PATH) -> str:
    """Downloads creditcard.csv if missing and checks row/fraud counts."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) < 100_000_000:
        print(f"Downloading dataset to {csv_path}...")
        urllib.request.urlretrieve(TF_MIRROR_URL, csv_path)
        print(f"Download finished ({os.path.getsize(csv_path) / (1024**2):.2f} MB).")
    else:
        print(f"Dataset found at {csv_path} ({os.path.getsize(csv_path) / (1024**2):.2f} MB).")

    df = pd.read_csv(csv_path)
    n_rows, n_cols = df.shape
    n_frauds = int(df["Class"].sum())
    print(f"Dataset shape: ({n_rows}, {n_cols}), Frauds: {n_frauds} ({100.0 * n_frauds / n_rows:.4f}%)")
    assert n_rows == 284807 and n_cols == 31, f"Unexpected dataset shape: {df.shape}"
    assert n_frauds == 492, f"Unexpected fraud count: {n_frauds}"
    return csv_path


if __name__ == "__main__":
    ensure_dataset()
