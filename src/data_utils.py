"""
Data loading, preprocessing, stratified splitting, and PyTorch DataLoader utilities
for the ULB Credit Card Fraud Detection dataset.
"""

import os
import random
from typing import Dict, Tuple, List
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import torch
from torch.utils.data import TensorDataset, DataLoader


SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Sets random seeds across Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_and_preprocess_data(
    csv_path: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = SEED,
    save_artifacts_dir: str = None,
) -> Dict:
    """
    Loads `creditcard.csv`, performs a stratified 70% / 15% / 15% Train/Val/Test split,
    and scales `Time` and `Amount` using RobustScaler fitted strictly on the training split
    to prevent any data leakage.
    """
    set_seed(seed)
    df = pd.read_csv(csv_path)

    feature_cols: List[str] = [c for c in df.columns if c != "Class"]
    X = df[feature_cols].copy()
    y = df["Class"].values.astype(np.int64)

    # First split: separate Test set (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    # Second split: separate Validation set (15% of total -> val_size / (1 - test_size) of temp)
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, stratify=y_temp, random_state=seed
    )

    # Fit RobustScaler ONLY on training data for 'Time' and 'Amount'
    scaler = RobustScaler()
    scale_cols = ["Time", "Amount"]

    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_val_scaled[scale_cols] = scaler.transform(X_val[scale_cols])
    X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])

    if save_artifacts_dir:
        os.makedirs(save_artifacts_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(save_artifacts_dir, "robust_scaler.joblib"))

    # Compute class imbalance statistics on training set
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    pos_weight = float(n_neg) / float(n_pos)

    return {
        "df_raw": df,
        "feature_cols": feature_cols,
        "scaler": scaler,
        "X_train_df": X_train_scaled,
        "X_val_df": X_val_scaled,
        "X_test_df": X_test_scaled,
        "X_test_raw_df": X_test,
        "X_train": X_train_scaled.values.astype(np.float32),
        "y_train": y_train,
        "X_val": X_val_scaled.values.astype(np.float32),
        "y_val": y_val,
        "X_test": X_test_scaled.values.astype(np.float32),
        "y_test": y_test,
        "n_neg_train": n_neg,
        "n_pos_train": n_pos,
        "pos_weight": pos_weight,
    }


def create_dataloaders(
    data_dict: Dict,
    batch_size: int = 1024,
    seed: int = SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates PyTorch DataLoaders for training, validation, and testing."""
    g = torch.Generator()
    g.manual_seed(seed)

    train_ds = TensorDataset(
        torch.from_numpy(data_dict["X_train"]),
        torch.from_numpy(data_dict["y_train"].astype(np.float32)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(data_dict["X_val"]),
        torch.from_numpy(data_dict["y_val"].astype(np.float32)),
    )
    test_ds = TensorDataset(
        torch.from_numpy(data_dict["X_test"]),
        torch.from_numpy(data_dict["y_test"].astype(np.float32)),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)

    return train_loader, val_loader, test_loader
