"""
Training, validation, threshold optimization, and evaluation routines for Credit Card Fraud Detection.
"""

import copy
import time
from typing import Dict, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    matthews_corrcoef,
    balanced_accuracy_score,
    brier_score_loss,
)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_dataloader(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Runs inference on a DataLoader and returns (mean_loss, y_true, y_prob)."""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_probs = []
    all_targets = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        batch_size = x_batch.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_targets.append(y_batch.detach().cpu().numpy())

    mean_loss = total_loss / max(1, n_samples)
    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_targets, axis=0).astype(np.int64)
    return mean_loss, y_true, y_prob


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """
    Finds the probability threshold tau* on the validation set that maximizes the F1-score.
    Returns (best_threshold, best_val_f1).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # Compute F1 for each threshold
    f1_scores = (2.0 * precisions[:-1] * recalls[:-1]) / np.clip(
        precisions[:-1] + recalls[:-1], 1e-12, None
    )
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, Any]:
    """Computes a full suite of classification metrics at a specified decision threshold."""
    y_pred = (y_prob >= threshold).astype(np.int64)

    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    brier = float(brier_score_loss(y_true, y_prob))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = float(tn / max(1, (tn + fp)))

    # Compute F1 at threshold=0.5 as well for reference
    y_pred_05 = (y_prob >= 0.5).astype(np.int64)
    f1_05 = float(f1_score(y_true, y_pred_05, zero_division=0))
    prec_05 = float(precision_score(y_true, y_pred_05, zero_division=0))
    rec_05 = float(recall_score(y_true, y_pred_05, zero_division=0))

    return {
        "threshold": float(threshold),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "mcc": mcc,
        "balanced_accuracy": bal_acc,
        "specificity": specificity,
        "brier_score": brier,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "f1_at_05": f1_05,
        "precision_at_05": prec_05,
        "recall_at_05": rec_05,
    }


def train_mlp_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 35,
    patience: int = 8,
    device: torch.device = None,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Trains a FraudMLP model with AdamW, ReduceLROnPlateau, gradient clipping,
    and Early Stopping monitored on Validation PR-AUC (Average Precision).
    """
    if device is None:
        device = get_device()
    model = model.to(device)
    criterion = criterion.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_pr_auc": [],
        "val_roc_auc": [],
        "val_f1_opt": [],
        "lr": [],
    }

    best_val_pr_auc = -1.0
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            batch_size = x_batch.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size

        train_loss = running_loss / max(1, n_samples)
        val_loss, y_val_true, y_val_prob = predict_dataloader(model, val_loader, criterion, device)
        val_pr_auc = float(average_precision_score(y_val_true, y_val_prob))
        val_roc_auc = float(roc_auc_score(y_val_true, y_val_prob))
        _, val_f1_opt = find_optimal_threshold(y_val_true, y_val_prob)

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_pr_auc"].append(val_pr_auc)
        history["val_roc_auc"].append(val_roc_auc)
        history["val_f1_opt"].append(val_f1_opt)
        history["lr"].append(current_lr)

        scheduler.step(val_pr_auc)

        if val_pr_auc > best_val_pr_auc + 1e-5:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch == 1 or epoch % 5 == 0 or epochs_no_improve == 0):
            print(
                f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.5f} | "
                f"Val Loss: {val_loss:.5f} | Val PR-AUC: {val_pr_auc:.4f} | "
                f"Val ROC-AUC: {val_roc_auc:.4f} | Val F1*: {val_f1_opt:.4f}"
            )

        if epochs_no_improve >= patience:
            if verbose:
                print(f"  [Early Stopping] Triggered at epoch {epoch} (Best epoch: {best_epoch}, Val PR-AUC: {best_val_pr_auc:.4f})")
            break

    elapsed_sec = time.time() - start_time
    model.load_state_dict(best_weights)
    history["best_epoch"] = best_epoch
    history["best_val_pr_auc"] = best_val_pr_auc
    history["training_time_sec"] = elapsed_sec

    return model, history
