"""
Publication-quality figure generation (300 DPI) for the Credit Card Fraud Detection
Technical Report, Jupyter Notebook, Presentation, and Streamlit Dashboard.
"""

import os
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

MODEL_COLORS = {
    "Baseline MLP (BCE)": "#64748B",
    "Weighted MLP (Cost-Sensitive)": "#F59E0B",
    "Focal Loss MLP": "#3B82F6",
    "Optuna Focal MLP": "#10B981",
    "XGBoost (Baseline)": "#8B5CF6",
}


def plot_eda_overview(df: pd.DataFrame, out_path: str) -> None:
    """Generates Figure 1: 4-panel Exploratory Data Analysis of ULB Credit Card Dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Exploratory Data Analysis — ULB Credit Card Fraud Dataset (N = 284,807)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # Panel A: Class Distribution (Log Scale)
    ax = axes[0, 0]
    counts = df["Class"].value_counts().sort_index()
    bars = ax.bar(
        ["Legitimate (0)", "Fraudulent (1)"],
        counts.values,
        color=["#2563EB", "#DC2626"],
        width=0.5,
        edgecolor="black",
    )
    ax.set_yscale("log")
    ax.set_title("(a) Severe Class Imbalance (Log Scale)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Transactions (log10)")
    ax.grid(axis="y", linestyle="--")
    for bar, count in zip(bars, counts.values):
        pct = 100.0 * count / len(df)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count * 1.25,
            f"{count:,}\n({pct:.3f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )
    ax.set_ylim(100, 1_500_000)

    # Panel B: Transaction Amount Distribution (log1p)
    ax = axes[0, 1]
    legit_amt = np.log1p(df.loc[df["Class"] == 0, "Amount"])
    fraud_amt = np.log1p(df.loc[df["Class"] == 1, "Amount"])
    sns.kdeplot(legit_amt, ax=ax, label="Legitimate (0)", color="#2563EB", fill=True, alpha=0.25, linewidth=2)
    sns.kdeplot(fraud_amt, ax=ax, label="Fraudulent (1)", color="#DC2626", fill=True, alpha=0.25, linewidth=2)
    ax.set_title("(b) Transaction Amount Distribution — log(1 + Amount [EUR])", fontsize=12, fontweight="bold")
    ax.set_xlabel("log(1 + Amount)")
    ax.set_ylabel("Density")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--")

    # Panel C: Temporal Distribution over 48 Hours
    ax = axes[1, 0]
    hours_legit = df.loc[df["Class"] == 0, "Time"] / 3600.0
    hours_fraud = df.loc[df["Class"] == 1, "Time"] / 3600.0
    ax.hist(hours_legit, bins=48, density=True, alpha=0.45, color="#2563EB", label="Legitimate (0)")
    ax.hist(hours_fraud, bins=48, density=True, alpha=0.55, color="#DC2626", label="Fraudulent (1)")
    ax.set_title("(c) Temporal Transaction Density Across 48 Hours", fontsize=12, fontweight="bold")
    ax.set_xlabel("Elapsed Time (Hours from First Transaction)")
    ax.set_ylabel("Normalized Density")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--")

    # Panel D: Top Discriminative PCA Features Boxplots / Violins
    ax = axes[1, 1]
    top_feats = ["V14", "V17", "V12", "V10", "V4", "V11"]
    # Subsample legitimate class for clean plotting alongside all 492 frauds
    df_legit_sub = df[df["Class"] == 0].sample(n=3000, random_state=42)
    df_fraud_all = df[df["Class"] == 1]
    df_plot = pd.concat([df_legit_sub, df_fraud_all], ignore_index=True)
    df_melted = df_plot.melt(
        id_vars="Class", value_vars=top_feats, var_name="PCA Feature", value_name="Value"
    )
    df_melted["Class Label"] = df_melted["Class"].map({0: "Legitimate", 1: "Fraud"})
    sns.boxplot(
        data=df_melted,
        x="PCA Feature",
        y="Value",
        hue="Class Label",
        ax=ax,
        palette={"Legitimate": "#2563EB", "Fraud": "#DC2626"},
        showfliers=False,
        width=0.6,
    )
    ax.set_title("(d) Distributions of Most Discriminative PCA Components", fontsize=12, fontweight="bold")
    ax.set_xlabel("Principal Component Feature")
    ax.set_ylabel("Component Value (Outliers Omitted)")
    ax.legend(title="Class", frameon=True)
    ax.grid(True, linestyle="--")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def plot_focal_loss_theory(out_path: str) -> None:
    """Generates Figure 2: Focal Loss modulating factor for varying gamma values (Lin et al., 2017)."""
    pt = np.linspace(0.01, 0.99, 300)
    gammas = [0.0, 0.5, 1.0, 2.0, 5.0]
    colors = ["#1E293B", "#0EA5E9", "#3B82F6", "#10B981", "#EF4444"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Loss curves
    ax = axes[0]
    for g, c in zip(gammas, colors):
        fl = -((1.0 - pt) ** g) * np.log(pt)
        lbl = f"γ = {g:.1f} (Standard BCE)" if g == 0 else f"γ = {g:.1f}"
        ax.plot(pt, fl, label=lbl, color=c, linewidth=2.4)
    ax.set_title("(a) Focal Loss FL(p_t) = -(1 - p_t)^γ log(p_t)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Probability of Ground-Truth Class (p_t)")
    ax.set_ylabel("Loss Value")
    ax.set_ylim(0, 5)
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--")
    ax.annotate(
        "Well-classified negatives\n(p_t > 0.8) are strongly\ndown-weighted when γ ≥ 2",
        xy=(0.8, 0.05),
        xytext=(0.45, 2.2),
        arrowprops=dict(arrowstyle="->", color="#10B981", lw=1.5),
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="#ECFDF5", ec="#10B981"),
    )

    # Right: Relative Weighting Factor (1 - p_t)^gamma
    ax = axes[1]
    for g, c in zip(gammas, colors):
        mod = (1.0 - pt) ** g
        ax.plot(pt, mod, label=f"γ = {g:.1f}", color=c, linewidth=2.4)
    ax.set_title("(b) Modulating Factor (1 - p_t)^γ Across Confidence Levels", fontsize=12, fontweight="bold")
    ax.set_xlabel("Probability of Ground-Truth Class (p_t)")
    ax.set_ylabel("Modulating Factor Weight")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def plot_training_histories(histories: Dict[str, Dict[str, Any]], out_path: str) -> None:
    """Generates Figure 3: Training Loss and Validation PR-AUC learning curves across MLP variants."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Validation PR-AUC across epochs
    ax = axes[0]
    for name, hist in histories.items():
        epochs = range(1, len(hist["val_pr_auc"]) + 1)
        color = MODEL_COLORS.get(name, "#333333")
        ax.plot(epochs, hist["val_pr_auc"], marker="o", markersize=4, linewidth=2.2, label=name, color=color)
        best_ep = hist["best_epoch"]
        best_val = hist["best_val_pr_auc"]
        ax.scatter([best_ep], [best_val], s=90, color=color, edgecolors="black", zorder=5)

    ax.set_title("(a) Validation PR-AUC (AUPRC) Trajectory Across Epochs", fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Validation PR-AUC")
    ax.set_ylim(0.65, 0.90)
    ax.legend(frameon=True, loc="lower right")
    ax.grid(True, linestyle="--")

    # Right: Normalized Training Loss progression
    ax = axes[1]
    for name, hist in histories.items():
        epochs = range(1, len(hist["train_loss"]) + 1)
        color = MODEL_COLORS.get(name, "#333333")
        losses = np.array(hist["train_loss"])
        norm_losses = losses / max(losses[0], 1e-12)
        ax.plot(epochs, norm_losses, linewidth=2.2, label=name, color=color)

    ax.set_title("(b) Normalized Training Loss Convergence (Loss / Initial Loss)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Relative Training Loss (Epoch 1 = 1.0)")
    ax.legend(frameon=True, loc="upper right")
    ax.grid(True, linestyle="--")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def plot_pr_and_roc_curves(
    y_test: np.ndarray,
    test_probs_dict: Dict[str, np.ndarray],
    metrics_dict: Dict[str, Dict[str, Any]],
    out_path: str,
) -> None:
    """Generates Figure 4: Precision-Recall (AUPRC) and ROC Curves on Held-Out Test Set."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Precision-Recall Curve
    ax = axes[0]
    for name, probs in test_probs_dict.items():
        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_auc = metrics_dict[name]["pr_auc"]
        color = MODEL_COLORS.get(name, "#333333")
        ax.plot(
            rec,
            prec,
            linewidth=2.3,
            color=color,
            label=f"{name} (PR-AUC = {pr_auc:.4f})",
        )
    baseline_rate = y_test.mean()
    ax.axhline(baseline_rate, linestyle="--", color="gray", label=f"Random Classifier ({baseline_rate:.4f})")
    ax.set_title("(a) Precision-Recall Curves on Held-Out Test Set", fontsize=12, fontweight="bold")
    ax.set_xlabel("Recall (Sensitivity / Fraud Detection Rate)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=True, loc="lower left", fontsize=9.5)
    ax.grid(True, linestyle="--")

    # Panel B: ROC Curve (Zoomed on low FPR region)
    ax = axes[1]
    for name, probs in test_probs_dict.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = metrics_dict[name]["roc_auc"]
        color = MODEL_COLORS.get(name, "#333333")
        ax.plot(
            fpr,
            tpr,
            linewidth=2.3,
            color=color,
            label=f"{name} (ROC-AUC = {roc_auc:.4f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier (0.5000)")
    ax.set_title("(b) Receiver Operating Characteristic (ROC) Curves", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_xlim(-0.01, 0.25)
    ax.set_ylim(0.70, 1.02)
    ax.legend(frameon=True, loc="lower right", fontsize=9.5)
    ax.grid(True, linestyle="--")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def plot_confusion_matrices(metrics_dict: Dict[str, Dict[str, Any]], out_path: str) -> None:
    """Generates Figure 5: Confusion Matrices across all 5 evaluated models on Test Set."""
    model_names = list(metrics_dict.keys())
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(3.4 * n_models, 3.8))
    if n_models == 1:
        axes = [axes]

    for ax, name in zip(axes, model_names):
        m = metrics_dict[name]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        # Plot log-scaled colors so both 42,640+ TN and ~60 TP are clearly shaded
        cm_log = np.log1p(cm)
        sns.heatmap(
            cm_log,
            annot=False,
            cmap="Blues",
            cbar=False,
            ax=ax,
            linewidths=1.5,
            linecolor="white",
        )
        labels = [
            [f"TN\n{m['tn']:,}", f"FP\n{m['fp']:,}"],
            [f"FN\n{m['fn']:,}", f"TP\n{m['tp']:,}"],
        ]
        for i in range(2):
            for j in range(2):
                text_color = "white" if cm_log[i, j] > 5.0 else "#1E293B"
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    labels[i][j],
                    ha="center",
                    va="center",
                    color=text_color,
                    fontweight="bold",
                    fontsize=10.5,
                )
        ax.set_title(f"{name}\n(F1={m['f1']:.3f}, τ*={m['threshold']:.2f})", fontsize=10, fontweight="bold")
        ax.set_xticklabels(["Pred Legit (0)", "Pred Fraud (1)"], fontsize=9)
        ax.set_yticklabels(["True Legit (0)", "True Fraud (1)"], fontsize=9, rotation=0)

    fig.suptitle(
        "Test Set Confusion Matrices at Validation-Calibrated Thresholds (N_test = 42,722 | 74 Frauds)",
        fontsize=13,
        fontweight="bold",
        y=1.04,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def plot_metric_comparison_bars(metrics_dict: Dict[str, Dict[str, Any]], out_path: str) -> None:
    """Generates Figure 6: Grouped Bar Chart comparing PR-AUC, F1, Precision, Recall, and MCC."""
    rows = []
    for model_name, m in metrics_dict.items():
        rows.append({
            "Model": model_name,
            "PR-AUC": m["pr_auc"],
            "F1-Score": m["f1"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "MCC": m["mcc"],
        })
    df_m = pd.DataFrame(rows)
    df_melt = df_m.melt(id_vars="Model", var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(13, 5.8))
    sns.barplot(
        data=df_melt,
        x="Metric",
        y="Score",
        hue="Model",
        palette=MODEL_COLORS,
        ax=ax,
        edgecolor="black",
    )
    ax.set_title(
        "Comparative Evaluation Across Deep Learning (MLP) Variants and XGBoost on Test Set",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylim(0.55, 1.02)
    ax.set_ylabel("Metric Score")
    ax.set_xlabel("")
    ax.legend(frameon=True, loc="lower right", ncol=2)
    ax.grid(axis="y", linestyle="--")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2, rotation=90)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def plot_optuna_results(trials_df: pd.DataFrame, param_importances: Dict[str, float], out_path: str) -> None:
    """Generates Figure 7: Optuna Optimization Trajectory and Hyperparameter Importance."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # Left: Optimization History
    ax = axes[0]
    trial_nums = trials_df["number"].values
    values = trials_df["value"].values
    best_so_far = np.maximum.accumulate(values)

    ax.scatter(trial_nums, values, color="#3B82F6", s=55, alpha=0.8, label="Trial Validation PR-AUC", zorder=3)
    ax.plot(trial_nums, best_so_far, color="#EF4444", linewidth=2.5, label="Best Validation PR-AUC So Far", zorder=4)
    ax.set_title("(a) Optuna TPE Optimization History Across Trials", fontsize=12, fontweight="bold")
    ax.set_xlabel("Optuna Trial Number")
    ax.set_ylabel("Validation PR-AUC (Objective)")
    ax.legend(frameon=True, loc="lower right")
    ax.grid(True, linestyle="--")

    # Right: Hyperparameter Importances
    ax = axes[1]
    sorted_items = sorted(param_importances.items(), key=lambda x: x[1], reverse=True)
    p_names = [k for k, _ in sorted_items][::-1]
    p_vals = [v for _, v in sorted_items][::-1]
    bars = ax.barh(p_names, p_vals, color="#10B981", edgecolor="black", height=0.55)
    ax.set_title("(b) Hyperparameter Importance (fANOVA / Correlation)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Relative Importance Score")
    ax.set_xlim(0, max(p_vals) * 1.22 if p_vals else 1.0)
    ax.grid(axis="x", linestyle="--")
    for bar, v in zip(bars, p_vals):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontweight="bold", fontsize=9.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def plot_shap_explanations(
    shap_values: np.ndarray,
    X_explain_df: pd.DataFrame,
    y_explain: np.ndarray,
    probs_explain: np.ndarray,
    global_out_path: str,
    local_out_path: str,
) -> None:
    """
    Generates Figure 8 (Global SHAP Feature Importance & Beeswarm-style distribution)
    and Figure 9 (Local SHAP Waterfall / Contribution Plots for a True Positive Fraud vs Legitimate case).
    """
    feature_names = list(X_explain_df.columns)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:15]
    top_features = [feature_names[i] for i in top_idx]

    # Figure 8: Global SHAP Summary (2-panel: Mean |SHAP| Bar + Feature Value Impact Scatter)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))

    # Left: Mean |SHAP| Bar Chart
    ax = axes[0]
    rev_idx = top_idx[::-1]
    rev_names = [feature_names[i] for i in rev_idx]
    rev_vals = mean_abs_shap[rev_idx]
    bars = ax.barh(rev_names, rev_vals, color="#2563EB", edgecolor="black", height=0.65)
    ax.set_title("(a) Global Feature Importance — Mean |SHAP Value|", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean |SHAP Value| (Impact on Fraud Logit / Probability)")
    ax.set_xlim(0, max(rev_vals) * 1.2)
    ax.grid(axis="x", linestyle="--")
    for bar, v in zip(bars, rev_vals):
        ax.text(v + max(rev_vals) * 0.015, bar.get_y() + bar.get_height() / 2, f"{v:.4f}", va="center", fontsize=8.5)

    # Right: Beeswarm-style Impact Distribution colored by Normalized Feature Value
    ax = axes[1]
    rng = np.random.default_rng(42)
    for pos, feat_i in enumerate(rev_idx):
        sv = shap_values[:, feat_i]
        fv = X_explain_df.iloc[:, feat_i].values
        # Clip feature values to [5th, 95th] percentile for clean colormap
        p5, p95 = np.percentile(fv, 5), np.percentile(fv, 95)
        fv_norm = np.clip((fv - p5) / max(p95 - p5, 1e-8), 0.0, 1.0)
        jitter = rng.normal(0.0, 0.14, size=len(sv))
        sc = ax.scatter(
            sv,
            np.full_like(sv, pos) + jitter,
            c=fv_norm,
            cmap="coolwarm",
            s=16,
            alpha=0.75,
            vmin=0.0,
            vmax=1.0,
        )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_yticks(range(len(rev_names)))
    ax.set_yticklabels(rev_names)
    ax.set_title("(b) SHAP Value Distribution per Feature (Beeswarm View)", fontsize=12, fontweight="bold")
    ax.set_xlabel("SHAP Value (Right = Increases Fraud Risk, Left = Decreases Fraud Risk)")
    ax.grid(True, linestyle="--")
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Normalized Feature Value (Blue = Low, Red = High)", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(global_out_path), exist_ok=True)
    plt.savefig(global_out_path)
    plt.close(fig)

    # Figure 9: Local SHAP Explanations (True Positive Fraud vs. True Negative Legitimate)
    fraud_indices = np.where((y_explain == 1) & (probs_explain > 0.85))[0]
    legit_indices = np.where((y_explain == 0) & (probs_explain < 0.05))[0]
    fraud_idx = int(fraud_indices[0]) if len(fraud_indices) > 0 else int(np.argmax(probs_explain))
    legit_idx = int(legit_indices[0]) if len(legit_indices) > 0 else int(np.argmin(probs_explain))

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    cases = [
        (axes[0], fraud_idx, "True Positive Fraud Transaction", "#DC2626"),
        (axes[1], legit_idx, "True Negative Legitimate Transaction", "#2563EB"),
    ]

    for ax, idx, title_prefix, _ in cases:
        sv_case = shap_values[idx]
        fv_case = X_explain_df.iloc[idx].values
        top_local = np.argsort(np.abs(sv_case))[::-1][:10][::-1]
        labels = [f"{feature_names[i]} = {fv_case[i]:.2f}" for i in top_local]
        vals = sv_case[top_local]
        bar_colors = ["#DC2626" if v > 0 else "#2563EB" for v in vals]
        bars = ax.barh(labels, vals, color=bar_colors, edgecolor="black", height=0.6)
        ax.axvline(0.0, color="black", linewidth=1.0)
        ax.set_title(
            f"{title_prefix}\nPredicted Fraud Prob: {probs_explain[idx]:.4f} (Ground Truth: {y_explain[idx]})",
            fontsize=11.5,
            fontweight="bold",
        )
        ax.set_xlabel("Local SHAP Contribution (Red = +Fraud, Blue = -Fraud)")
        ax.grid(axis="x", linestyle="--")
        span = max(np.max(np.abs(vals)), 0.05)
        ax.set_xlim(-span * 1.28, span * 1.28)
        for bar, v in zip(bars, vals):
            offset = span * 0.03 if v >= 0 else -span * 0.03
            ha = "left" if v >= 0 else "right"
            ax.text(
                v + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{v:+.4f}",
                va="center",
                ha=ha,
                fontweight="bold",
                fontsize=8.5,
            )

    plt.tight_layout()
    os.makedirs(os.path.dirname(local_out_path), exist_ok=True)
    plt.savefig(local_out_path)
    plt.close(fig)
