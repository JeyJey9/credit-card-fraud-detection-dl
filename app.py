import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import matplotlib.pyplot as plt
import streamlit as st

from src.models import FraudMLP

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    bundle_path = os.path.join(MODELS_DIR, "dashboard_bundle.joblib")
    mlp_ckpt_path = os.path.join(MODELS_DIR, "optuna_focal_mlp.pt")
    xgb_path = os.path.join(MODELS_DIR, "xgboost_model.json")
    scaler_path = os.path.join(MODELS_DIR, "robust_scaler.joblib")
    metrics_path = os.path.join(RESULTS_DIR, "all_metrics.json")
    optuna_path = os.path.join(RESULTS_DIR, "optuna_summary.json")

    bundle = joblib.load(bundle_path)
    scaler = joblib.load(scaler_path)

    ckpt = torch.load(mlp_ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    mlp_model = FraudMLP(
        input_dim=cfg["input_dim"],
        hidden_dims=cfg["hidden_dims"],
        dropout=cfg["dropout"],
        activation=cfg["activation"],
        use_batchnorm=cfg["use_batchnorm"],
    )
    mlp_model.load_state_dict(ckpt["state_dict"])
    mlp_model.eval()

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(xgb_path)

    with open(metrics_path, "r") as f:
        all_metrics = json.load(f)
    with open(optuna_path, "r") as f:
        optuna_summary = json.load(f)

    trials_df = pd.read_csv(os.path.join(RESULTS_DIR, "optuna_trials.csv"))
    shap_imp_df = pd.read_csv(os.path.join(RESULTS_DIR, "shap_feature_importance.csv"))

    return {
        "bundle": bundle,
        "scaler": scaler,
        "mlp_model": mlp_model,
        "mlp_config": cfg,
        "xgb_model": xgb_model,
        "all_metrics": all_metrics,
        "optuna_summary": optuna_summary,
        "trials_df": trials_df,
        "shap_imp_df": shap_imp_df,
    }


def main():
    st.title("Credit Card Fraud Detection - Model Evaluation & SHAP Explorer")
    st.caption(
        "Research Project #1 | Course: AAIMR (Prof. Costin Badica) | Student: George-Catalin Rece"
    )

    if not os.path.exists(os.path.join(MODELS_DIR, "dashboard_bundle.joblib")):
        st.warning("Model files not found. Run `python run_experiments.py` first.")
        return

    art = load_artifacts()
    bundle = art["bundle"]
    summary_df = bundle["summary_df"]
    test_metrics = art["all_metrics"]["test"]

    best_mlp = test_metrics["Optuna Focal MLP"]
    xgb_m = test_metrics["XGBoost (Baseline)"]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Dataset Size", "284,807", "492 Frauds (0.1727%)")
    col2.metric("Optuna MLP PR-AUC", f"{best_mlp['pr_auc']:.4f}", f"F1: {best_mlp['f1']:.4f}")
    col3.metric("Optuna MLP ROC-AUC", f"{best_mlp['roc_auc']:.4f}", f"MCC: {best_mlp['mcc']:.4f}")
    col4.metric("XGBoost PR-AUC", f"{xgb_m['pr_auc']:.4f}", f"F1: {xgb_m['f1']:.4f}")
    col5.metric("Test Frauds Caught (MLP)", f"{best_mlp['tp']} / 74", f"{best_mlp['fp']} False Positives")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Model Comparison",
        "2. Transaction Inspector & SHAP",
        "3. Optuna Tuning",
        "4. Dataset & Training Plots",
    ])

    with tab1:
        st.subheader("Test Set Results (N = 42,722 | 74 Frauds)")
        display_cols = [
            "Model",
            "Test_PR_AUC",
            "Test_ROC_AUC",
            "Test_F1",
            "Test_Precision",
            "Test_Recall",
            "Test_MCC",
            "Opt_Threshold",
            "TP",
            "FP",
            "TN",
            "FN",
            "Training_Time_Sec",
        ]
        st.dataframe(
            summary_df[display_cols].style.format({
                "Test_PR_AUC": "{:.4f}",
                "Test_ROC_AUC": "{:.4f}",
                "Test_F1": "{:.4f}",
                "Test_Precision": "{:.4f}",
                "Test_Recall": "{:.4f}",
                "Test_MCC": "{:.4f}",
                "Opt_Threshold": "{:.3f}",
                "Training_Time_Sec": "{:.1f}s",
            }),
            use_container_width=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.image(
                os.path.join(FIGURES_DIR, "fig4_pr_and_roc_curves.png"),
                caption="Precision-Recall and ROC curves on the test split.",
                use_container_width=True,
            )
        with c2:
            st.image(
                os.path.join(FIGURES_DIR, "fig6_model_comparison_bars.png"),
                caption="Comparison of test metrics across all 5 models.",
                use_container_width=True,
            )

        st.image(
            os.path.join(FIGURES_DIR, "fig5_confusion_matrices.png"),
            caption="Confusion matrices on the test set using validation-calibrated thresholds.",
            use_container_width=True,
        )

    with tab2:
        st.subheader("Single Transaction Prediction & Local SHAP Values")
        X_exp = bundle["X_explain_df"]
        X_raw = bundle["X_explain_raw_df"]
        y_exp = bundle["y_explain"]
        p_mlp = bundle["probs_mlp"]
        p_xgb = bundle["probs_xgb"]
        shap_vals = bundle["shap_values"]
        feat_cols = bundle["feature_cols"]

        filter_type = st.radio(
            "Filter test subset:",
            ["All Sampled (300)", "Fraud Only (74)", "Legitimate Only (226)"],
            horizontal=True,
        )
        if "Fraud Only" in filter_type:
            valid_indices = np.where(y_exp == 1)[0]
        elif "Legitimate Only" in filter_type:
            valid_indices = np.where(y_exp == 0)[0]
        else:
            valid_indices = np.arange(len(y_exp))

        options = [
            f"Sample #{i:03d} | True Class: {'FRAUD (1)' if y_exp[i]==1 else 'LEGIT (0)'} | "
            f"Amount: {X_raw.iloc[i]['Amount']:.2f} EUR | MLP Prob: {p_mlp[i]:.4f} | XGB Prob: {p_xgb[i]:.4f}"
            for i in valid_indices
        ]
        selected_str = st.selectbox("Select transaction:", options)
        sel_idx = int(selected_str.split("#")[1].split(" ")[0])

        threshold_slider = st.slider(
            "Decision Threshold:",
            min_value=0.05,
            max_value=0.95,
            value=float(art["mlp_config"]["threshold"]),
            step=0.01,
        )

        k1, k2, k3, k4 = st.columns(4)
        gt_label = "FRAUD (1)" if y_exp[sel_idx] == 1 else "LEGITIMATE (0)"
        mlp_pred = "FRAUD" if p_mlp[sel_idx] >= threshold_slider else "LEGITIMATE"
        xgb_pred = "FRAUD" if p_xgb[sel_idx] >= threshold_slider else "LEGITIMATE"

        k1.metric("True Label", gt_label)
        k2.metric("Optuna MLP Score", f"{p_mlp[sel_idx]:.4f}", f"Pred: {mlp_pred}")
        k3.metric("XGBoost Score", f"{p_xgb[sel_idx]:.4f}", f"Pred: {xgb_pred}")
        k4.metric("Transaction Amount", f"{X_raw.iloc[sel_idx]['Amount']:.2f} EUR", f"Time: {X_raw.iloc[sel_idx]['Time']/3600:.1f}h")

        left_col, right_col = st.columns([1.25, 1.0])
        with left_col:
            st.markdown("#### Local SHAP Values (Optuna Focal MLP)")
            sv = shap_vals[sel_idx]
            fv = X_exp.iloc[sel_idx].values
            top_k = np.argsort(np.abs(sv))[::-1][:12][::-1]

            fig, ax = plt.subplots(figsize=(8, 4.8))
            labels = [f"{feat_cols[i]} = {fv[i]:.2f}" for i in top_k]
            vals = sv[top_k]
            colors = ["#DC2626" if v > 0 else "#2563EB" for v in vals]
            ax.barh(labels, vals, color=colors, edgecolor="black", height=0.6)
            ax.axvline(0.0, color="black", linewidth=1.0)
            ax.set_xlabel("SHAP Value (+ pushes toward Fraud, - pushes toward Legitimate)")
            ax.set_title(f"Sample #{sel_idx:03d} - Top 12 Features")
            ax.grid(axis="x", linestyle="--", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with right_col:
            st.markdown("#### Global SHAP Importance")
            st.image(
                os.path.join(FIGURES_DIR, "fig8_shap_global.png"),
                caption="Global SHAP feature importance across the test sample.",
                use_container_width=True,
            )

    with tab3:
        st.subheader("Optuna Hyperparameter Search Results")
        opt_sum = art["optuna_summary"]
        st.write("Best parameters found (Trial #17):")
        st.json(opt_sum["best_params"])

        c1, c2 = st.columns(2)
        with c1:
            st.image(
                os.path.join(FIGURES_DIR, "fig7_optuna_study.png"),
                caption="Optuna trial history and parameter importance.",
                use_container_width=True,
            )
        with c2:
            st.image(
                os.path.join(FIGURES_DIR, "fig2_focal_loss_theory.png"),
                caption="Focal Loss curves for different gamma values.",
                use_container_width=True,
            )

        st.markdown("#### All Trials")
        st.dataframe(art["trials_df"], use_container_width=True)

    with tab4:
        st.subheader("Exploratory Data Analysis & Learning Curves")
        st.image(
            os.path.join(FIGURES_DIR, "fig1_eda_distributions.png"),
            caption="Dataset distributions (class balance, amount, time, top PCA features).",
            use_container_width=True,
        )
        st.image(
            os.path.join(FIGURES_DIR, "fig3_training_curves.png"),
            caption="Validation PR-AUC and normalized training loss per epoch.",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
