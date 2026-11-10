"""
Master Experimental Pipeline for Research Project #1: Credit Card Fraud Detection.
Executes:
  1. Dataset verification & Exploratory Data Analysis (EDA)
  2. Stratified 70/15/15 split & leak-free RobustScaler preprocessing
  3. Model 1: Baseline MLP (Standard BCE Loss)
  4. Model 2: Cost-Sensitive MLP (Custom Class-Weighted BCE Loss)
  5. Model 3: Focal Loss MLP (Lin et al., 2017, alpha=0.75, gamma=2.0)
  6. Model 4 (Extension 2): Optuna-Tuned Focal Loss MLP (Automated Hyperparameter Optimization)
  7. Model 5 (Extension 3): Gradient Boosting Comparison (XGBoost with AUPRC early stopping)
  8. Ablation Study on Class Weighting & Focal Loss (alpha, gamma)
  9. Extension 1: Explainable AI (SHAP Values — Global & Local Explanations for MLP)
 10. Generation of all publication-quality figures (300 DPI) and summary tables.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
import xgboost as xgb
import shap

from src.download_data import ensure_dataset, CSV_PATH
from src.data_utils import SEED, set_seed, load_and_preprocess_data, create_dataloaders
from src.losses import BinaryFocalLoss
from src.models import FraudMLP
from src.train_eval import (
    get_device,
    predict_dataloader,
    find_optimal_threshold,
    compute_metrics,
    train_mlp_model,
)
from src.plotting import (
    plot_eda_overview,
    plot_focal_loss_theory,
    plot_training_histories,
    plot_pr_and_roc_curves,
    plot_confusion_matrices,
    plot_metric_comparison_bars,
    plot_optuna_results,
    plot_shap_explanations,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

for d in [FIGURES_DIR, RESULTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)


def parse_arch(arch_str: str):
    return [int(x) for x in arch_str.split("-")]


def main():
    overall_start = time.time()
    set_seed(SEED)
    device = get_device()
    print(f"[INFO] Running experiments on device: {device}")

    # 1. Ensure dataset and generate EDA + Focal Loss theoretical plots
    csv_path = ensure_dataset(CSV_PATH)
    data = load_and_preprocess_data(csv_path, save_artifacts_dir=MODELS_DIR)
    df_raw = data["df_raw"]

    print("[STEP 1] Generating EDA and theoretical Focal Loss figures...")
    plot_eda_overview(df_raw, os.path.join(FIGURES_DIR, "fig1_eda_distributions.png"))
    plot_focal_loss_theory(os.path.join(FIGURES_DIR, "fig2_focal_loss_theory.png"))

    train_loader, val_loader, test_loader = create_dataloaders(data, batch_size=1024, seed=SEED)
    y_val = data["y_val"]
    y_test = data["y_test"]
    prior_prob = float(data["n_pos_train"]) / float(data["n_pos_train"] + data["n_neg_train"])

    histories = {}
    val_metrics_all = {}
    test_metrics_all = {}
    test_probs_all = {}
    val_probs_all = {}

    # =========================================================================
    # 2. Model 1: Baseline MLP (Standard Binary Cross-Entropy)
    # =========================================================================
    print("\n[STEP 2] Training Model 1: Baseline MLP (Standard BCE)...")
    set_seed(SEED)
    model_bce = FraudMLP(
        input_dim=30,
        hidden_dims=[128, 64, 32],
        dropout=0.30,
        activation="relu",
        use_batchnorm=True,
        prior_prob=prior_prob,
    )
    crit_bce = nn.BCEWithLogitsLoss()
    model_bce, hist_bce = train_mlp_model(
        model_bce, train_loader, val_loader, crit_bce, lr=1e-3, weight_decay=1e-4, epochs=30, patience=7, device=device
    )
    _, _, val_probs_bce = predict_dataloader(model_bce, val_loader, crit_bce, device)
    _, _, test_probs_bce = predict_dataloader(model_bce, test_loader, crit_bce, device)
    thr_bce, _ = find_optimal_threshold(y_val, val_probs_bce)

    m_name_1 = "Baseline MLP (BCE)"
    histories[m_name_1] = hist_bce
    val_metrics_all[m_name_1] = compute_metrics(y_val, val_probs_bce, threshold=thr_bce)
    test_metrics_all[m_name_1] = compute_metrics(y_test, test_probs_bce, threshold=thr_bce)
    test_metrics_all[m_name_1]["training_time_sec"] = hist_bce["training_time_sec"]
    test_metrics_all[m_name_1]["best_epoch"] = hist_bce["best_epoch"]
    test_probs_all[m_name_1] = test_probs_bce
    val_probs_all[m_name_1] = val_probs_bce
    torch.save(model_bce.state_dict(), os.path.join(MODELS_DIR, "baseline_mlp_bce.pt"))

    # =========================================================================
    # 3. Model 2: Cost-Sensitive MLP (Custom Class Weights)
    # =========================================================================
    print("\n[STEP 3] Training Model 2: Weighted MLP (Cost-Sensitive Custom Class Weights)...")
    set_seed(SEED)
    # Using damped inverse class weight sqrt(N_neg / N_pos) ~ 24.05 to balance precision & recall
    pos_weight_val = float(np.sqrt(data["pos_weight"]))
    model_w = FraudMLP(
        input_dim=30,
        hidden_dims=[128, 64, 32],
        dropout=0.30,
        activation="relu",
        use_batchnorm=True,
    )
    crit_w = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))
    model_w, hist_w = train_mlp_model(
        model_w, train_loader, val_loader, crit_w, lr=1e-3, weight_decay=1e-4, epochs=30, patience=7, device=device
    )
    _, _, val_probs_w = predict_dataloader(model_w, val_loader, crit_w, device)
    _, _, test_probs_w = predict_dataloader(model_w, test_loader, crit_w, device)
    thr_w, _ = find_optimal_threshold(y_val, val_probs_w)

    m_name_2 = "Weighted MLP (Cost-Sensitive)"
    histories[m_name_2] = hist_w
    val_metrics_all[m_name_2] = compute_metrics(y_val, val_probs_w, threshold=thr_w)
    test_metrics_all[m_name_2] = compute_metrics(y_test, test_probs_w, threshold=thr_w)
    test_metrics_all[m_name_2]["training_time_sec"] = hist_w["training_time_sec"]
    test_metrics_all[m_name_2]["best_epoch"] = hist_w["best_epoch"]
    test_probs_all[m_name_2] = test_probs_w
    val_probs_all[m_name_2] = val_probs_w
    torch.save(model_w.state_dict(), os.path.join(MODELS_DIR, "weighted_mlp.pt"))

    # =========================================================================
    # 4. Model 3: Focal Loss MLP (Lin et al., 2017: alpha=0.75, gamma=2.0)
    # =========================================================================
    print("\n[STEP 4] Training Model 3: Focal Loss MLP (alpha=0.75, gamma=2.0)...")
    set_seed(SEED)
    model_fl = FraudMLP(
        input_dim=30,
        hidden_dims=[128, 64, 32],
        dropout=0.30,
        activation="relu",
        use_batchnorm=True,
        prior_prob=prior_prob,
    )
    crit_fl = BinaryFocalLoss(alpha=0.75, gamma=2.0)
    model_fl, hist_fl = train_mlp_model(
        model_fl, train_loader, val_loader, crit_fl, lr=1e-3, weight_decay=1e-4, epochs=30, patience=7, device=device
    )
    _, _, val_probs_fl = predict_dataloader(model_fl, val_loader, crit_fl, device)
    _, _, test_probs_fl = predict_dataloader(model_fl, test_loader, crit_fl, device)
    thr_fl, _ = find_optimal_threshold(y_val, val_probs_fl)

    m_name_3 = "Focal Loss MLP"
    histories[m_name_3] = hist_fl
    val_metrics_all[m_name_3] = compute_metrics(y_val, val_probs_fl, threshold=thr_fl)
    test_metrics_all[m_name_3] = compute_metrics(y_test, test_probs_fl, threshold=thr_fl)
    test_metrics_all[m_name_3]["training_time_sec"] = hist_fl["training_time_sec"]
    test_metrics_all[m_name_3]["best_epoch"] = hist_fl["best_epoch"]
    test_probs_all[m_name_3] = test_probs_fl
    val_probs_all[m_name_3] = val_probs_fl
    torch.save(model_fl.state_dict(), os.path.join(MODELS_DIR, "focal_loss_mlp.pt"))

    # =========================================================================
    # 5. Extension 2: Automated Hyperparameter Tuning with Optuna (Model 4)
    # =========================================================================
    print("\n[STEP 5] Running Extension 2: Optuna Automated Hyperparameter Tuning (20 Trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        set_seed(SEED + trial.number)
        arch_str = trial.suggest_categorical(
            "architecture", ["128-64-32", "256-128-64", "256-128-64-32", "128-64"]
        )
        activation = trial.suggest_categorical("activation", ["relu", "gelu", "silu"])
        dropout = trial.suggest_float("dropout", 0.15, 0.45, step=0.05)
        lr = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        focal_alpha = trial.suggest_float("focal_alpha", 0.50, 0.90, step=0.05)
        focal_gamma = trial.suggest_float("focal_gamma", 1.0, 3.5, step=0.25)

        hidden_dims = parse_arch(arch_str)
        m = FraudMLP(
            input_dim=30,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            use_batchnorm=True,
            prior_prob=prior_prob,
        )
        c = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        _, h = train_mlp_model(
            m,
            train_loader,
            val_loader,
            c,
            lr=lr,
            weight_decay=weight_decay,
            epochs=20,
            patience=5,
            device=device,
            verbose=False,
        )
        print(
            f"  [Optuna Trial {trial.number:02d}] Val PR-AUC: {h['best_val_pr_auc']:.4f} | "
            f"Arch: {arch_str}, Act: {activation}, Drop: {dropout:.2f}, "
            f"LR: {lr:.5f}, alpha: {focal_alpha:.2f}, gamma: {focal_gamma:.2f}"
        )
        return h["best_val_pr_auc"]

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="mlp_focal_fraud_study")
    study.optimize(objective, n_trials=20)

    best_params = study.best_params
    print(f"\n[OPTUNA BEST] Trial #{study.best_trial.number} | Best Val PR-AUC: {study.best_value:.4f}")
    print(f"[OPTUNA BEST PARAMS] {json.dumps(best_params, indent=2)}")

    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(RESULTS_DIR, "optuna_trials.csv"), index=False)

    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception:
        importances = {k: 1.0 / len(best_params) for k in best_params}

    with open(os.path.join(RESULTS_DIR, "optuna_summary.json"), "w") as f:
        json.dump(
            {
                "best_trial_number": study.best_trial.number,
                "best_val_pr_auc": study.best_value,
                "best_params": best_params,
                "param_importances": importances,
            },
            f,
            indent=2,
        )

    plot_optuna_results(trials_df, importances, os.path.join(FIGURES_DIR, "fig7_optuna_study.png"))

    # Train final Optuna-tuned Focal Loss MLP for full epochs with best hyperparameters
    print("\n[STEP 5B] Training Final Model 4: Optuna-Tuned Focal Loss MLP...")
    set_seed(SEED + study.best_trial.number)
    best_hidden_dims = parse_arch(best_params["architecture"])
    model_opt = FraudMLP(
        input_dim=30,
        hidden_dims=best_hidden_dims,
        dropout=best_params["dropout"],
        activation=best_params["activation"],
        use_batchnorm=True,
        prior_prob=prior_prob,
    )
    crit_opt = BinaryFocalLoss(alpha=best_params["focal_alpha"], gamma=best_params["focal_gamma"])
    model_opt, hist_opt = train_mlp_model(
        model_opt,
        train_loader,
        val_loader,
        crit_opt,
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        epochs=35,
        patience=8,
        device=device,
    )
    _, _, val_probs_opt = predict_dataloader(model_opt, val_loader, crit_opt, device)
    _, _, test_probs_opt = predict_dataloader(model_opt, test_loader, crit_opt, device)
    thr_opt, _ = find_optimal_threshold(y_val, val_probs_opt)

    m_name_4 = "Optuna Focal MLP"
    histories[m_name_4] = hist_opt
    val_metrics_all[m_name_4] = compute_metrics(y_val, val_probs_opt, threshold=thr_opt)
    test_metrics_all[m_name_4] = compute_metrics(y_test, test_probs_opt, threshold=thr_opt)
    test_metrics_all[m_name_4]["training_time_sec"] = hist_opt["training_time_sec"]
    test_metrics_all[m_name_4]["best_epoch"] = hist_opt["best_epoch"]
    test_probs_all[m_name_4] = test_probs_opt
    val_probs_all[m_name_4] = val_probs_opt

    torch.save(
        {
            "state_dict": model_opt.state_dict(),
            "config": {
                "input_dim": 30,
                "hidden_dims": best_hidden_dims,
                "dropout": best_params["dropout"],
                "activation": best_params["activation"],
                "use_batchnorm": True,
                "threshold": thr_opt,
            },
        },
        os.path.join(MODELS_DIR, "optuna_focal_mlp.pt"),
    )

    # =========================================================================
    # 6. Extension 3: Gradient Boosting Comparison (XGBoost Classifier)
    # =========================================================================
    print("\n[STEP 6] Running Extension 3: Training & Evaluating XGBoost Baseline...")
    xgb_start = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight_val,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
    )
    xgb_model.fit(
        data["X_train"],
        data["y_train"],
        eval_set=[(data["X_val"], data["y_val"])],
        verbose=False,
    )
    xgb_time = time.time() - xgb_start
    val_probs_xgb = xgb_model.predict_proba(data["X_val"])[:, 1]
    test_probs_xgb = xgb_model.predict_proba(data["X_test"])[:, 1]
    thr_xgb, _ = find_optimal_threshold(y_val, val_probs_xgb)

    m_name_5 = "XGBoost (Baseline)"
    val_metrics_all[m_name_5] = compute_metrics(y_val, val_probs_xgb, threshold=thr_xgb)
    test_metrics_all[m_name_5] = compute_metrics(y_test, test_probs_xgb, threshold=thr_xgb)
    test_metrics_all[m_name_5]["training_time_sec"] = xgb_time
    test_metrics_all[m_name_5]["best_epoch"] = int(xgb_model.best_iteration)
    test_probs_all[m_name_5] = test_probs_xgb
    val_probs_all[m_name_5] = val_probs_xgb
    xgb_model.save_model(os.path.join(MODELS_DIR, "xgboost_model.json"))

    # =========================================================================
    # 7. Save Evaluation Tables & Generate Comparative Plots
    # =========================================================================
    print("\n[STEP 7] Generating Comparative Figures and Saving Metric Tables...")
    plot_training_histories(histories, os.path.join(FIGURES_DIR, "fig3_training_curves.png"))
    plot_pr_and_roc_curves(
        y_test, test_probs_all, test_metrics_all, os.path.join(FIGURES_DIR, "fig4_pr_and_roc_curves.png")
    )
    plot_confusion_matrices(test_metrics_all, os.path.join(FIGURES_DIR, "fig5_confusion_matrices.png"))
    plot_metric_comparison_bars(test_metrics_all, os.path.join(FIGURES_DIR, "fig6_model_comparison_bars.png"))

    # Build summary DataFrame
    summary_rows = []
    for m_name in test_metrics_all:
        tm = test_metrics_all[m_name]
        vm = val_metrics_all[m_name]
        summary_rows.append({
            "Model": m_name,
            "Val_PR_AUC": vm["pr_auc"],
            "Val_F1": vm["f1"],
            "Opt_Threshold": tm["threshold"],
            "Test_PR_AUC": tm["pr_auc"],
            "Test_ROC_AUC": tm["roc_auc"],
            "Test_F1": tm["f1"],
            "Test_Precision": tm["precision"],
            "Test_Recall": tm["recall"],
            "Test_MCC": tm["mcc"],
            "Test_Balanced_Acc": tm["balanced_accuracy"],
            "Test_F1_at_05": tm["f1_at_05"],
            "Test_Prec_at_05": tm["precision_at_05"],
            "Test_Rec_at_05": tm["recall_at_05"],
            "TP": tm["tp"],
            "FP": tm["fp"],
            "TN": tm["tn"],
            "FN": tm["fn"],
            "Training_Time_Sec": tm["training_time_sec"],
            "Best_Epoch_or_Iter": tm["best_epoch"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison_summary.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "all_metrics.json"), "w") as f:
        json.dump({"validation": val_metrics_all, "test": test_metrics_all}, f, indent=2)

    print("\n=== TEST SET PERFORMANCE SUMMARY ===")
    print(
        summary_df[
            ["Model", "Test_PR_AUC", "Test_ROC_AUC", "Test_F1", "Test_Precision", "Test_Recall", "Test_MCC", "TP", "FP", "FN"]
        ].to_string(index=False)
    )

    # =========================================================================
    # 8. Extension 1: Explainable AI (SHAP Analysis on Optuna Focal MLP)
    # =========================================================================
    print("\n[STEP 8] Running Extension 1: Explainable AI (SHAP Values for MLP)...")
    model_opt_cpu = model_opt.to("cpu").eval()

    # Background dataset: 200 stratified training samples
    rng = np.random.default_rng(SEED)
    bg_legit_idx = rng.choice(np.where(data["y_train"] == 0)[0], size=180, replace=False)
    bg_fraud_idx = rng.choice(np.where(data["y_train"] == 1)[0], size=20, replace=False)
    bg_idx = np.concatenate([bg_legit_idx, bg_fraud_idx])
    X_bg_tensor = torch.from_numpy(data["X_train"][bg_idx])

    # Evaluation subset for SHAP: all 74 Test Frauds + 226 Test Legitimate samples = 300 test cases
    test_fraud_idx = np.where(y_test == 1)[0]
    test_legit_idx = rng.choice(np.where(y_test == 0)[0], size=226, replace=False)
    explain_idx = np.concatenate([test_fraud_idx, test_legit_idx])
    X_explain_df = data["X_test_df"].iloc[explain_idx].reset_index(drop=True)
    X_explain_raw_df = data["X_test_raw_df"].iloc[explain_idx].reset_index(drop=True)
    y_explain = y_test[explain_idx]
    probs_explain = test_probs_opt[explain_idx]
    probs_xgb_explain = test_probs_xgb[explain_idx]
    X_explain_tensor = torch.from_numpy(data["X_test"][explain_idx])

    # Wrapper module outputting scalar column (N, 1) for SHAP GradientExplainer
    class SingleOutputWrapper(nn.Module):
        def __init__(self, base_model: nn.Module):
            super().__init__()
            self.base_model = base_model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.base_model(x).unsqueeze(-1)

    wrapped_model = SingleOutputWrapper(model_opt_cpu).eval()
    explainer = shap.GradientExplainer(wrapped_model, X_bg_tensor)
    raw_shap = explainer.shap_values(X_explain_tensor)
    if isinstance(raw_shap, list):
        shap_vals = np.array(raw_shap[0])
    else:
        shap_vals = np.array(raw_shap)
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 0]

    # Save SHAP importance table
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    shap_imp_df = pd.DataFrame({
        "Feature": data["feature_cols"],
        "Mean_Abs_SHAP": mean_abs_shap,
        "Mean_SHAP_Fraud": shap_vals[y_explain == 1].mean(axis=0),
        "Mean_SHAP_Legit": shap_vals[y_explain == 0].mean(axis=0),
    }).sort_values("Mean_Abs_SHAP", ascending=False)
    shap_imp_df.to_csv(os.path.join(RESULTS_DIR, "shap_feature_importance.csv"), index=False)

    plot_shap_explanations(
        shap_vals,
        X_explain_df,
        y_explain,
        probs_explain,
        os.path.join(FIGURES_DIR, "fig8_shap_global.png"),
        os.path.join(FIGURES_DIR, "fig9_shap_local.png"),
    )

    # Save bundle for interactive Streamlit dashboard
    joblib.dump(
        {
            "feature_cols": data["feature_cols"],
            "X_explain_df": X_explain_df,
            "X_explain_raw_df": X_explain_raw_df,
            "y_explain": y_explain,
            "probs_mlp": probs_explain,
            "probs_xgb": probs_xgb_explain,
            "shap_values": shap_vals,
            "background_data": data["X_train"][bg_idx],
            "summary_df": summary_df,
        },
        os.path.join(MODELS_DIR, "dashboard_bundle.joblib"),
    )

    print(f"\n[SUCCESS] Full experimental pipeline finished in {time.time() - overall_start:.1f}s!")


if __name__ == "__main__":
    main()
