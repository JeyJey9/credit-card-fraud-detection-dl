# Credit Card Fraud Detection Using Multi-Layer Perceptrons

**Course:** Advanced Artificial Intelligence and Machine Learning Research (AAIMR)  
**Professor:** Prof. Dr. Costin Bădică  
**Project:** Research Project #1 (Section 2.1 - Multi-Layer Perceptron: Credit Card Fraud Detection)  
**Student:** George-Cătălin Rece  
**GitHub Repository:** https://github.com/JeyJey9/credit-card-fraud-detection-dl  
**Jupyter Notebook (GitHub):** https://github.com/JeyJey9/credit-card-fraud-detection-dl/blob/main/Credit_Card_Fraud_Detection_DL.ipynb  
**Jupyter Notebook (Google Colab):** https://colab.research.google.com/github/JeyJey9/credit-card-fraud-detection-dl/blob/main/Credit_Card_Fraud_Detection_DL.ipynb  

## Overview

This repository contains the code, experiments, notebook, and technical report for Project #1 (Credit Card Fraud Detection).

The project addresses binary classification on the strongly imbalanced ULB Credit Card Fraud dataset [33] ($284,807$ transactions, out of which only $492$ are frauds, or $0.1727\%$). Beyond the baseline Multi-Layer Perceptron (MLP) trained with Binary Cross-Entropy, the project implements:
1. **Imbalance-aware loss functions**: Cost-sensitive weighted Binary Cross-Entropy and Binary Focal Loss [31].
2. **Automated hyperparameter tuning (Extension 2)**: 20-trial Optuna TPE search [5] over network layer sizes, activation function (`ReLU`, `GELU`, `SiLU`), dropout rate, learning rate, weight decay, and Focal Loss parameters ($\alpha, \gamma$).
3. **Gradient Boosting comparison (Extension 3)**: Benchmarking the neural models against an `XGBoost` classifier trained on the same data splits.
4. **Explainable AI with SHAP (Extension 1)**: Using `shap.GradientExplainer` [32] to inspect which PCA features drive global and individual predictions.
5. **Web Dashboard**: A Streamlit app (`app.py`) for inspecting individual test transactions, adjusting the classification threshold, and viewing local SHAP plots.

## Project Structure

```text
├── data/
│   └── creditcard.csv                                  # ULB dataset (downloaded via src/download_data.py)
├── src/
│   ├── download_data.py                                # Downloads and checks the dataset
│   ├── data_utils.py                                   # Stratified 70/15/15 split and RobustScaler
│   ├── losses.py                                       # BinaryFocalLoss implementation
│   ├── models.py                                       # Configurable FraudMLP network
│   ├── train_eval.py                                   # Training loop, PR-AUC early stopping, metrics
│   └── plotting.py                                     # Scripts for generating plots
├── figures/                                            # Generated plots (fig1 - fig9)
├── results/                                            # CSV and JSON experiment logs
├── models/                                             # Saved model weights (.pt, .json, .joblib)
├── run_experiments.py                                  # Main script to run all experiments
├── app.py                                              # Streamlit dashboard
├── Credit_Card_Fraud_Detection_DL.ipynb                # Jupyter Notebook with code and outputs
├── Technical_Report_Credit_Card_Fraud_Detection.pdf    # Technical Report (PDF)
├── Technical_Report_Credit_Card_Fraud_Detection.docx   # Technical Report (Word)
├── Presentation_Credit_Card_Fraud_Detection.pptx       # Presentation slides (PPTX)
└── requirements.txt                                    # Python dependencies
```

## Results Summary (Test Set: 42,722 Transactions, 74 Frauds)

| Model | Val PR-AUC | Test PR-AUC | Test ROC-AUC | Test F1 ($\tau^*$) | Test F1 ($\tau=0.5$) | Precision ($\tau^*$) | Recall ($\tau^*$) | MCC ($\tau^*$) | TP | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline MLP (BCE)** | 0.8698 | 0.8408 | 0.9789 | 0.7771 | 0.8000 | 0.7349 | 0.8243 | 0.7779 | 61 | 22 | 13 |
| **Weighted MLP (Cost-Sensitive)** | 0.7867 | 0.7019 | 0.9818 | 0.7746 | 0.6563 | 0.8088 | 0.7432 | 0.7750 | 55 | 13 | 19 |
| **Focal Loss MLP ($\alpha=0.75, \gamma=2.0$)** | 0.8485 | 0.8268 | 0.9706 | 0.7742 | 0.7703 | 0.7407 | 0.8108 | 0.7746 | 60 | 21 | 14 |
| **Optuna Focal MLP** | **0.8798** | 0.8280 | 0.9649 | **0.8286** | 0.8451 | 0.8788 | 0.7838 | 0.8297 | 58 | 8 | 16 |
| **XGBoost (Baseline)** | 0.8775 | **0.8422** | 0.9726 | 0.8244 | **0.8489** | **0.9474** | 0.7297 | **0.8312** | 54 | 3 | 20 |

## How to Run

1. **Install dependencies:**
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. **Run the training and evaluation pipeline:**
```bash
python run_experiments.py
```

3. **Open the Jupyter Notebook:**
```bash
jupyter lab Credit_Card_Fraud_Detection_DL.ipynb
```

4. **Start the Streamlit dashboard:**
```bash
streamlit run app.py
```
