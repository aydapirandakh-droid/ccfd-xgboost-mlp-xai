# Credit Card Fraud Detection with XGBoost, MLP, and Explainable AI

## Project Overview

This project explores credit card fraud detection using two machine learning approaches: **XGBoost** and a **Multi-Layer Perceptron (MLP)**.

Beyond predictive performance, the project investigates model explainability using SHAP and evaluates explanation quality through quantitative Fidelity and Stability experiments.

The main objective is to compare a tree-based model and a neural network under severe class imbalance and examine how consistently their explanations reflect model behavior.

## Dataset

The project uses the Credit Card Fraud Detection dataset.

- Total transactions: 284,807
- Normal transactions: 284,315
- Fraudulent transactions: 492
- Fraud rate: approximately 0.173%
- Input features: 30
- Target variable: `Class`

The dataset contains anonymized features (`V1`–`V28`), along with `Time` and `Amount`.

Exact duplicate transactions were removed, resulting in 283,726 records, including 473 fraudulent transactions.

## Methodology

### Data Preparation

- Removed exact duplicate transactions.
- Examined transaction timestamps and their distribution.
- Applied a chronological 70% / 15% / 15% train-validation-test split.
- Used RobustScaler for MLP input scaling, fitted exclusively on the training set.
- Kept the original feature representation for XGBoost.

### Models

**1. XGBoost Classifier**

A tree-based gradient boosting model trained for imbalanced binary classification.

**2. Multi-Layer Perceptron (MLP)**

A feed-forward neural network implemented with PyTorch, using fully connected layers, batch normalization, ReLU activation, and dropout.

The MLP was trained with class-weighted binary cross-entropy loss and early stopping based on validation performance.

## Model Performance

The following results were obtained on the chronological test set:

| Metric | XGBoost | MLP |
|---|---:|---:|
| PR-AUC | 0.7529 | 0.7582 |
| ROC-AUC | 0.9745 | 0.9759 |
| F1-score | 0.8000 | 0.7907 |
| True Positives | 38 | 34 |
| False Positives | 5 | 0 |
| False Negatives | 14 | 18 |

Decision thresholds were selected using validation data.

The two models achieved similar ranking performance. XGBoost obtained a slightly higher F1-score, while MLP achieved slightly higher PR-AUC and ROC-AUC.

## Explainable AI (XAI)

SHAP was used to investigate feature contributions:

- **XGBoost:** TreeSHAP
- **MLP:** GradientSHAP

The top five features identified by SHAP were:

| XGBoost | MLP |
|---|---|
| V4 | V14 |
| V14 | Time |
| V12 | V4 |
| V10 | V12 |
| V11 | V10 |

## Quantitative XAI Evaluation

### Fidelity

Explanation faithfulness was evaluated by removing the top-ranked features and measuring the average decrease in predicted fraud probability.

| Top Features Removed | XGBoost | MLP |
|---|---:|---:|
| Top-1 | 0.2511 | 0.1076 |
| Top-2 | 0.5387 | 0.2016 |
| Top-3 | 0.7326 | 0.4141 |
| Top-5 | 0.7491 | 0.7301 |

Higher probability drops indicate a stronger effect of removing the features identified as important.

### Stability

Explanation stability was evaluated by adding Gaussian noise (σ = 0.05) to fraud samples and comparing the resulting SHAP explanations.

| Metric | XGBoost | MLP |
|---|---:|---:|
| Pearson Correlation | 0.9983 | 0.9986 |
| Spearman Rank Correlation | 0.9815 | 0.9883 |

The results indicate high explanation consistency under the tested perturbation.

## Technologies

- Python
- XGBoost
- PyTorch
- SHAP
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Google Colab

## Repository Structure

```text
ccfd-xgboost-mlp-xai/
│
├── CCFD.XGboost.MLP.XAI.ipynb
└── README.md
```

## How to Run

1. Open the notebook in Google Colab.
2. Install the required Python libraries if they are not already available.
3. Run the notebook cells sequentially.
4. The dataset is expected to be available in the notebook environment.

## Notes

This project is an experimental implementation focused on model comparison and quantitative explainability evaluation.

The Fidelity and Stability results are based on the specific evaluation samples and perturbation settings used in the notebook. They should not be interpreted as universal guarantees of explanation quality.
