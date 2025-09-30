# 🏦 Credit Risk Prediction - Machine Learning Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-Champion-purple.svg)](https://catboost.ai/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Challenger-red.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Challenger-yellow.svg)](https://lightgbm.readthedocs.io/)
[![made-with-colab](https://colab.research.google.com/assets/colab-badge.svg )](https://colab.research.google.com/drive/18-iZh37kNQfgvg3mnamOU4SwZcsCoq0G?authuser=4#scrollTo=4fDFz94iDpAP)

## 📋 Project Overview

This machine learning project aims to build an accurate credit risk prediction model using historical lending data from 2007-2014. The project implements a complete end-to-end ML workflow, from data cleaning, advanced feature engineering, multi-model comparison, and hyperparameter tuning to in-depth model interpretability and robustness analysis.

### 🎯 Key Objectives
- To build an accurate credit risk prediction model to minimize potential financial losses.
- To compare the performance of various machine learning algorithms, including Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost.
- To implement comprehensive model interpretability using **SHAP (SHapley Additive exPlanations)**.
- To conduct a thorough model robustness and stability analysis.
- To provide a financial impact analysis to support business decision-making.

## 🏗️ Project Architecture

The project follows a structured approach divided into logical sections, with a clean folder structure to ensure reproducibility.

```
📁 Project Structure
├── 00_data_raw/           # Storage for the original dataset
├── 01_data_processed/     # Cleaned and model-ready data
├── 02_models/             # Trained models (.joblib) and preprocessors
├── 03_results/            # Evaluation results, metrics, and visualizations
│   ├── analysis_data/     # Numerical analysis results (model comparison, etc.)
│   └── visualizations/    # Charts and plots
│       ├── univariate/    # Individual feature distributions
│       ├── bivariate/     # Feature-target relationships
│       ├── korelasi/      # Correlation analysis
│       └── shap_catboost/ # SHAP interpretability plots for the CatBoost model
└── 04_notebooks/          # (Optional) Location for storing notebooks
```

## 🛠️ Technical Implementation

### Data Processing Pipeline
- **Data Cleaning**: Handling of missing values, identification and removal of extreme outliers using the 3xIQR method.
- **Feature Engineering**:
  - Calculation of domain-specific features like `credit_history_length_years`.
  - Creation of important financial ratios (`loan_to_income_ratio`, `installment_to_income_ratio`).
  - Generation of interaction features (`term_x_int_rate`, `grade_x_annual_inc`).
  - Ordinal Encoding for `grade` and `sub_grade` features.
- **Preprocessing**: Utilizes `RobustScaler` for numerical features (resistant to outliers) and `OneHotEncoder` for categorical features, combined within a `ColumnTransformer`.

### Machine Learning Models
The project implements and compares 5 different algorithms:

1.  **Logistic Regression** - A linear model as a baseline.
2.  **Random Forest** - A bagging-based ensemble model.
3.  **XGBoost** - A popular gradient boosting algorithm with regularization.
4.  **LightGBM** - A fast and efficient gradient boosting framework.
5.  **CatBoost** - A gradient boosting algorithm optimized for categorical features **(Champion Model)**.

### Model Evaluation Framework
- **Primary Metrics**:
  - **Gini Coefficient**: Measures the model's discriminatory power (2 * AUC - 1).
  - **KS Statistic (Kolmogorov-Smirnov)**: Measures the separation capability between positive and negative classes.
  - **AUC-ROC**: An overall performance metric.
- **Business Metrics**: F1-Score, Precision, and Recall to balance the identification of risky and good customers.
- **Financial Impact Analysis**: Simulation of profit/loss based on loan amounts and interest rates.

### Advanced Analysis Components
- **Hyperparameter Tuning**: `RandomizedSearchCV` with F1-score as the target metric to find the best parameters for each model.
- **SHAP Analysis**: Both global (to understand the most important features) and local (to explain individual predictions).
- **Cross-Validation**: Uses `StratifiedKFold` to ensure model stability.
- **Robustness Testing**:
  - Sensitivity analysis to data noise.
  - Performance analysis across various data segments (based on `grade`, `home_ownership`, etc.).
- **Threshold Optimization**: Finding the optimal cutoff point to maximize the F1-Score.

## 📊 Key Results

### Hyperparameter Tuning Summary
The tuning process was performed using `RandomizedSearchCV` with `n_iter=25` and `cv=3`, optimizing for the F1-score. CatBoost demonstrated the highest cross-validation F1-score, reinforcing its selection as the champion model.

| Model | Best CV F1-Score | Best Parameters |
| :--- | :--- | :--- |
| **CatBoost** | **0.5312** | `{'depth': 9, 'iterations': 752, 'l2_leaf_reg': 9.6, 'learning_rate': 0.02}` |
| XGBoost | 0.5298 | `{'colsample_bytree': 0.8, 'gamma': 0, 'learning_rate': 0.02, 'max_depth': 8, 'n_estimators': 415, 'subsample': 0.9}` |
| LightGBM | 0.5285 | `{'colsample_bytree': 0.7, 'learning_rate': 0.03, 'max_depth': 15, 'n_estimators': 658, 'num_leaves': 58, 'subsample': 0.9}` |
| Logistic Regression | 0.4921 | `{'C': 0.01, 'solver': 'liblinear'}` |

### Final Model Performance (on Test Set)
After tuning, the models were evaluated on the unseen test set. CatBoost maintained its superior performance, especially in terms of Gini and KS statistics.

| Model | Gini | KS | AUC | F1-Score (Bad Loan) | Recall (Bad Loan) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CatBoost** | **0.5968** | **0.4472** | **0.7984** | **0.52** | **0.70** |
| XGBoost | 0.5822 | 0.4344 | 0.7911 | 0.52 | 0.70 |
| LightGBM | 0.5779 | 0.4245 | 0.7890 | 0.52 | 0.68 |
| Logistic Regression | 0.5215 | 0.3852 | 0.7607 | 0.49 | 0.70 |

### Financial Impact Analysis
The implemented **CatBoost** model provides significant financial value compared to a "no-model" scenario (approving all loans).

- **Net Impact (With CatBoost Model)**: **$430,755.30**
- **Net Impact (Without Model)**: $-40,000,061.12
- **Financial Value Add of the Model**: **$40,430,816.42**

## 🚀 Getting Started

### Prerequisites
Ensure you have the following Python libraries installed:
```
pandas, numpy, scikit-learn, matplotlib, seaborn, catboost, xgboost, lightgbm, shap, imbalanced-learn, joblib
```

### Installation
```bash
# For local users, clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Install dependencies
pip install -r requirements.txt

# For Google Colab users (recommended)
# Simply run the installation cell within the notebook:
!pip install pandas numpy scikit-learn matplotlib seaborn catboost xgboost lightgbm shap imbalanced-learn --quiet
```

### Usage

#### Option 1: Google Colab (Recommended)
1.  Upload the notebook to Google Colab.
2.  Mount your Google Drive for data and results storage.
3.  Run all cells sequentially. The notebook is designed for full automation, from project setup to final analysis.

#### Option 2: Local Jupyter Environment
1.  Ensure all dependencies are installed.
2.  Start Jupyter Notebook from your terminal: `jupyter notebook`.
3.  Open the notebook file and follow the cell execution order as numbered in each section.

### Data Requirements
- **Primary Dataset**: `loan_data_2007_2014.csv` (required in the `00_data_raw` folder).
- **Size**: Approximately 466,285 rows and 75 columns.
- **Target**: `credit_risk` (binary classification: 0 = Good Loan, 1 = Bad Loan).

## 📈 Project Highlights

### Comprehensive EDA
- **Univariate & Bivariate Analysis**: Analysis of the distribution and relationship of each feature with the target variable.
- **Correlation Analysis**: Detection of multicollinearity to reduce feature redundancy.
- **Dynamic Summary**: Automatic creation of a preprocessing action plan based on EDA findings.

### Model Interpretability
- **Global Interpretability**: SHAP summary plots to view the most influential features overall.
- **Local Interpretability**: SHAP force plots to explain predictions for individual customers (good vs. bad loan cases).
- **Business-Friendly Visualizations**: Easy-to-understand plots for non-technical stakeholders.

### Robustness Validation
- **Cross-Validation Stability**: Consistent model performance across a 10-fold CV (Gini Std Dev < 0.03).
- **Noise Sensitivity**: Model behavior tested by adding noise (up to 10%) to the data.
- **Segment Performance Analysis**: Ensuring the model performs well across various customer segments (e.g., all loan `grades`).

```
## 📧 Contact

For questions, suggestions, or collaboration opportunities:
- **LinkedIn**: [Engki Nandatama](https://www.linkedin.com/in/engkinandatama/)
```
---

*This project demonstrates end-to-end machine learning capabilities, from data preprocessing to post-modeling analysis, with a special emphasis on interpretability, robustness, and business impact assessment for financial risk management applications.*
