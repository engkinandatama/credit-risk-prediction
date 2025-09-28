# 🏦 Credit Risk Prediction - Machine Learning Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Latest-yellow.svg)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-Latest-purple.svg)](https://catboost.ai/)

## 📋 Project Overview

This comprehensive machine learning project predicts credit default risk using historical lending data from 2007-2014. The project implements a complete end-to-end ML pipeline with advanced feature engineering, multiple model comparison, hyperparameter tuning, and robust model interpretability analysis.

### 🎯 Key Objectives
- Build accurate credit risk prediction models to minimize financial losses
- Compare performance across multiple ML algorithms
- Implement comprehensive model interpretability using SHAP
- Conduct thorough robustness and stability analysis
- Provide financial impact assessment for business decision-making

## 🏗️ Project Architecture

The project follows a structured approach divided into logical sections:

```
📁 Project Structure
├── 00_data_raw/           # Original dataset storage
├── 01_data_processed/     # Cleaned and processed data
├── 02_models/            # Trained models and preprocessors
├── 03_results/           # Evaluation results and visualizations
│   ├── analysis_data/    # Numerical analysis results
│   └── visualizations/   # Charts and plots
│       ├── univariate/   # Individual feature distributions
│       ├── bivariate/    # Feature-target relationships
│       ├── korelasi/     # Correlation analysis
│       └── shap_*/       # SHAP interpretability plots
└── 04_notebooks/         # Jupyter notebooks
```

## 🛠️ Technical Implementation

### Data Processing Pipeline
- **Data Cleaning**: Missing value treatment, outlier handling
- **Feature Engineering**: 
  - Credit history length calculation
  - Financial ratios (loan-to-income, DTI ratios)
  - Interaction features between key variables
  - Ordinal encoding for risk grades
- **Preprocessing**: RobustScaler for numerical features, OneHotEncoder for categorical

### Machine Learning Models
The project implements and compares 5 different algorithms:

1. **Logistic Regression** - Linear baseline model
2. **Random Forest** - Ensemble method for feature importance
3. **XGBoost** - Gradient boosting with advanced regularization
4. **LightGBM** - Fast gradient boosting framework
5. **CatBoost** - Categorical feature-optimized boosting (Champion Model)

### Model Evaluation Framework
- **Primary Metrics**: 
  - Gini Coefficient (model discrimination power)
  - KS Statistic (separation capability)
  - AUC-ROC (overall performance)
- **Business Metrics**: F1-Score, Precision, Recall
- **Financial Impact**: Profit/loss analysis with real monetary values

### Advanced Analysis Components
- **SHAP Analysis**: Global and local feature importance
- **Cross-validation**: 10-fold stratified validation for stability
- **Robustness Testing**: 
  - Noise sensitivity analysis
  - Population Stability Index (PSI)
  - Characteristic Stability Index (CSI)
- **Threshold Optimization**: Multiple methods for optimal cutoff selection

## 📊 Key Results

### Model Performance Summary
| Model | Gini | KS Score | AUC | F1-Score |
|-------|------|----------|-----|----------|
| CatBoost | 0.6847 | 0.4521 | 0.8424 | 0.3892 |
| XGBoost | 0.6791 | 0.4483 | 0.8395 | 0.3856 |
| LightGBM | 0.6756 | 0.4467 | 0.8378 | 0.3834 |
| Logistic Regression | 0.6234 | 0.4012 | 0.8117 | 0.3567 |
| Random Forest | 0.6189 | 0.3989 | 0.8094 | 0.3534 |

### Financial Impact Analysis
The CatBoost champion model provides:
- **Risk Reduction**: 45.2% separation between good and bad loans (KS Score)
- **Financial Value**: Quantified profit/loss analysis based on actual loan amounts
- **Business ROI**: Measurable improvement over "approve all" baseline strategy

## 🚀 Getting Started

### Prerequisites
```bash
# Core ML libraries
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# Advanced ML models
xgboost>=1.5.0
lightgbm>=3.3.0
catboost>=1.0.0

# Model interpretation
shap>=0.40.0

# Imbalanced data handling
imbalanced-learn>=0.8.0

# Utilities
joblib>=1.1.0
```

### Installation
```bash
# Clone the repository
git clone https://github.com/engkinandatama/credit-risk-prediction.git
cd credit-risk-prediction

# Install dependencies
pip install -r requirements.txt

# For Google Colab users
!pip install catboost xgboost lightgbm shap imbalanced-learn --quiet
```

### Usage

#### Option 1: Google Colab (Recommended)
1. Upload the notebook to Google Colab
2. Mount Google Drive for data storage
3. Run all cells sequentially - the notebook is designed for complete automation

#### Option 2: Local Jupyter Environment
```bash
# Start Jupyter notebook
jupyter notebook

# Open the main notebook file
# Follow the cell execution order as numbered in sections
```

### Data Requirements
- **Primary Dataset**: `loan_data_2007_2014.csv`
- **Size**: ~800MB, 887,000+ loan records
- **Features**: 74 original features covering loan details, borrower information, and credit history
- **Target**: Binary classification (Good Loan vs Bad Loan)

## 📈 Project Highlights

### Comprehensive EDA
- **Univariate Analysis**: Distribution analysis for all 74+ features
- **Bivariate Analysis**: Feature-target relationships with risk quantification  
- **Correlation Analysis**: Multi-collinearity detection and feature redundancy identification
- **Missing Value Strategy**: Intelligent imputation based on feature characteristics

### Advanced Feature Engineering
- **Domain Knowledge Integration**: Credit history length, financial ratios
- **Interaction Features**: Grade × Income, Term × Interest Rate combinations
- **Automated Feature Selection**: Correlation-based and importance-based filtering
- **Pipeline Integration**: All transformations embedded in sklearn pipelines

### Model Interpretability
- **Global Interpretability**: SHAP summary plots for overall feature importance
- **Local Interpretability**: Individual prediction explanations
- **Feature Impact Analysis**: Quantified contribution of each variable to risk assessment
- **Business-friendly Visualizations**: Easy-to-understand plots for stakeholders

### Robustness Validation
- **Cross-validation Stability**: Performance consistency across different data splits
- **Noise Sensitivity**: Model behavior under data perturbation (1%-10% noise levels)
- **Population Stability**: PSI analysis for data drift detection
- **Threshold Sensitivity**: Optimal cutoff point determination with multiple methods

## 📁 File Organization

### Main Components
- `credit_risk_prediction.ipynb` - Main notebook with complete pipeline
- `requirements.txt` - All required Python packages
- `README.md` - This comprehensive documentation

### Generated Outputs
- **Models**: Trained model files (.joblib format)
- **Preprocessors**: Data transformation pipelines
- **Results**: CSV files with model comparison metrics
- **Visualizations**: PNG files for all charts and plots
- **Analysis**: Numerical results from robustness tests

## 🔧 Customization Options

### Model Configuration
- Change `MODEL_TO_ANALYZE` variable to switch between models for interpretation
- Modify hyperparameter grids in Section 4.4 for different tuning strategies
- Adjust cross-validation folds and scoring metrics as needed

### Financial Analysis
- Update profit/loss assumptions in Section 5.3 based on business requirements  
- Modify risk thresholds for different business risk appetites
- Customize financial impact calculations for specific use cases

### Visualization Settings
- All plots automatically save to organized folder structures
- Customizable DPI and format settings for publication-quality outputs
- Color schemes and themes easily adjustable

## 🎯 Business Impact

### Risk Management
- **Quantified Risk Assessment**: Numerical scores for each loan application
- **Threshold Optimization**: Data-driven cutoff points for accept/reject decisions
- **Portfolio Analysis**: Segment-wise risk evaluation capabilities

### Financial Performance
- **Loss Prevention**: Identify high-risk loans before approval
- **Profit Optimization**: Balance between risk mitigation and business growth
- **ROI Measurement**: Clear financial metrics for model value assessment

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:
- Additional ML algorithms (Neural Networks, SVM)
- Advanced feature engineering techniques
- Real-time scoring pipeline implementation
- Model monitoring and drift detection systems
- Integration with business intelligence tools

## 📄 License

This project is available under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset sourced from Lending Club historical data
- Built with scikit-learn ecosystem and modern ML libraries
- SHAP library for state-of-the-art model interpretability
- Google Colab for accessible cloud-based development environment

## 📧 Contact

For questions, suggestions, or collaboration opportunities:
- GitHub Issues: Use the issue tracker for technical questions
- LinkedIn: [Engki Nandatama](https://www.linkedin.com/in/engkinandatama/)

---

*This project demonstrates end-to-end machine learning capabilities from data preprocessing through model deployment, with particular emphasis on interpretability and business impact assessment for financial risk management applications.*
