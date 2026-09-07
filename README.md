# Credit Risk Modeling: Loan Default Prediction

## 📌 Project Overview
This project is an end-to-end Machine Learning pipeline for predicting Credit Risk (Loan Default), developed as the Final Task for the Data Scientist Project-Based Internship at **ID/X Partners**.

The objective of this project is to build a classification model to evaluate the creditworthiness of borrowers based on historical lending data (2007-2014). By predicting the probability of default, the financial institution can make data-driven lending decisions and minimize financial risks.

## 🎯 Objectives
- Build a robust prediction model using historical loan data.
- Ensure the model generalizes well to new data without **Data Leakage** (avoiding post-origination features).
- Use highly interpretable models like **Logistic Regression** (Mandatory) alongside robust algorithms like **Random Forest Classifier**.
- Provide data-driven insights through Exploratory Data Analysis (EDA).

## 📊 Dataset & Data Dictionary
- **Dataset**: Lending Club Loan Data (2007-2014)
  - **Shape**: 466,285 rows × 74 initial columns.
  - **Download Link**: [loan_data_2007_2014.csv](https://rakamin-lms.s3.ap-southeast-1.amazonaws.com/vix-assets/idx-partners/loan_data_2007_2014.csv)
  - **Note**: The dataset file itself is ignored in version control (`.gitignore`) due to GitHub's 100MB file limit. Please download it from the link above and place it in the root directory before running the code.
- **Data Dictionary**: 
  - The local file `LCDataDictionary.xlsx` is included in this repository.
  - **Online Reference**: [Google Sheets - LC Data Dictionary](https://docs.google.com/spreadsheets/d/1iT1JNOBwU4l616_rnJpo0iny7blZvNBs/edit?gid=1666154857#gid=1666154857)

## 🛠️ Tech Stack & Tools
- **Language**: Python 3.11
- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn` (Logistic Regression, Random Forest, Metrics)
- **Environment**: JupyterLab / Jupyter Notebook

## ⚙️ Methodology & Pipeline

1. **Data Understanding**:
   - Creating the target variable (`bad_flag`). Based on `loan_status`, borrowers with status *Charged Off, Default, Late (31-120 days), Late (16-30 days),* or *In Grace Period* are flagged as `1` (Bad/Default), otherwise `0` (Good).
   - Removed **Data Leakage** columns (features representing post-origination status, e.g., `total_pymnt`, `recoveries`, `out_prncp`).

2. **Exploratory Data Analysis (EDA)**:
   - Evaluated the distribution of Good vs Bad loans (Imbalanced data: ~12% Bad).
   - Bivariate analysis to see the correlation between numeric/categorical features and default rates.
   - Identified correlations via Heatmap.

3. **Data Preparation**:
   - **Missing Value Handling**: Imputed medians for missing continuous variables and generated flag columns (`flag_mths_since_...`) for sparse features.
   - **Feature Engineering**: Converted `term` to integer, ordinal encoded `grade`, parsed dates (`earliest_cr_line`).
   - **Encoding**: One-hot encoding applied to `home_ownership`, `verification_status`, and `purpose`.
   - **Scaling**: Standardized features using `StandardScaler`.

4. **Data Modelling**:
   - Models trained with `class_weight='balanced'` to handle class imbalance.
   - **Logistic Regression**: Interpretable baseline model.
   - **Random Forest**: Non-linear ensemble model.

5. **Model Evaluation**:
   - Evaluated models using Accuracy, Precision, Recall, F1-Score, and **ROC-AUC** (primary metric for credit scoring).

## 🚀 Key Results & Insights
- Borrowers with lower loan grades (e.g., F, G) or high Interest Rates have a significantly higher probability of defaulting.
- Higher Debt-To-Income (DTI) ratio correlates with increased risk.
- **Random Forest** achieved superior predictive capability, but **Logistic Regression** provided great interpretability to explain why a user is rejected/accepted based on the coefficients.
- **Evaluation Metric**: Refer to the final evaluation summary outputted in the Jupyter Notebook for the exact ROC-AUC comparisons.

## 📁 Repository Structure
- `CreditRiskLoanAnalysis.ipynb` - The primary Jupyter Notebook containing the end-to-end data science pipeline.
- `CreditRiskLoanAnalysis.py` - The Python script version of the notebook for deployment or fast execution.
- `*.png` - Various generated visualizations from the EDA and Model Evaluation phase.
- `.gitignore` - Configurations to ignore large datasets and system files.

## ✍️ Author
- **Dhani** ([Dhani2612](https://github.com/Dhani2612))
