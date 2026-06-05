
# Credit Scoring Model (Classification Pipeline)

## 📌 Project Overview
This repository contains a robust, end-to-end Machine Learning classification pipeline designed to predict an individual's creditworthiness using historical financial data. The model evaluates features such as age, annual income, existing debt metrics, and historical payment delinquencies to accurately classify borrowers into distinct risk categories, ultimately predicting the probability of default.

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.14
* **Frameworks & Frameworks:** Scikit-Learn, XGBoost, Pandas, NumPy

## 📊 Methodology & Pipeline Architecture
1. **Automated Data Management:** Features a built-in synthetic pipeline that replicates standard financial structures (e.g., matching structures like the Kaggle *Give Me Some Credit* dataset) to guarantee immediate execution.
2. **Feature Engineering:** Computes domain-specific indicators including:
   * **Debt-to-Income Ratio (DTI):** A critical indicator of financial leverage.
   * **Risk Factor Index:** Interaction term multiplying credit card utilization rates by historical late payment behavior.
3. **Imbalance Handling & Scaling:** Utilizes stratified train-test splits to preserve class distributions alongside uniform feature scaling using `StandardScaler`.
4. **Predictive Modeling:** Implements an optimized **XGBoost Classifier** configured with customized log-loss evaluation to handle non-linear decision boundaries effectively.

## 🚀 Execution Instructions
Ensure you are inside the root repository directory, then run:
```bash
python3 -m pip install -r requirements.txt --break-system-packages
python3 src/train.py
