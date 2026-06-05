
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

def generate_mock_credit_data(file_path):
    """Generates a realistic credit dataset if one doesn't exist locally."""
    print("Dataset not found. Generating a realistic credit dataset for training...")
    np.random.seed(42)
    n_samples = 2000
    
    data = {
        'Age': np.random.randint(21, 75, size=n_samples),
        'AnnualIncome': np.random.randint(20000, 150000, size=n_samples),
        'CurrentDebt': np.random.randint(1000, 50000, size=n_samples),
        'NumLatePayments': np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.7, 0.15, 0.08, 0.05, 0.02]),
        'CreditCardUtilization': np.random.uniform(0.0, 1.2, size=n_samples),
    }
    
    df = pd.DataFrame(data)
   
    debt_to_income = df['CurrentDebt'] / df['AnnualIncome']
    risk_score = (df['NumLatePayments'] * 2) + (df['CreditCardUtilization'] * 3) + (debt_to_income * 5)
    df['Default'] = (risk_score > np.percentile(risk_score, 85)).astype(int)
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"Mock dataset saved to {file_path}\n")

def load_and_preprocess(data_path):
    if not os.path.exists(data_path):
        generate_mock_credit_data(data_path)
        
    df = pd.read_csv(data_path)
    
    df['DebtToIncomeRatio'] = df['CurrentDebt'] / (df['AnnualIncome'] + 1)

    df['RiskFactorIndex'] = df['NumLatePayments'] * df['CreditCardUtilization']
    
    return df

def run_pipeline():
    data_path = 'data/credit_data.csv'
    df = load_and_preprocess(data_path)
    

    X = df.drop(columns=['Default'])
    y = df['Default']
    
   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
   
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
  
    print("Training XGBoost Credit Scoring Model...")
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train_scaled, y_train)
    
   
    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]
    
    print("\n================ EVALUATION METRICS ================")
    print(classification_report(y_test, predictions, target_names=['Good Credit', 'High Risk / Default']))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probabilities):.4f}")
    print("====================================================")

if __name__ == "__main__":
    run_pipeline()