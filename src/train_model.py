import sqlite3
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# --- SETUP & DATA LOADING ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
db_path = os.path.join(project_root, 'data', 'telecom_churn.db')

conn = sqlite3.connect(db_path)
query = """
SELECT 
    c.gender, c.SeniorCitizen, c.Partner, c.Dependents,
    s.PhoneService, s.InternetService, s.OnlineSecurity, s.TechSupport, s.StreamingTV, s.StreamingMovies,
    a.Contract, a.PaperlessBilling, a.PaymentMethod, a.MonthlyCharges, a.TotalCharges, a.tenure,
    t.Churn
FROM customers c
JOIN services s ON c.customerID = s.customerID
JOIN accounts a ON c.customerID = a.customerID
JOIN churn_status t ON c.customerID = t.customerID;
"""
df = pd.read_sql_query(query, conn)
conn.close()

# --- PREPROCESSING ---
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'InternetService',
                    'OnlineSecurity', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                    'Contract', 'PaperlessBilling', 'PaymentMethod']

df_model = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df_model.drop('Churn', axis=1)
y = df_model['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- OVERSAMPLING (SMOTE) ---
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# --- MODEL TRAINING ---
model = XGBClassifier(
    n_estimators=350,
    learning_rate=0.1,
    max_depth=4,  
    random_state=42, 
    eval_metric='logloss'
)
model.fit(X_train_resampled, y_train_resampled)

# --- REALISTIC FINANCIAL OPTIMIZATION ---

# --- BUSINESS DATA ---
AVG_MONTHLY_BILL = df['MonthlyCharges'].mean()
NET_MARGIN = 0.20       # 20% Real profit margin
EXTENSION_PERIOD = 6    # Months gained if retained
DISCOUNT_RATE = 0.50    # Cost: 50% discount one month

# Equation: Benefit = Ticket * Months * Margin
REVENUE_PER_SAVED = AVG_MONTHLY_BILL * EXTENSION_PERIOD * NET_MARGIN

# Equation: Cost = Ticket * Discount
COST_PER_ACTION = AVG_MONTHLY_BILL * DISCOUNT_RATE

# Simulation
y_proba = model.predict_proba(X_test)[:, 1]
thresholds = np.arange(0.1, 1.0, 0.0001) # every 0.1%
profits = []
best_threshold = 0.5
max_net_profit = -float('inf')

for t in thresholds:
    y_pred_t = (y_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
    
    # Net Profit = (True Positives * Benefit) - (Total Contacted * Cost)
    current_profit = (tp * REVENUE_PER_SAVED) - ((tp + fp) * COST_PER_ACTION)
    profits.append(current_profit)
    
    if current_profit > max_net_profit:
        max_net_profit = current_profit
        best_threshold = t

# --- FINANCIAL RESULTS REPORT ---
print("-" * 50)
print(f"OPTIMAL THRESHOLD: {best_threshold:.4f}")
print(f"PROJECTED NET PROFIT (Test Set): ${max_net_profit:,.2f}")
print("-" * 50)

# --- VISUALIZATIONS ---

# A. Profit vs Threshold Curve
plt.figure(figsize=(10, 6))
plt.plot(thresholds, profits, color='green', linewidth=2, label='Net Profit Curve')
plt.axvline(best_threshold, color='red', linestyle='--', label=f'Optimal ({best_threshold:.2f})')
# Draw a baseline at 0 (Break-even point)
plt.axhline(0, color='gray', linestyle=':', alpha=0.7)
plt.title(f'Financial Optimization: Profit vs. Risk Threshold\n(Max Profit: ${max_net_profit:,.0f})')
plt.xlabel('Probability Threshold (Risk Tolerance)')
plt.ylabel('Net Profit ($)')
plt.legend()
plt.grid(True, alpha=0.3)

# APPLY OPTIMAL THRESHOLD FOR FINAL EVALUATION
y_pred_final = (y_proba >= best_threshold).astype(int)

# B. Confusion Matrix (Blue)
cm = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix (Threshold {best_threshold:.2f})')

# C. Feature Importance
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.values, y=feature_importances.index, hue=feature_importances.index, legend=False)
plt.title('Top 10 Drivers of Churn')
plt.xlabel('Relative Weight')

# --- FINAL METRICS & SAVE ---
acc = accuracy_score(y_test, y_pred_final)
print(f"\nAccuracy: {acc:.2%}")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred_final))

system_data = {
    "model": model,
    "features": X_train.columns.tolist(),
    "threshold": best_threshold
}
joblib.dump(system_data, os.path.join(project_root, 'models', 'churn_system.pkl'))
print("Done.")

plt.show() # Show all figures at the end