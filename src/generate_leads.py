import sqlite3
import pandas as pd
import joblib
import os
from datetime import datetime

# --- LOAD MODEL ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
system_path = os.path.join(project_root, 'models', 'churn_system.pkl')
db_path = os.path.join(project_root, 'data', 'telecom_churn.db') 

if not os.path.exists(system_path):
    print("Error: 'churn_system.pkl' not found. Run train_model.py first.")
    exit()

# load model with diccionary, features and optimal threshold 
system_data = joblib.load(system_path)
model = system_data['model']
features_trained = system_data['features'] 
threshold = system_data['threshold']

print(f"System Loaded. calibrated Threshold: {threshold:.4f}")

# --- GET ACTIVE CLIENTES ---
print("Fetching active customers from Database...")
conn = sqlite3.connect(db_path)

# Only those who hadn't leave, churn=no
query = """
SELECT * 
FROM customers c
JOIN services s ON c.customerID = s.customerID
JOIN accounts a ON c.customerID = a.customerID
JOIN churn_status t ON c.customerID = t.customerID
WHERE t.Churn = 0
"""
df_active = pd.read_sql_query(query, conn)
conn.close()

df_active = df_active.loc[:, ~df_active.columns.duplicated()] # cannot reindex on an axis with duplicate labels

print(f"   -> Analyzing {len(df_active)} active customers.")

# --- VECTORIAL ALINEATION ---
# just as in the training:

# A-One-Hot Encoding
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'InternetService',
                    'OnlineSecurity', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                    'Contract', 'PaperlessBilling', 'PaymentMethod']

# Esto crea las columnas dummies, PERO puede que falten algunas (ej: si hoy nadie tiene 'Fiber Optic')
df_input = pd.get_dummies(df_active, columns=categorical_cols, drop_first=True)

# B- Projection to the model's vectorial space
# df_input must have exactly the same columns as X_train
# If one's missing: put 0 0. If one extra, ignores.
df_input = df_input.reindex(columns=features_trained, fill_value=0)

# --- PREDICTION ---
print("Predicting Churn Probability...")

# Probability 
probs = model.predict_proba(df_input)[:, 1]

# clean data frame
leads_df = df_active[['customerID', 'MonthlyCharges', 'tenure', 'Contract', 'InternetService']].copy()
leads_df['Churn_Probability'] = probs

# --- APPLY OPTIMAL THRESHOLD ---
high_risk_leads = leads_df[leads_df['Churn_Probability'] >= threshold].copy()
high_risk_leads = high_risk_leads.sort_values(by='Churn_Probability', ascending=False) # Order by Risk

# --- RESULTS ---
print("-" * 50)
print(f"REPORT SUMMARY")
print(f"   Total Active Customers: {len(df_active)}")
print(f"   High Risk Leads detected: {len(high_risk_leads)} ({(len(high_risk_leads)/len(df_active)):.1%})")
print(f"   (Threshold used: {threshold:.4f})")
print("-" * 50)

if not high_risk_leads.empty:
    # File name with date
    filename = f"ACTION_PLAN_{datetime.now().strftime('%Y-%m-%d')}.csv"
    save_path = os.path.join(current_dir, filename)
    
    high_risk_leads.to_csv(save_path, index=False)
    print(f"🚀 ACTION PLAN GENERATED: {filename}")
    print("   Send this file to the Retention Team immediately.")
    
    print("\nTop 5 Customers to Call:")
    print(high_risk_leads.head().to_string(index=False))
else:
    print("Good news! No customers are currently in the high-risk zone.")