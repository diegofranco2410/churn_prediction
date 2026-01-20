import pandas as pd 
import sqlite3
import os

# Read path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

csv_path = os.path.join(project_root, 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
db_path = os.path.join(project_root, 'data', 'telecom_churn.db')

# Load original data
df = pd.read_csv(csv_path)
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# fast cleaning
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce').fillna(0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}) # converting to binary

# Connect to SQLite 

customers_cols = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents']
df_customers = df[customers_cols].copy()
df_customers.to_sql('customers', conn, if_exists='replace', index = False)

service_cols = ['customerID', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
df_services = df[service_cols].copy()
df_services.to_sql('services', conn, if_exists='replace', index=False)

account_cols = ['customerID', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
                'MonthlyCharges', 'TotalCharges', 'tenure']
df_accounts = df[account_cols].copy()
df_accounts.to_sql('accounts', conn, if_exists='replace', index=False)

# Target table
df_churn = df[['customerID', 'Churn']].copy()
df_churn.to_sql('churn_status', conn, if_exists='replace', index=False)

print("Database 'telecom_churn' has been succesfully created.")
conn.close()
