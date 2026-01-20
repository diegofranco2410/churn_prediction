import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set up and data loading
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
db_path = os.path.join(project_root, 'data', 'telecom_churn.db') 

conn = sqlite3.connect(db_path)

query = """
SELECT
    c.gender, c.SeniorCitizen, c.Partner, c.Dependents,
    s.PhoneService, s.InternetService, s.OnlineSecurity, s.TechSupport,
    a.Contract, a.PaperlessBilling, a.PaymentMethod, a.MonthlyCharges, a.TotalCharges, a.tenure,
    t.Churn
FROM customers c
JOIN services s ON (c.customerID = s.customerID)
JOIN accounts a ON (c.customerID = a.customerID)
JOIN churn_status t ON (c.customerID = t.customerID);
"""

print("Loading data for visualization")
df = pd.read_sql_query(query, conn)
conn.close()

sns.set_style("whitegrid") # scientific style plots

# Do we have a class imbalance problem?
plt.figure(figsize=(6,4))
ax = sns.countplot(x='Churn', data=df, palette='viridis')
plt.title('Distribution of Churn')
plt.ylabel('Number of Customers')

# Add percentages
total = len(df)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100*p.get_height()/total)
    x = p.get_x() + p.get_width()/2 - 0.05
    y = p.get_height()
    ax.annotate(percentage, (x,y), ha='center', va='bottom')


# Histograms

plt.figure(figsize=(10,6))
sns.histplot(data=df, x='MonthlyCharges', hue='Churn', kde=True, element="step", palette='coolwarm')
plt.title('Monthly Charges Distribution by Churn Status')
plt.xlabel('Monthly Charges ($)')

# Correlation matrix
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
corr_matrix = df[numerical_cols].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Numerical Features')

plt.show()