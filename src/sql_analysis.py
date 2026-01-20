import sqlite3
import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
db_path = os.path.join(project_root, 'data', 'telecom_churn.db') 

conn = sqlite3.connect(db_path)

print("Data base correctly loaded")

# Business questions
# Which payment has the highest average monthly cost?

query_payment = """
SELECT 
    PaymentMethod,
    COUNT(*) AS CustomerCount,
    ROUND(AVG(MonthlyCharges),2) AS AvgMonthlyBill
FROM accounts
GROUP BY PaymentMethod
ORDER BY AvgMonthlyBill DESC;
"""

df_payment = pd.read_sql_query(query_payment, conn)
print("Analysis by Payment Method")
print(df_payment)
print("\n")

# Risk by internet type
query_churn = """
SELECT 
    s.InternetService,
    SUM(c.Churn) as ChurnedCustomers,
    ROUND(CAST(SUM(c.Churn) as FLOAT)/COUNT(*)*100, 1) AS ChurnRate_Percent
FROM services s
JOIN churn_status c ON s.customerID = c.customerID
GROUP BY s.InternetService
ORder BY ChurnRate_Percent DESC;
"""

df_churn_rate = pd.read_sql_query(query_churn, conn)

print("Churn Risk by Internet Type")
print(df_churn_rate)
conn.close()