import numpy as np
from datetime import timedelta

import dataBase
df = dataBase.df

# Calculate the date which is compared in the recency attribute
latest_day = df['Customer.LastPurchaseDate'].max()
comparison_date = latest_day + timedelta(days=1)

# Group by customer id to get RFM
print('Grouping users by customer ID...')
df_rfm = df.groupby(['Customer.CustomerNo']).agg({
 'Customer.LastPurchaseDate': lambda x: (comparison_date - x.max()).days,
 'Invoice.InvoiceNo': 'nunique',
 'Invoice.AmountPaid': 'sum'
}).reset_index()

print('Columns in df_rfm:', list(df_rfm.columns))
print('df_rfm length:', len(df_rfm))
print(df_rfm.head())

print('Removing zero values from df_rfm...')
columns = ['Customer.LastPurchaseDate', 'Invoice.InvoiceNo', 'Invoice.AmountPaid']
df_rfm = df_rfm.replace(0, np.nan).dropna(axis=0, how='any', subset=columns).fillna(0)

print('Columns in df_rfm:', list(df_rfm.columns))
print('df_rfm length:', len(df_rfm))
