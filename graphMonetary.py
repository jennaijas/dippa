import matplotlib.pyplot as plt

import dataRFM
df_rfm = dataRFM.df_rfm

# Scatter plots for outlier detection
# Monetary & Frequency
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(df_rfm['Invoice.AmountPaid'], df_rfm['Invoice.InvoiceNo'])

# x-axis label
ax.set_xlabel('(Monetary)')

# y-axis label
ax.set_ylabel('(Frequency)')
plt.show()

# Monetary & Recency
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(df_rfm['Invoice.AmountPaid'], df_rfm['Customer.LastPurchaseDate'])

# x-axis label
ax.set_xlabel('(Monetary)')

# y-axis label
ax.set_ylabel('(Recency)')
plt.show()

# Frequency & Recency
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(df_rfm['Invoice.InvoiceNo'], df_rfm['Customer.LastPurchaseDate'])

# x-axis label
ax.set_xlabel('(Frequency)')

# y-axis label
ax.set_ylabel('(Recency)')
plt.show()

print(df_rfm['Invoice.InvoiceNo'].head())
# Delete the outlier
df_rfm_pretty = df_rfm.drop(df_rfm[df_rfm['Invoice.InvoiceNo'] > 500].index)

# Monetary & Recency
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(df_rfm_pretty['Invoice.AmountPaid'], df_rfm_pretty['Invoice.InvoiceNo'])

# x-axis label
ax.set_xlabel('(Monetary)')

# y-axis label
ax.set_ylabel('(Frequency)')
plt.show()
