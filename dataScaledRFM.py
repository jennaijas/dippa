import pandas as pd
from sklearn.preprocessing import StandardScaler

import dataRFM
df_rfm = dataRFM.df_rfm

# Scaling and standardizing the data
scaler = StandardScaler()

print('Creating scaled RFM...')
reduced_df_rfm = df_rfm[['Customer.LastPurchaseDate', 'Invoice.InvoiceNo', 'Invoice.AmountPaid']]
scaled_rfm = pd.DataFrame(scaler.fit_transform(reduced_df_rfm))

print('Columns in scaled_rfm:', list(scaled_rfm.columns))
print('scaled_rfm length:', len(scaled_rfm))
print(scaled_rfm.head())
