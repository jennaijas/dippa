import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#fetch the needed data
df = pd.read_csv('segmentointi_rfm.csv',  sep=';', encoding='ANSI', low_memory=False)

print(len(df))
#remove duplicates
df = df.drop_duplicates()

#remove rows with unnecessary values
df = df.dropna(subset=['Customer.CustomerNo', 'Customer.LastPurchaseDate'])

#remove unnecessary columns
df = df.drop(['Customer.PurchaseSum', 'Subscription.name', 'Subscription.startdate',
            'Subscription.originalstartdate', 'Subscription.Club-ID', 'Subscription.Club-ID', 'Subscription.Active',
             'Invoice.InvoiceDate', 'Invoice.InvoiceTotal'
        ], axis=1)

print(len(df))
print(df)
print(df.head())

#convert the string type date to date type
df['Customer.LastPurchaseDate'] = pd.to_datetime(df['Customer.LastPurchaseDate'], format='%Y%m%d')

#calculate the date which is compared in the recency attribute
latest_day = df['Customer.LastPurchaseDate'].max()
comparison_date = latest_day + timedelta(days=1)

#group by customer id to get RFM
df_rfm = df.groupby(['Customer.CustomerNo']).agg({
 'Customer.LastPurchaseDate': lambda x: (comparison_date - x.max()).days,
 'Invoice.InvoiceNo': 'nunique',
 'Invoice.AmountPaid': 'sum'
}).reset_index()

print(df_rfm.head())

#remove zero values
columns = ['Customer.LastPurchaseDate', 'Invoice.InvoiceNo', 'Invoice.AmountPaid']
df_rfm = df_rfm.replace(0, np.nan).dropna(axis=0, how='any', subset=columns).fillna(0)

#Scatter plots for outlier detection
#Monetary & Frequency
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(df_rfm['Invoice.AmountPaid'], df_rfm['Invoice.InvoiceNo'])

# x-axis label
ax.set_xlabel('(Monetary)')

# y-axis label
ax.set_ylabel('(Frequency)')
plt.show()

#Monetary & Recency
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(df_rfm['Invoice.AmountPaid'], df_rfm['Customer.LastPurchaseDate'])

# x-axis label
ax.set_xlabel('(Monetary)')

# y-axis label
ax.set_ylabel('(Recency)')
plt.show()

#Frequency & Recency
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(df_rfm['Invoice.InvoiceNo'], df_rfm['Customer.LastPurchaseDate'])

# x-axis label
ax.set_xlabel('(Frequency)')

# y-axis label
ax.set_ylabel('(Recency)')
plt.show()

print(df_rfm['Invoice.InvoiceNo'].head())
#Delete the outlier
df_rfm = df_rfm.drop(df_rfm[df_rfm['Invoice.InvoiceNo'] > 500].index)

#Monetary & Recency
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(df_rfm['Invoice.AmountPaid'], df_rfm['Invoice.InvoiceNo'])

# x-axis label
ax.set_xlabel('(Monetary)')

# y-axis label
ax.set_ylabel('(Frequency)')
plt.show()

# Scaling and standardizing the data
scaler = StandardScaler()
scaled_rfm = pd.DataFrame(scaler.fit_transform(df_rfm))

# Find optimal number of clusters using "elbow" method
inertia = []
clusters_range = range(1, 10)

# Calculate the inertia for each cluster number
X = np.array(scaled_rfm)
for n_clusters in clusters_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 4))
plt.plot(clusters_range, inertia, marker='o', linestyle='-', color='b')
plt.xlabel('Klustereiden määrä')
plt.ylabel('Klusterin sisällä olevien neliöiden summa')
plt.title('Sopivan klusterimäärän etsintä Elbow Methodilla')
plt.grid(True)
plt.show()

# Fit the KMeans model to your data
model = KMeans(n_clusters = 3, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
y_clusters = model.fit_predict(scaled_rfm)

# Check the number of clusters and number of customers in each cluster
sns.countplot(y_clusters)
plt.show()