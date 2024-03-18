import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

import dataScaledRFM
scaled_rfm = dataScaledRFM.scaled_rfm
df_rfm = dataScaledRFM.df_rfm

# Fit the KMeans model to your data, change n_clusters to "4" if you want 4 clusters
model = KMeans(n_clusters=3, init="k-means++", max_iter=300, n_init=10, random_state=0)
print('Fitting scaled RFM to KMeans model...')
y_clusters = model.fit_predict(scaled_rfm)


# Check the number of clusters and number of customers in each cluster
counts = [0, 0, 0, 0]
for cluster_idx in y_clusters:
    counts[cluster_idx] = counts[cluster_idx] + 1

plt.bar(['0', '1', '2', '3'], counts, color ='maroon', width = 0.4)
plt.title("Kuhunkin klusteriin kuuluvien asiakkaiden lukumäärä")
plt.show()

# Create 3D visualization
print('Creating 3D visualization of clusters...')
fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111, projection='3d')
x = scaled_rfm.values
ax.scatter(x[y_clusters == 0,0],x[y_clusters == 0,1],x[y_clusters == 0,2], s = 40 , color = 'blue', label = "cluster 0")
ax.scatter(x[y_clusters == 1,0],x[y_clusters == 1,1],x[y_clusters == 1,2], s = 40 , color = 'orange', label = "cluster 1")
ax.scatter(x[y_clusters == 2,0],x[y_clusters == 2,1],x[y_clusters == 2,2], s = 40 , color = 'green', label = "cluster 2")
ax.scatter(x[y_clusters == 3,0],x[y_clusters == 3,1],x[y_clusters == 3,2], s = 40 , color = 'red', label = "cluster 3")
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
ax.legend()
plt.show()

# Create snake plot
df_rfm["Cluster"] = y_clusters
rfm = pd.melt(df_rfm.reset_index(),
                      id_vars=['Customer.CustomerNo', 'Cluster'],
                      value_vars=['Customer.LastPurchaseDate', 'Invoice.InvoiceNo', 'Invoice.AmountPaid'],
                      var_name='RFM-arvo',
                      value_name='Arvot')
print(rfm.head())
sns.lineplot(data=rfm, x='RFM-arvo', y='Arvot', style='Cluster')
plt.show()


# Add the cluster labels to your original DataFrame
df_rfm['Cluster'] = y_clusters

# Get the mean values for each cluster
print("Mean values of clusters:")
cluster_means = df_rfm.groupby('Cluster').mean().round(2)

print(cluster_means)
