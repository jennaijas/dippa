import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import dataScaledRFM
scaled_rfm = dataScaledRFM.scaled_rfm

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
