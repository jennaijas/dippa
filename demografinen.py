import pandas as pd
import numpy as np
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

#fetch the needed data
df = pd.read_csv('segmentointi_demografiset.csv',  sep=';', encoding='ANSI', low_memory=False)

#remove duplicates
df = df.drop(['Subscription.startdate',
            'Subscription.originalstartdate', 'Subscription.numberofcircular', 'Subscription.Active',
            'Customer.LastPurchaseDate', 'Customer.PurchaseSum', 'Subscription.startdate', 'Subscription.name',
              'Subscription.Club-ID'
        ], axis=1)
df = df.drop_duplicates()

#remove rows with empty values
df = df.dropna(subset=['Customer.CustomerNo', 'Customer.Sex', 'Customer.DateOfBirth'])

#remove Customer.Sex values with "0"
df = df.drop(df[df['Customer.Sex'] == 0].index)


#remove possible outliers
#customer is older than 100 years or younger than 18 years
df1 = df[df['Customer.DateOfBirth'] > 19230101]
df2 = df1[df1['Customer.DateOfBirth'] < 20050101]

print(df2.head(10))

#get the birth years and decades
birth_years = df2['Customer.DateOfBirth'].map(lambda x: math.floor(x/10000))
birth_years = birth_years.sort_values()

decades = df2['Customer.DateOfBirth'].map(lambda x: math.floor(x/100000)*10)
decades = decades.sort_values()

#get the customer sexes
customer_sexes = df2['Customer.Sex']

#visualize the data
#create bar chart with customers' sexes
axls = customer_sexes.value_counts().plot(kind='bar')
plt.title('Amount of women and men in customer group')
axls.bar_label(axls.containers[0])
plt.show()


#create bar chart with customers' birth years
axls = decades.value_counts()[decades.unique()].plot(kind='bar')
plt.title('Birth decades of customers')
axls.bar_label(axls.containers[0])
plt.show()

#visualize customer sexes and birth years together
plt.plot()
plt.xlim([1923, 2005])
plt.ylim([0, 2])
plt.title('Asiakkaiden syntymäajat ja sukupuolet')
plt.scatter(birth_years, customer_sexes)
plt.show()

#find optimal number of clusters using "elbow" method
inertia = []
clusters_range = range(1, 10)

# Calculate the inertia for each cluster number
X = np.array(list(zip(birth_years, customer_sexes)))
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

# Specify the number of clusters
n_clusters = 2

# Create a KMeans instance with the specified number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)

# Fit the KMeans model to your data
kmeans.fit(X)
cluster_labels = kmeans.labels_
df2.reset_index(drop=True, inplace=True)
df2['Cluster'] = cluster_labels

print(df2.head())

plt.figure(figsize=(10, 6))

# Define colors for each cluster
cluster_colors = ['r', 'g', 'b', 'y']

# Loop through unique cluster labels and plot data points for each cluster
for cluster_label in df2['Cluster'].unique():
    cluster_data = df2[df2['Cluster'] == cluster_label]
    plt.scatter(cluster_data['Customer.DateOfBirth'] / 10000, cluster_data['Customer.Sex'], label=f'Cluster {cluster_label}', c=cluster_colors[cluster_label])

# Customize the plot
plt.title('Scatterplot of Clusters')
plt.xlabel('Customer.DateOfBirth')
plt.ylabel('Customer.Sex')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Create heatmap
agg_df = pd.DataFrame({
    'Birth Year': decades,
    'Sexes': customer_sexes
})
agg_df = agg_df.groupby(['Birth Year', 'Sexes']).size().unstack(fill_value=0)

plt.figure(figsize=(8, 6))
sns.heatmap(agg_df, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Customer Counts by Birth Year and Sex")
plt.show()

#export csv
df2.to_csv(path_or_buf='results.csv', sep=';')