import pandas as pd
import matplotlib.pyplot as plt
import dataRFM

df_rfm = dataRFM.df_rfm

print(df_rfm.head())

# calculate the min and max values
max_values = df_rfm.max()
min_values = df_rfm.min()

print(f"min values are {min_values}")

#calculate quartiles for each column
quartiles = df_rfm.quantile(q=[0.25, 0.5, 0.75], axis=0)

# set the thresholds [<min, x, x, max] from quartiles
r_thresholds = quartiles['Customer.LastPurchaseDate'].tolist()
f_thresholds = quartiles['Invoice.InvoiceNo'].tolist()
m_thresholds = quartiles['Invoice.AmountPaid'].tolist()

print("Null values:")
print(df_rfm.isnull())

print("testi1")
print(f_thresholds)

# drop second values from the lists and keep 25 % and 75 %
r_thresholds.pop(1)
f_thresholds.pop(1)
m_thresholds.pop(1)

print("testi2")
print(f_thresholds)

# add the max and min values
r_thresholds.append(max_values[1])
r_thresholds.insert(0, min_values[1])

f_thresholds.append(max_values[2])
f_thresholds.insert(0, min_values[2])

m_thresholds.append(max_values[3])
m_thresholds.insert(0, min_values[3])

# define score for every attribute
df_rfm['r_score'] = pd.cut(df_rfm['Customer.LastPurchaseDate'], r_thresholds, labels = [3, 2, 1])
df_rfm['f_score'] = pd.cut(df_rfm['Invoice.InvoiceNo'], f_thresholds, labels = [1, 2, 3])
df_rfm['m_score'] = pd.cut(df_rfm['Invoice.AmountPaid'], m_thresholds, labels = [1, 2, 3])

# combine the scores
df_rfm['rfm_group'] = df_rfm['r_score'].astype(str).str.cat(df_rfm['f_score'].astype(str)).str.cat(df_rfm['m_score'].astype(str))

print(df_rfm.head())



# create human friendly RFM labels

segt_map = {
    r'3[1-3]3': 'Huiput',
    r'2[2-3]3' : 'Tyytymättömät',
    r'1[2-3][2-3]': 'Menetetty',
    r'[2-3][1-3][1-2]' : 'Petturit'
}

# Create a boolean mask based on segmentation mapping
mask = df_rfm['rfm_group'].str.contains('|'.join(segt_map.keys()), regex=True)

# Filter the DataFrame
filtered_df = df_rfm[mask]

# Display the resulting DataFrame
print(filtered_df)

# Apply segmentation mapping to create a new column 'segment'
filtered_df['segment'] = filtered_df['rfm_group'].replace(segt_map, regex=True)

# Get the count of customers for each segment
segment_counts = filtered_df['segment'].value_counts()

# Plotting the bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(segment_counts.index, segment_counts.values, color='skyblue', edgecolor='black')

# Display the count of customers above each bar
for bar, count in zip(bars, segment_counts):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(count),
             ha='center', va='bottom', color='black')

plt.title('Asiakkaiden kategorisointi RFM-pisteiden mukaan')
plt.xlabel('Asiakaskategoria')
plt.ylabel('Asiakkaiden lukumäärä')

plt.show()

