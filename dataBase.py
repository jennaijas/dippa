import pandas as pd

print('Reading source data...')
df = pd.read_csv('segmentointi_rfm.csv',  sep=';', encoding='ANSI', low_memory=False)
print('Source data rows:', len(df))
print('Source data columns:', list(df.columns))

# Take only the necessary columns and drop all others
df = df[['Customer.CustomerNo', 'Customer.LastPurchaseDate', 'Invoice.InvoiceNo', 'Invoice.AmountPaid']]

# Remove rows with missing values
df = df.dropna(subset=['Customer.CustomerNo', 'Customer.LastPurchaseDate'])
print('Data rows after removing lines with missing values:', len(df))

# Remove duplicates
df = df.drop_duplicates()
print('Data rows after duplicate removal:', len(df))

# Convert the string type date to date type
df['Customer.LastPurchaseDate'] = pd.to_datetime(df['Customer.LastPurchaseDate'], format='%Y%m%d')

print('Columns in df:', list(df.columns))

# Set .head() to always display all columns
pd.set_option('display.max_columns', None)
print(df.head())
