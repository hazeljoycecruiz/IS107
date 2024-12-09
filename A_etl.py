import pandas as pd
import numpy as np
import os

# Step 1: Extract
file_path = 'Online Retail.xlsx'  # Path to the dataset
if not os.path.exists(file_path):
    print(f"Error: The file {file_path} does not exist.")
    exit()

try:
    data = pd.read_excel(file_path)
except Exception as e:
    print(f"Error reading the Excel file: {e}")
    exit()

# Step 2: Transform
try:
    # Clean missing values
    data_cleaned = data.dropna(subset=['CustomerID', 'StockCode', 'InvoiceDate']).copy()
    data_cleaned['Description'] = data_cleaned['Description'].fillna('Unknown Product')

    # Remove invalid records
    data_cleaned = data_cleaned[(data_cleaned['Quantity'] > 0) & (data_cleaned['UnitPrice'] > 0)]

    # Create new columns
    data_cleaned['TotalPrice'] = data_cleaned['Quantity'] * data_cleaned['UnitPrice']
    data_cleaned['InvoiceDate'] = pd.to_datetime(data_cleaned['InvoiceDate'])
    data_cleaned['InvoiceYear'] = data_cleaned['InvoiceDate'].dt.year
    data_cleaned['InvoiceMonth'] = data_cleaned['InvoiceDate'].dt.month
    data_cleaned['InvoiceDay'] = data_cleaned['InvoiceDate'].dt.day

    # Handle outliers
    price_cap = data_cleaned['TotalPrice'].quantile(0.99)
    data_cleaned['TotalPrice'] = np.where(data_cleaned['TotalPrice'] > price_cap, price_cap, data_cleaned['TotalPrice'])

    # Deduplicate key columns
    data_cleaned = data_cleaned.drop_duplicates()

    # Save cleaned data
    output_csv_path = 'cleaned_data.csv'
    data_cleaned.to_csv(output_csv_path, index=False)
    print(f"Cleaned data saved to {output_csv_path}")
except Exception as e:
    print(f"Error during data transformation: {e}")
    exit()
