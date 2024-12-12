import pandas as pd


# Step 1: Extract
file_path = 'Online Retail.xlsx'
data = pd.read_excel(file_path)

# Step 2: Transform
try:
    # Count rows in the uncleaned data
    uncleaned_row_count = data.shape[0]
    print(f"Number of rows in uncleaned data: {uncleaned_row_count}")

    # Clean missing values
    data = data.dropna(subset=['CustomerID', 'InvoiceDate'])
    data['CustomerID'] = data['CustomerID'].astype(int)
    
    # Fill missing descriptions based on existing descriptions for the same StockCode
    description_map = data.dropna(subset=['Description']).groupby('StockCode')['Description'].first()
    data['Description'] = data.apply(
        lambda row: description_map[row['StockCode']] if pd.isna(row['Description']) and row['StockCode'] in description_map else row['Description'],
        axis=1
    )

    # Identify unique StockCodes with missing descriptions
    unique_stockcodes = data[data['Description'].isna()]['StockCode'].unique()

    # Fill missing descriptions for unique StockCodes
    data.loc[data['StockCode'].isin(unique_stockcodes) & data['Description'].isna(), 'Description'] = 'Unknown Product'

    # Remove rows with zero and negative quantities
    data = data[data['Quantity'] > 0]

    # Remove invalid records with non-positive UnitPrice
    data = data[data['UnitPrice'] > 0]

    # Calculate the 1st and 99th percentiles for Quantity and UnitPrice
    quantity_thresholds = data['Quantity'].quantile([0.01, 0.99])
    unitprice_thresholds = data['UnitPrice'].quantile([0.01, 0.99])

    # Filter rows to keep only values within the defined percentile range
    data = data[
        data['Quantity'].between(quantity_thresholds.iloc[0], quantity_thresholds.iloc[1]) &
        data['UnitPrice'].between(unitprice_thresholds.iloc[0], unitprice_thresholds.iloc[1])
    ]

    # Create new columns
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    data['InvoiceYear'] = data['InvoiceDate'].dt.year
    data['InvoiceMonth'] = data['InvoiceDate'].dt.month
    data['InvoiceDay'] = data['InvoiceDate'].dt.day

    # Deduplicate key columns
    # data = data.drop_duplicates()

    # Count rows in the cleaned data
    cleaned_row_count = data.shape[0]
    print(f"Number of rows in cleaned data: {cleaned_row_count}")

    # Save cleaned data
    output_csv_path = 'cleaned_data.csv'
    data.to_csv(output_csv_path, index=False)
    print(f"Cleaned data saved to {output_csv_path}")
except Exception as e:
    print(f"Error during data transformation: {e}")
    exit()
