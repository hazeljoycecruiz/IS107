import pandas as pd
import numpy as np
import os

<<<<<<< HEAD
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
=======
# Step 1: Extract Data from CSV File
def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    - file_path: str, path to the CSV file.

    Returns:
    - DataFrame containing the loaded data or None if an error occurs.
    """
    try:
        data = pd.read_csv(file_path)
        print("Data successfully loaded.")
        return data
    except FileNotFoundError as e:
        print("Error: File not found. Check the file name and path.")
        print(e)
        return None
    except Exception as e:
        print("An error occurred while loading the file.")
        print(e)
        return None

# Step 2: Transform Data
def clean_data(data):
    """
    Clean and transform the data.

    Parameters:
    - data: DataFrame, the raw data to be cleaned.

    Returns:
    - DataFrame containing the cleaned data.
    """
    # a. Handle Missing Values
    data['Ship Date'] = data['Ship Date'].ffill()  # Forward fill missing ship dates
    data['Postal Code'] = data['Postal Code'].fillna(0)  # Fill missing postal codes with 0
>>>>>>> 88c8458477f1acfc9aafe547ee1f74161518613a

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

<<<<<<< HEAD
    # Save cleaned data
    output_csv_path = 'cleaned_data.csv'
    data_cleaned.to_csv(output_csv_path, index=False)
    print(f"Cleaned data saved to {output_csv_path}")
except Exception as e:
    print(f"Error during data transformation: {e}")
    exit()
=======
    # g. Handle Categorical Data Inconsistencies
    data['Category'] = data['Category'].str.strip()
    data['Sub-Category'] = data['Sub-Category'].str.strip()

    # h. Standardize Numeric Columns
    data['Sales'] = pd.to_numeric(data['Sales'], errors='coerce')
    data['Postal Code'] = data['Postal Code'].astype('int16', errors='ignore')

    return data

# Step 3: Load Cleaned Data into a New File
def save_cleaned_data(data, output_path):
    """
    Save the cleaned data to a CSV file.

    Parameters:
    - data: DataFrame, the cleaned data to be saved.
    - output_path: str, path to save the cleaned CSV file.
    """
    try:
        data.to_csv(output_path, index=False)
        print(f"Cleaned data saved to '{output_path}'.")
    except Exception as e:
        print("An error occurred while saving the cleaned data.")
        print(e)

# Step 4: Preview the Cleaned Data
def preview_data(data):
    """
    Preview the cleaned data.

    Parameters:
    - data: DataFrame, the cleaned data to be previewed.
    """
    print("\nPreview of the cleaned data:")
    print(data.head())  # Show the first 5 rows
    print("\nSummary of the dataset after cleaning:")
    print(data.info())  # Summary of the dataset

# Main Program
if __name__ == "__main__":
    # File paths
    input_file_path = r'C:\Users\User\Downloads\IS107-main\train.csv'
    output_file_path = r'C:\Users\User\Downloads\IS107-main\cleaned_data_v3.csv'

    # Load the data
    raw_data = load_data(input_file_path)

    # Proceed only if data is successfully loaded
    if raw_data is not None:
        # Clean the data
        cleaned_data = clean_data(raw_data)

        # Save the cleaned data
        save_cleaned_data(cleaned_data, output_file_path)

        # Preview the cleaned data
        preview_data(cleaned_data)
>>>>>>> 88c8458477f1acfc9aafe547ee1f74161518613a
