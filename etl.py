import pandas as pd

# Step 1: Extract Data from CSV File
def load_data(file_path):
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
    # a. Handle Missing Values
    data['Ship Date'] = data['Ship Date'].ffill()  # Forward fill missing ship dates
    data['Postal Code'] = data['Postal Code'].fillna(0)  # Fill missing postal codes with 0

    # b. Convert Date Columns to DateTime
    data['Order Date'] = pd.to_datetime(data['Order Date'], errors='coerce')
    data['Ship Date'] = pd.to_datetime(data['Ship Date'], errors='coerce')

    # c. Remove Rows with Invalid Dates
    data = data.dropna(subset=['Order Date', 'Ship Date'])

    # d. Ensure Logical Consistency (Ship Date >= Order Date)
    data = data[data['Ship Date'] >= data['Order Date']]

    # e. Remove Duplicate Rows
    data.drop_duplicates(subset=['Order ID'], keep='first', inplace=True)

    # f. Handle Negative Sales (Outliers)
    data = data[data['Sales'] >= 0]

    # g. Handle Categorical Data Inconsistencies
    data['Category'] = data['Category'].str.strip()
    data['Sub-Category'] = data['Sub-Category'].str.strip()

    # h. Standardize Numeric Columns
    data['Sales'] = pd.to_numeric(data['Sales'], errors='coerce')
    data['Postal Code'] = data['Postal Code'].astype('int16', errors='ignore')

    return data

# Step 3: Load Cleaned Data into a New File
def save_cleaned_data(data, output_path):
    try:
        data.to_csv(output_path, index=False)
        print(f"Cleaned data saved to '{output_path}'.")
    except Exception as e:
        print("An error occurred while saving the cleaned data.")
        print(e)

# Step 4: Preview the Cleaned Data
def preview_data(data):
    print("\nPreview of the cleaned data:")
    print(data.head())  # Show the first 5 rows
    print("\nSummary of the dataset after cleaning:")
    print(data.info())  # Summary of the dataset

# Main Program
if __name__ == "__main__":
    # File paths
    input_file_path = r'C:\Users\User\Downloads\IS107 - WebApp\train.csv'
    output_file_path = r'C:\Users\User\Downloads\IS107 - WebApp\cleaned_data_v3.csv'

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
