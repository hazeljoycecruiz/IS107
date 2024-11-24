import pandas as pd
import psycopg2

# Load cleaned data
cleaned_data = pd.read_csv('cleaned_data_v3.csv')

# Connect to the PostgreSQL database
try:
    connection = psycopg2.connect(
        host='localhost',
        user='postgres',  # Default PostgreSQL username
        password='@l03e1t3',  # Your password
        database='data_warehouse'  # Your PostgreSQL database name
    )
    cursor = connection.cursor()
    print("Database connection successful!")

    # Insert unique products into Dim_Products
    unique_products = cleaned_data[['Product Name', 'Category', 'Sub-Category']].drop_duplicates()

    for index, row in unique_products.iterrows():
        cursor.execute("""
            INSERT INTO Dim_Products (Product_Name, Category, Sub_Category)
            VALUES (%s, %s, %s)
            ON CONFLICT (Product_Name) DO NOTHING;  -- Avoid duplicate inserts
        """, (row['Product Name'], row['Category'], row['Sub-Category']))

    connection.commit()
    print("Data successfully inserted into Dim_Products!")

except Exception as e:
    print("An error occurred while connecting to the database or inserting data.")
    print(e)

finally:
    if cursor:
        cursor.close()
    if connection:
        connection.close()
        print("Database connection closed.")

