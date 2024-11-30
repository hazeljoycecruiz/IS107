# import pandas as pd
# import psycopg2

# # Load cleaned data
# cleaned_data = pd.read_csv('cleaned_data_v3.csv')

# # Connect to the PostgreSQL database
# try:
#     connection = psycopg2.connect(
#         host='localhost',
#         user='postgres',  # Default PostgreSQL username
#         password='postgres',  # Your password
#         database='data_warehouse'  # Your PostgreSQL database name
#     )
#     cursor = connection.cursor()
#     print("Database connection successful!")

#     # Insert unique products into Products
#     unique_products = cleaned_data[['Product Name', 'Category', 'Sub-Category']].drop_duplicates()

#     for index, row in unique_products.iterrows():
#         cursor.execute("""
#             INSERT INTO Products (Product_Name, Category, Sub_Category)
#             VALUES (%s, %s, %s)
#             ON CONFLICT (Product_Name) DO NOTHING;  -- Avoid duplicate inserts
#         """, (row['Product Name'], row['Category'], row['Sub-Category']))

#     connection.commit()
#     print("Data successfully inserted into Dim_Products!")

# except Exception as e:
#     print("An error occurred while connecting to the database or inserting data.")
#     print(e)

# finally:
#     if cursor:
#         cursor.close()
#     if connection:
#         connection.close()
#         print("Database connection closed.")

import pandas as pd
import psycopg2

# Load cleaned data
cleaned_data = pd.read_csv('cleaned_data_v3.csv')

# Connect to the PostgreSQL database
try:
    connection = psycopg2.connect(
        host='localhost',
        user='postgres',  # Default PostgreSQL username
        password='postgres',  # Your password
        database='data_warehouse'  # Your PostgreSQL database name
    )
    cursor = connection.cursor()
    print("Database connection successful!")

    # Insert data into Products
    for index, row in cleaned_data.iterrows():
        cursor.execute("""
            INSERT INTO Products (Product_ID, Product_Name, Category, Sub_Category)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (Product_ID) DO NOTHING;
        """, (row['Product ID'], row['Product Name'], row['Category'], row['Sub-Category']))

    # Insert data into Customers
    for index, row in cleaned_data.iterrows():
        cursor.execute("""
            INSERT INTO Customers (Customer_ID, Customer_Name, Segment, Country, City, State, Postal_Code, Region)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (Customer_ID) DO NOTHING;
        """, (row['Customer ID'], row['Customer Name'], row['Segment'], row['Country'], row['City'], row['State'], row['Postal Code'], row['Region']))

    # Insert data into Orders
    for index, row in cleaned_data.iterrows():
        cursor.execute("""
            INSERT INTO Orders (Order_ID, Order_Date, Ship_Date, Ship_Mode)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (Order_ID) DO NOTHING;
        """, (row['Order ID'], row['Order Date'], row['Ship Date'], row['Ship Mode']))

    # Insert data into Sales with order_id check
    print("Inserting data into Sales table...")
    for index, row in cleaned_data.iterrows():
        # Check if order_id already exists in Sales table
        cursor.execute("""
            SELECT 1 FROM Sales WHERE Order_ID = %s;
        """, (row['Order ID'],))
        
        # Only insert if order_id doesn't exist
        if cursor.fetchone() is None:
            cursor.execute("""
                INSERT INTO Sales (Order_ID, Customer_ID, Product_ID, Sales)
                VALUES (%s, %s, %s, %s);
            """, (row['Order ID'], row['Customer ID'], row['Product ID'], row['Sales']))

    connection.commit()
    print("Data successfully inserted into the database!")

    # Print summary of inserted records
    cursor.execute("SELECT COUNT(*) FROM Products;")
    products_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM Customers;")
    customers_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM Orders;")
    orders_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM Sales;")
    sales_count = cursor.fetchone()[0]
    
    print("\nInsertion Summary:")
    print(f"Products table: {products_count} records")
    print(f"Customers table: {customers_count} records")
    print(f"Orders table: {orders_count} records")
    print(f"Sales table: {sales_count} records")

    connection.commit()
    print("Data successfully inserted into the database!")

except Exception as e:
    print("An error occurred while connecting to the database or inserting data.")
    print(e)

    # Rollback the transaction if an error occurs
    if connection:
        connection.rollback()

finally:
    if cursor:
        cursor.close()
    if connection:
        connection.close()
        print("Database connection closed.")

