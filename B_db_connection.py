import pandas as pd
from sqlalchemy import create_engine

# Database connection setup
db_name = "data_warehouse"
db_user = "postgres"
db_password = "postgres"
db_host = "localhost"
db_port = 5432
engine = create_engine(f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}", echo=True)

# Load cleaned data
csv_path = "cleaned_data.csv"
cleaned_data = pd.read_csv(csv_path)

# Convert data types to match the database schema
cleaned_data['CustomerID'] = cleaned_data['CustomerID'].astype(int)
cleaned_data['Quantity'] = cleaned_data['Quantity'].astype(int)
cleaned_data['UnitPrice'] = cleaned_data['UnitPrice'].astype(float)
cleaned_data['InvoiceDate'] = pd.to_datetime(cleaned_data['InvoiceDate'])

# Load country_dimension with deduplication
try:
    existing_countries = pd.read_sql("SELECT country FROM country_dimension", engine)
    new_countries = pd.DataFrame(cleaned_data['Country'].unique(), columns=['country'])
    new_countries = new_countries[~new_countries['country'].isin(existing_countries['country'])]
    if not new_countries.empty:
        new_countries.to_sql('country_dimension', engine, if_exists='append', index=False)
    print("Countries successfully loaded into country_dimension.")
except Exception as e:
    print(f"Error during loading data into country_dimension: {e}")

# Load customer_dimension with conflict handling
try:
    country_map = pd.read_sql("SELECT country, country_id FROM country_dimension", engine)
    customer_dim = cleaned_data[['CustomerID', 'Country']].drop_duplicates().rename(columns={'CustomerID': 'customer_id', 'Country': 'country'})
    customer_dim = customer_dim.merge(country_map, how='left', on='country')
    customer_dim = customer_dim[['customer_id', 'country_id']]
    existing_customers = pd.read_sql("SELECT customer_id FROM customer_dimension", engine)
    new_customers = customer_dim[~customer_dim['customer_id'].isin(existing_customers['customer_id'])]
    if new_customers.empty:
        print("No new customers to load into customer_dimension.")
    else:
        new_customers.to_sql('customer_dimension', engine, if_exists='append', index=False)
        print("New customers successfully loaded into customer_dimension.")
except Exception as e:
    print(f"Error during loading data into customer_dimension: {e}")

# Load product_dimension
try:
    product_dim = cleaned_data[['StockCode', 'Description', 'UnitPrice']].drop_duplicates().rename(columns={
        'StockCode': 'stockcode',
        'Description': 'description',
        'UnitPrice': 'unit_price'
    })
    product_dim.to_sql('product_dimension', engine, if_exists='append', index=False)
    print("Products successfully loaded into product_dimension.")
except Exception as e:
    print(f"Error during loading data into product_dimension: {e}")

# Load time_dimension
try:
    time_dim = cleaned_data[['InvoiceDate']].drop_duplicates()
    time_dim['year'] = time_dim['InvoiceDate'].dt.year
    time_dim['month'] = time_dim['InvoiceDate'].dt.month
    time_dim['day'] = time_dim['InvoiceDate'].dt.day
    time_dim.rename(columns={'InvoiceDate': 'invoice_date'}, inplace=True)
    time_dim.to_sql('time_dimension', engine, if_exists='append', index=False)
    print("Time data successfully loaded into time_dimension.")
except Exception as e:
    print(f"Error during loading data into time_dimension: {e}")

# Calculate total_sales
cleaned_data['total_sales'] = cleaned_data['Quantity'] * cleaned_data['UnitPrice']

# Load sales_fact with proper foreign key mappings
try:
    product_map = pd.read_sql("SELECT stockcode, product_id FROM product_dimension", engine)
    cleaned_data = cleaned_data.merge(product_map, how='left', left_on='StockCode', right_on='stockcode')

    time_map = pd.read_sql("SELECT invoice_date, time_id FROM time_dimension", engine)
    time_map['invoice_date'] = pd.to_datetime(time_map['invoice_date'])
    cleaned_data = cleaned_data.merge(time_map, how='left', left_on='InvoiceDate', right_on='invoice_date')

    country_map = pd.read_sql("SELECT country, country_id FROM country_dimension", engine)
    cleaned_data = cleaned_data.merge(country_map, how='left', left_on='Country', right_on='country')

    existing_customers = pd.read_sql("SELECT customer_id FROM customer_dimension", engine)
    cleaned_data = cleaned_data[cleaned_data['CustomerID'].isin(existing_customers['customer_id'])]

    sales_fact = cleaned_data.groupby(['CustomerID', 'country_id', 'time_id', 'InvoiceNo', 'product_id']) \
        .agg({
            'Quantity': 'sum',
            'total_sales': 'sum'
        }).reset_index()

    sales_fact = sales_fact.rename(columns={
        'InvoiceNo': 'invoice_no',
        'CustomerID': 'customer_id',
        'Quantity': 'quantity',
        'total_sales': 'total_sales'
    })

    # Debug print to check data before insertion
    print("Data ready for sales_fact:", sales_fact.head())

    # Check for missing foreign keys
    print("Missing product_id:", sales_fact[sales_fact['product_id'].isnull()].head())
    print("Missing time_id:", sales_fact[sales_fact['time_id'].isnull()].head())
    print("Missing customer_id:", sales_fact[sales_fact['customer_id'].isnull()].head())
    print("Missing country_id:", sales_fact[sales_fact['country_id'].isnull()].head())

    sales_fact.to_sql('sales_fact', engine, if_exists='append', index=False)
    print("Sales data successfully loaded into sales_fact.")
except Exception as e:
    print(f"Error during loading data into sales_fact: {e}")
