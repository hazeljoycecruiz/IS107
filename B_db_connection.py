import pandas as pd
from sqlalchemy import create_engine, text

# Database connection setup
db_name = "data_warehouse"
db_user = "postgres"
db_password = "postgres"
db_host = "localhost"
db_port = 5432
engine = create_engine(f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

# Load cleaned data
csv_path = "cleaned_data.csv"
cleaned_data = pd.read_csv(csv_path)

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
    # Map country to country_id
    country_map = pd.read_sql("SELECT country, country_id FROM country_dimension", engine)
    customer_dim = cleaned_data[['CustomerID', 'Country']].drop_duplicates().rename(columns={'CustomerID': 'customer_id', 'Country': 'country'})
    customer_dim = customer_dim.merge(country_map, how='left', on='country')  # Map country_id
    customer_dim = customer_dim[['customer_id', 'country_id']]

    # Prepare insert query to handle duplicates
    insert_query = """
    INSERT INTO customer_dimension (customer_id, country_id)
    VALUES (:customer_id, :country_id)
    ON CONFLICT (customer_id) DO NOTHING;
    """

    # Insert data into the database, using ON CONFLICT DO NOTHING
    with engine.connect() as conn:
        for _, row in customer_dim.iterrows():
            conn.execute(text(insert_query), {'customer_id': row['customer_id'], 'country_id': row['country_id']})

    print("Customers successfully loaded into customer_dimension.")
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
    time_dim['year'] = pd.to_datetime(time_dim['InvoiceDate']).dt.year
    time_dim['month'] = pd.to_datetime(time_dim['InvoiceDate']).dt.month
    time_dim['day'] = pd.to_datetime(time_dim['InvoiceDate']).dt.day
    time_dim.rename(columns={'InvoiceDate': 'invoice_date'}, inplace=True)
    time_dim.to_sql('time_dimension', engine, if_exists='append', index=False)
    print("Time data successfully loaded into time_dimension.")
except Exception as e:
    print(f"Error during loading data into time_dimension: {e}")

# Ensure cleaned_data has the necessary columns
cleaned_data['quantity'] = cleaned_data['Quantity']  # Assuming 'Quantity' is available in the raw data
cleaned_data['total_sales'] = cleaned_data['quantity'] * cleaned_data['UnitPrice']  # Assuming 'UnitPrice' exists

# Load sales_fact with proper foreign key mappings
try:
    # Map product_id
    product_map = pd.read_sql("SELECT stockcode, product_id FROM product_dimension", engine)
    cleaned_data = cleaned_data.merge(product_map, how='left', left_on='StockCode', right_on='stockcode')

    # Map time_id
    time_map = pd.read_sql("SELECT invoice_date, time_id FROM time_dimension", engine)
    cleaned_data = cleaned_data.merge(time_map, how='left', left_on='InvoiceDate', right_on='invoice_date')

    # Map country_id
    country_map = pd.read_sql("SELECT country, country_id FROM country_dimension", engine)
    cleaned_data = cleaned_data.merge(country_map, how='left', left_on='Country', right_on='country')

    # Ensure all customer_id values exist in customer_dimension
    existing_customers = pd.read_sql("SELECT customer_id FROM customer_dimension", engine)
    cleaned_data = cleaned_data[cleaned_data['CustomerID'].isin(existing_customers['customer_id'])]

    # Prepare sales_fact by grouping data for the same customer purchasing different items
    sales_fact = cleaned_data.groupby(['CustomerID', 'country_id', 'time_id', 'InvoiceNo']) \
        .agg({
            'product_id': 'unique',  # All unique product_ids for this transaction
            'quantity': 'sum',  # Total quantity for each transaction
            'total_sales': 'sum'  # Total sales for this transaction
        }).reset_index()

    # Flatten 'product_id' (unique products per transaction) to be inserted correctly into sales_fact
    sales_fact['product_id'] = sales_fact['product_id'].apply(lambda x: ','.join(map(str, x)))

    # Prepare sales_fact columns for insertion
    sales_fact = sales_fact.rename(columns={
        'InvoiceNo': 'invoice_no',
        'CustomerID': 'customer_id',
        'quantity': 'quantity',
        'total_sales': 'total_sales'
    })

    # Load sales_fact into the database
    sales_fact.to_sql('sales_fact', engine, if_exists='append', index=False)
    print("Sales data successfully loaded into sales_fact.")
except Exception as e:
    print(f"Error during loading data into sales_fact: {e}")
