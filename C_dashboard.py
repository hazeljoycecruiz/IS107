import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from sqlalchemy import create_engine

# Database connection setup
db_name = "data_warehouse"
db_user = "postgres"
db_password = "postgres"
db_host = "localhost"
db_port = 5432
engine = create_engine(f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

# Fetch Data from Database
def load_data():
    sales_query = """
        SELECT s.invoice_no, s.customer_id, s.quantity, s.total_sales, s.product_id, 
               t.invoice_date, c.country, p.description AS product_description
        FROM sales_fact s
        JOIN time_dimension t ON s.time_id = t.time_id
        JOIN country_dimension c ON s.country_id = c.country_id
        JOIN product_dimension p ON s.product_id = p.product_id
    """
    sales_data = pd.read_sql(sales_query, engine)
    return sales_data

# Load the data
data = load_data()

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Sales Dashboard"

# Layout
app.layout = html.Div([
    html.H1("Sales Dashboard", style={'textAlign': 'center'}),
    
    # Filters
    html.Div([
        html.Label("Date Range:"),
        dcc.DatePickerRange(
            id='date-picker',
            start_date=data['invoice_date'].min(),
            end_date=data['invoice_date'].max(),
            display_format='YYYY-MM-DD'
        ),
        html.Label("Countries:"),
        dcc.Dropdown(
            id='country-filter',
            options=[{'label': country, 'value': country} for country in data['country'].unique()],
            multi=True,
            placeholder="Select countries"
        ),
        html.Label("Products:"),
        dcc.Dropdown(
            id='product-filter',
            options=[{'label': product, 'value': product} for product in data['product_description'].unique()],
            multi=True,
            placeholder="Select products"
        ),
    ], style={'margin': '20px'}),
    
    # Key Metrics
    html.Div([
        html.Div(id='total-sales-metric', style={'margin': '10px', 'display': 'inline-block'}),
        html.Div(id='total-quantity-metric', style={'margin': '10px', 'display': 'inline-block'}),
        html.Div(id='top-product-metric', style={'margin': '10px', 'display': 'inline-block'}),
        html.Div(id='top-region-metric', style={'margin': '10px', 'display': 'inline-block'}),
    ], style={'textAlign': 'center', 'margin': '20px'}),
    
    # Visualizations
    html.Div([
        dcc.Graph(id='sales-by-region'),
        dcc.Graph(id='top-products'),
        dcc.Graph(id='sales-over-time'),
        dcc.Graph(id='quantity-by-product')
    ])
])

# Callbacks
@app.callback(
    [
        Output('total-sales-metric', 'children'),
        Output('total-quantity-metric', 'children'),
        Output('top-product-metric', 'children'),
        Output('top-region-metric', 'children'),
        Output('sales-by-region', 'figure'),
        Output('top-products', 'figure'),
        Output('sales-over-time', 'figure'),
        Output('quantity-by-product', 'figure'),
    ],
    [
        Input('date-picker', 'start_date'),
        Input('date-picker', 'end_date'),
        Input('country-filter', 'value'),
        Input('product-filter', 'value'),
    ]
)
def update_dashboard(start_date, end_date, countries, products):
    # Filter data
    filtered_data = data.copy()
    
    if start_date and end_date:
        filtered_data = filtered_data[
            (filtered_data['invoice_date'] >= start_date) & 
            (filtered_data['invoice_date'] <= end_date)
        ]
    
    if countries:
        filtered_data = filtered_data[filtered_data['country'].isin(countries)]
    
    if products:
        filtered_data = filtered_data[filtered_data['product_description'].isin(products)]
    
    # Key Metrics
    total_sales = filtered_data['total_sales'].sum()
    total_quantity = filtered_data['quantity'].sum()
    top_product = filtered_data.groupby('product_description')['total_sales'].sum().idxmax() if not filtered_data.empty else "No Data"
    top_region = filtered_data.groupby('country')['total_sales'].sum().idxmax() if not filtered_data.empty else "No Data"

    # Sales by Region
    sales_by_region = filtered_data.groupby('country')['total_sales'].sum().reset_index()
    fig1 = px.bar(sales_by_region, x='country', y='total_sales', title="Total Sales by Region", labels={'total_sales': 'Total Sales ($)'})

    # Top Products
    top_products = filtered_data.groupby('product_description')['total_sales'].sum().nlargest(10).reset_index()
    fig2 = px.bar(top_products, x='product_description', y='total_sales', title="Top 10 Selling Products", labels={'total_sales': 'Total Sales ($)'})

    # Sales Over Time
    sales_over_time = filtered_data.groupby('invoice_date')['total_sales'].sum().reset_index()
    fig3 = px.line(sales_over_time, x='invoice_date', y='total_sales', title="Sales Over Time", labels={'total_sales': 'Total Sales ($)'})

    # Quantity Sold by Product
    quantity_by_product = filtered_data.groupby('product_description')['quantity'].sum().nlargest(10).reset_index()
    fig4 = px.bar(quantity_by_product, x='product_description', y='quantity', title="Quantity Sold by Product", labels={'quantity': 'Quantity Sold'})

    return (
        f"Total Sales: ${total_sales:,.2f}",
        f"Total Quantity Sold: {total_quantity:,}",
        f"Top-Selling Product: {top_product}",
        f"Top Region: {top_region}",
        fig1,
        fig2,
        fig3,
        fig4
    )

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
