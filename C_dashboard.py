import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine

# Database connection setup
db_name = "data_warehouse"
db_user = "postgres"
db_password = "postgres"
db_host = "localhost"
db_port = 5432
engine = create_engine(f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

# Load data from the database
sales_query = """
SELECT sf.customer_id, sf.quantity, sf.total_sales, sf.invoice_no, sf.time_id, pd.description AS product_description,
       td.invoice_date, td.year, td.month, cd.country
FROM sales_fact sf
JOIN product_dimension pd ON sf.product_id = pd.product_id
JOIN time_dimension td ON sf.time_id = td.time_id
JOIN country_dimension cd ON sf.country_id = cd.country_id;
"""
data = pd.read_sql(sales_query, engine)

# Data preprocessing
data['invoice_date'] = pd.to_datetime(data['invoice_date'])

# Create the Dash app
app = dash.Dash(__name__)
app.title = "Sales Dashboard"

# Layout
app.layout = html.Div([
    html.H1("Sales Dashboard", style={'textAlign': 'center'}),

    # Filters
    html.Div([
        html.Div([
            html.Label("Select Date Range:"),
            dcc.DatePickerRange(
                id='date-range',
                start_date=data['invoice_date'].min(),
                end_date=data['invoice_date'].max(),
                display_format='YYYY-MM-DD',
            )
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '10px'}),
        
        html.Div([
            html.Label("Select Country:"),
            dcc.Dropdown(
                id='country-filter',
                options=[{'label': country, 'value': country} for country in data['country'].unique()],
                multi=True,
                placeholder="Filter by Country"
            )
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '10px'}),
        
        html.Div([
            html.Label("Select Product:"),
            dcc.Dropdown(
                id='product-filter',
                options=[{'label': product, 'value': product} for product in data['product_description'].unique()],
                multi=True,
                placeholder="Filter by Product"
            )
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}),

    # Key Metrics
    html.Div([
        html.Div([
            html.H4("Total Sales"),
            html.P(id="total-sales", style={'fontSize': '24px', 'fontWeight': 'bold'})
        ], style={'width': '30%', 'display': 'inline-block', 'textAlign': 'center'}),

        html.Div([
            html.H4("Total Quantity Sold"),
            html.P(id="total-quantity", style={'fontSize': '24px', 'fontWeight': 'bold'})
        ], style={'width': '30%', 'display': 'inline-block', 'textAlign': 'center'}),

        html.Div([
            html.H4("Unique Customers"),
            html.P(id="unique-customers", style={'fontSize': '24px', 'fontWeight': 'bold'})
        ], style={'width': '30%', 'display': 'inline-block', 'textAlign': 'center'}),
    ]),

    # Visualizations
    html.Div([
        dcc.Graph(id="sales-by-region"),
        dcc.Graph(id="top-products"),
        dcc.Graph(id="sales-trend"),
    ])
])


# Callbacks
@app.callback(
    [
        Output("total-sales", "children"),
        Output("total-quantity", "children"),
        Output("unique-customers", "children"),
        Output("sales-by-region", "figure"),
        Output("top-products", "figure"),
        Output("sales-trend", "figure"),
    ],
    [
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
        Input("country-filter", "value"),
        Input("product-filter", "value"),
    ]
)
def update_dashboard(start_date, end_date, selected_countries, selected_products):
    # Filter data
    filtered_data = data[
        (data['invoice_date'] >= pd.to_datetime(start_date)) &
        (data['invoice_date'] <= pd.to_datetime(end_date))
    ]
    
    if selected_countries:
        filtered_data = filtered_data[filtered_data['country'].isin(selected_countries)]
    
    if selected_products:
        filtered_data = filtered_data[filtered_data['product_description'].isin(selected_products)]

    # Key metrics
    total_sales = f"${filtered_data['total_sales'].sum():,.2f}"
    total_quantity = f"{filtered_data['quantity'].sum():,}"
    unique_customers = f"{filtered_data['customer_id'].nunique()}"

    # Sales by Region
    sales_by_region_fig = px.bar(
        filtered_data.groupby('country')['total_sales'].sum().reset_index(),
        x='country', y='total_sales',
        title="Sales by Region", labels={'total_sales': 'Total Sales', 'country': 'Country'}
    )

    # Top Products
    top_products_fig = px.bar(
        filtered_data.groupby('product_description')['total_sales'].sum().nlargest(10).reset_index(),
        x='product_description', y='total_sales',
        title="Top 10 Products by Sales", labels={'total_sales': 'Total Sales', 'product_description': 'Product'}
    )

    # Sales Trend
    sales_trend_fig = px.line(
        filtered_data.groupby('invoice_date')['total_sales'].sum().reset_index(),
        x='invoice_date', y='total_sales',
        title="Sales Trend Over Time", labels={'total_sales': 'Total Sales', 'invoice_date': 'Date'}
    )

    return total_sales, total_quantity, unique_customers, sales_by_region_fig, top_products_fig, sales_trend_fig

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
