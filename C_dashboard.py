# import dash
# import pandas as pd
# import plotly.express as px
# from dash import dcc, html
# from dash.dependencies import Input, Output

# # Load cleaned data (replace with the correct path to your dataset)
# data = pd.read_csv('cleaned_data_v3.csv')

# # Strip spaces from column names to avoid issues with extra spaces
# data.columns = data.columns.str.strip()

# # Check if 'Customer Name' column exists and group by 'Customer Name' for top customers
# if 'Customer Name' in data.columns:
#     top_customers = data.groupby('Customer Name')['Sales'].sum().sort_values(ascending=False).head(5)
# else:
#     print("The 'Customer Name' column does not exist in the dataset.")
#     top_customers = pd.Series()  # Empty series as fallback

# # Key Metrics Calculations
# total_sales = data['Sales'].sum()
# total_orders = data['Order ID'].nunique()  # Assuming 'Order ID' represents a unique order
# average_order_value = total_sales / total_orders if total_orders else 0
# top_selling_products = data.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(10)
# sales_by_region = data.groupby('Region')['Sales'].sum()

# # Monthly Sales Growth (Month-over-Month)
# data['Order Date'] = pd.to_datetime(data['Order Date'])
# monthly_sales = data.groupby(data['Order Date'].dt.to_period('M'))['Sales'].sum()
# sales_growth_mom = monthly_sales.pct_change() * 100  # MoM Growth

# # Additional Metrics (Sales by Payment Method)
# sales_by_payment_method = data.groupby('Ship Mode')['Sales'].sum()

# # Dashboard Layout
# app = dash.Dash(__name__)

# app.layout = html.Div([
#     html.Div([
#         html.H1('Sales Analytics Dashboard', className='header'),
#     ], className='header-container'),

#     html.Div([
#         # Key Metrics
#         html.Div([
#             html.Div([
#                 html.H3('Total Sales'),
#                 html.P(f'${total_sales:,.2f}')
#             ], className='metric-box'),

#             html.Div([
#                 html.H3('Total Orders'),
#                 html.P(f'{total_orders:,}')
#             ], className='metric-box'),

#             html.Div([
#                 html.H3('Average Order Value'),
#                 html.P(f'${average_order_value:,.2f}')
#             ], className='metric-box'),

#             html.Div([
#                 html.H3('Sales Growth (MoM)'),
#                 html.P(f'{sales_growth_mom.iloc[-1]:.2f}%')  # Show the last month's growth rate
#             ], className='metric-box'),

#             html.Div([
#                 html.H3('Top Selling Products'),
#                 html.Ul([html.Li(f'{product}: ${value:,.2f}') for product, value in top_selling_products.items()])
#             ], className='metric-box'),

#             html.Div([
#                 html.H3('Top 5 Customers'),
#                 html.Ul([html.Li(f'{customer}: ${value:,.2f}') for customer, value in top_customers.items()])
#             ], className='metric-box'),

#             html.Div([
#                 html.H3('Sales by Region'),
#                 dcc.Graph(
#                     id='sales-region-chart',
#                     figure=px.bar(sales_by_region, x=sales_by_region.index, y=sales_by_region.values, title="Sales by Region")
#                 )
#             ], className='metric-box'),

#             html.Div([
#                 html.H3('Sales by Ship Mode'),
#                 dcc.Graph(
#                     id='sales-payment-method-chart',
#                     figure=px.pie(sales_by_payment_method, names=sales_by_payment_method.index, values=sales_by_payment_method.values, title="Sales by Ship Mode")
#                 )
#             ], className='metric-box'),
#         ], className='metrics-container'),

#         # Filters Section
#         html.Div([
#             html.H3('Filters', className='filters-header'),
#             dcc.DatePickerRange(
#                 id='date-picker-range',
#                 start_date=data['Order Date'].min(),
#                 end_date=data['Order Date'].max(),
#                 display_format='YYYY-MM-DD',
#                 className='date-picker'
#             ),
#             dcc.Dropdown(
#                 id='category-dropdown',
#                 options=[{'label': category, 'value': category} for category in data['Category'].unique()],
#                 multi=True,
#                 placeholder='Select Category',
#                 className='category-dropdown'
#             )
#         ], className='filter-container'),

#         # Visualizations Section
#         html.Div([
#             dcc.Graph(id='sales-category-chart'),
#         ], className='graph-container')

#     ], className='content-container'),
# ])

# # Callbacks for interactivity
# @app.callback(
#     Output('sales-category-chart', 'figure'),
#     [
#         Input('date-picker-range', 'start_date'),
#         Input('date-picker-range', 'end_date'),
#         Input('category-dropdown', 'value')
#     ]
# )
# def update_graph(start_date, end_date, selected_categories):
#     filtered_data = data[
#         (data['Order Date'] >= start_date) &
#         (data['Order Date'] <= end_date)
#     ]
    
#     if selected_categories:
#         filtered_data = filtered_data[filtered_data['Category'].isin(selected_categories)]
    
#     sales_by_category = filtered_data.groupby('Category')['Sales'].sum()
    
#     return px.bar(sales_by_category, x=sales_by_category.index, y=sales_by_category.values, title='Sales by Category')

# # Run the app
# if __name__ == '__main__':
#     app.run_server(debug=True)


import dash
import pandas as pd
import plotly.express as px
from dash import dcc, html
from dash.dependencies import Input, Output
from sqlalchemy import create_engine

# Create a SQLAlchemy engine
engine = create_engine('postgresql://postgres:postgres@localhost/data_warehouse')

# Load data using SQLAlchemy engine
try:
    query = """
    SELECT o.order_id, o.order_date, o.ship_date, o.ship_mode, 
           c.customer_name, c.region, 
           p.product_name, p.category, p.sub_category, 
           s.sales
    FROM sales s
    JOIN orders o ON s.order_id = o.order_id
    JOIN customers c ON s.customer_id = c.customer_id
    JOIN products p ON s.product_id = p.product_id;
    """
    data = pd.read_sql(query, engine)
    print("Data loaded successfully from the database!")
    print("Columns in the DataFrame:", data.columns.tolist())
except Exception as e:
    print("An error occurred while connecting to the database or loading data.")
    print(e)

try:
    # Convert date columns to datetime
    data['order_date'] = pd.to_datetime(data['order_date'])
    data['ship_date'] = pd.to_datetime(data['ship_date'])

    # Key Metrics Calculations
    total_sales = data['sales'].sum()
    total_orders = data['order_id'].nunique()
    average_order_value = total_sales / total_orders if total_orders else 0
    top_selling_products = data.groupby('product_name')['sales'].sum().sort_values(ascending=False).head(10)
    sales_by_region = data.groupby('region')['sales'].sum()

    # Monthly Sales Growth (Month-over-Month)
    monthly_sales = data.groupby(data['order_date'].dt.to_period('M'))['sales'].sum()
    sales_growth_mom = monthly_sales.pct_change() * 100

    # Dashboard Layout
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.Div([
            html.H1('Sales Analytics Dashboard', className='header'),
        ], className='header-container'),

        html.Div([
            # Key Metrics
            html.Div([
                html.Div([
                    html.H3('Total Sales'),
                    html.P(f'${total_sales:,.2f}')
                ], className='metric-box'),

                html.Div([
                    html.H3('Total Orders'),
                    html.P(f'{total_orders:,}')
                ], className='metric-box'),

                html.Div([
                    html.H3('Average Order Value'),
                    html.P(f'${average_order_value:,.2f}')
                ], className='metric-box'),

                html.Div([
                    html.H3('Sales Growth (MoM)'),
                    html.P(f'{sales_growth_mom.iloc[-1]:.2f}%')
                ], className='metric-box'),

                html.Div([
                    html.H3('Top Selling Products'),
                    html.Ul([html.Li(f'{product}: ${value:,.2f}') 
                            for product, value in top_selling_products.items()])
                ], className='metric-box'),

                html.Div([
                    html.H3('Sales by Region'),
                    dcc.Graph(
                        id='sales-region-chart',
                        figure=px.bar(sales_by_region, 
                                    x=sales_by_region.index, 
                                    y=sales_by_region.values, 
                                    title="Sales by Region")
                    )
                ], className='metric-box'),
            ], className='metrics-container'),

            # Filters Section
            html.Div([
                html.H3('Filters', className='filters-header'),
                dcc.DatePickerRange(
                    id='date-picker-range',
                    start_date=data['order_date'].min(),
                    end_date=data['order_date'].max(),
                    display_format='YYYY-MM-DD',
                    className='date-picker'
                ),
                dcc.Dropdown(
                    id='category-dropdown',
                    options=[{'label': category, 'value': category} 
                            for category in data['category'].unique()],
                    multi=True,
                    placeholder='Select Category',
                    className='category-dropdown'
                )
            ], className='filter-container'),

            # Visualizations Section
            html.Div([
                dcc.Graph(id='sales-category-chart'),
            ], className='graph-container')

        ], className='content-container'),
    ])

    # Callbacks for interactivity
    @app.callback(
        Output('sales-category-chart', 'figure'),
        [
            Input('date-picker-range', 'start_date'),
            Input('date-picker-range', 'end_date'),
            Input('category-dropdown', 'value')
        ]
    )
    def update_graph(start_date, end_date, selected_categories):
        filtered_data = data[
            (data['order_date'] >= start_date) &
            (data['order_date'] <= end_date)
        ]
        
        if selected_categories:
            filtered_data = filtered_data[filtered_data['category'].isin(selected_categories)]
        
        sales_by_category = filtered_data.groupby('category')['sales'].sum()
        
        return px.bar(sales_by_category, 
                     x=sales_by_category.index, 
                     y=sales_by_category.values, 
                     title='Sales by Category')

    # Run the app
    if __name__ == '__main__':
        app.run_server(debug=True)

except Exception as e:
    print(f"An error occurred: {e}")
