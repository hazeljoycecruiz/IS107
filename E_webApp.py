import dash
from dash import dcc, html, Input, Output, State
import dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from datetime import datetime

# Initialize the Dash app with suppress_callback_exceptions=True
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Load data from PostgreSQL database
def load_data():
    try:
        engine = create_engine('postgresql://postgres:postgres@localhost/data_warehouse')
        data = pd.read_sql("""
            SELECT 
                o.order_id, o.order_date, o.ship_date, o.ship_mode,
                c.customer_id, c.customer_name, c.segment, c.region,
                p.product_id, p.product_name, p.category, p.sub_category,
                s.sales
            FROM sales s
            JOIN orders o ON s.order_id = o.order_id
            JOIN customers c ON s.customer_id = c.customer_id
            JOIN products p ON s.product_id = p.product_id
        """, engine)
        data['order_date'] = pd.to_datetime(data['order_date'])
        return data
    except Exception as e:
        print(f"Data loading error: {str(e)}")
        return None

data = load_data()

# Define the layout of the app
app.layout = html.Div([
    dcc.Tabs(id='tabs', value='dashboard', children=[
        dcc.Tab(label='Dashboard', value='dashboard'),
        dcc.Tab(label='Customer Segmentation', value='customer_segmentation'),
        dcc.Tab(label='Sales Prediction', value='sales_prediction'),
        dcc.Tab(label='Data Explorer', value='data_explorer'),
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'dashboard':
        return render_dashboard()
    elif tab == 'customer_segmentation':
        return render_customer_segmentation()
    elif tab == 'sales_prediction':
        return render_sales_prediction()
    elif tab == 'data_explorer':
        return render_data_explorer()

def render_dashboard():
    if data is None:
        return html.Div("Unable to load data. Please check your database connection.")

    total_sales = data['sales'].sum()
    total_orders = data['order_id'].nunique()
    total_customers = data['customer_id'].nunique()
    avg_order_value = total_sales / total_orders

    daily_sales = data.groupby('order_date')['sales'].sum().reset_index()
    fig_sales_trend = px.line(daily_sales, x='order_date', y='sales', title='Daily Sales Trend')

    region_sales = data.groupby('region')['sales'].sum().reset_index()
    fig_sales_region = px.pie(region_sales, values='sales', names='region', title='Sales by Region')

    category_sales = data.groupby('category')['sales'].sum().reset_index()
    fig_sales_category = px.bar(category_sales, x='category', y='sales', title='Sales by Category')

    return html.Div([
        html.H1("Retail Analytics Dashboard"),
        html.Div([
            html.Div(f"Total Sales: ${total_sales:,.2f}", style={'width': '25%', 'display': 'inline-block'}),
            html.Div(f"Total Orders: {total_orders:,}", style={'width': '25%', 'display': 'inline-block'}),
            html.Div(f"Total Customers: {total_customers:,}", style={'width': '25%', 'display': 'inline-block'}),
            html.Div(f"Average Order Value: ${avg_order_value:,.2f}", style={'width': '25%', 'display': 'inline-block'}),
        ]),
        dcc.Graph(figure=fig_sales_trend),
        html.Div([
            dcc.Graph(figure=fig_sales_region, style={'width': '50%', 'display': 'inline-block'}),
            dcc.Graph(figure=fig_sales_category, style={'width': '50%', 'display': 'inline-block'}),
        ])
    ])

def render_customer_segmentation():
    if data is None:
        return html.Div("Unable to load data. Please check your database connection.")

    customer_metrics = data.groupby('customer_id').agg({
        'sales': ['sum', 'mean', 'count'],
        'order_id': 'nunique'
    }).reset_index()
    customer_metrics.columns = ['customer_id', 'total_sales', 'avg_sales', 'transaction_count', 'order_count']

    features = ['total_sales', 'transaction_count', 'order_count']
    X = customer_metrics[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42)
    customer_metrics['Segment'] = kmeans.fit_predict(X_scaled)

    fig_segments = px.scatter(customer_metrics, x='total_sales', y='transaction_count', color='Segment', title='Customer Segments')

    segment_analysis = customer_metrics.groupby('Segment').agg({
        'customer_id': 'count',
        'total_sales': 'mean',
        'transaction_count': 'mean'
    }).round(2).reset_index()

    return html.Div([
        html.H1("Customer Segmentation Analysis"),
        dcc.Graph(figure=fig_segments),
        html.H2("Segment Analysis"),
        dash_table.DataTable(
            data=segment_analysis.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in segment_analysis.columns],
            page_size=5,  # Display 5 rows per page
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'}
            ]
        )
    ])

def render_sales_prediction():
    if data is None:
        return html.Div("Unable to load data. Please check your database connection.")

    # Prepare time series data for sales prediction
    data['order_date'] = pd.to_datetime(data['order_date'])
    monthly_sales = data.groupby(data['order_date'].dt.to_period('M'))['sales'].sum().reset_index()
    monthly_sales['order_date'] = monthly_sales['order_date'].astype(str)

    # Add time-based features
    monthly_sales['time_index'] = np.arange(len(monthly_sales))
    monthly_sales['month'] = pd.to_datetime(monthly_sales['order_date']).dt.month
    monthly_sales['year'] = pd.to_datetime(monthly_sales['order_date']).dt.year

    # Prepare features and target
    X = monthly_sales[['time_index', 'month', 'year']]
    y = monthly_sales['sales']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Predict future sales
    months_to_predict = 6
    last_time_index = monthly_sales['time_index'].iloc[-1]
    future_dates = pd.date_range(start=pd.to_datetime(monthly_sales['order_date'].iloc[-1]) + pd.DateOffset(months=1), periods=months_to_predict, freq='M')
    future_X = pd.DataFrame({
        'time_index': np.arange(last_time_index + 1, last_time_index + 1 + months_to_predict),
        'month': future_dates.month,
        'year': future_dates.year
    })

    future_predictions = model.predict(future_X)

    # Create prediction figure
    fig_prediction = go.Figure()
    fig_prediction.add_trace(go.Scatter(x=monthly_sales['order_date'], y=monthly_sales['sales'], name='Historical Sales'))
    fig_prediction.add_trace(go.Scatter(x=monthly_sales['order_date'].iloc[-len(y_pred):], y=y_pred, name='Predicted Sales', line=dict(dash='dash')))

    # Additional visualization: Bar chart for predicted sales
    future_months = future_dates.strftime('%Y-%m')
    fig_future_sales = px.bar(x=future_months, y=future_predictions, labels={'x': 'Month', 'y': 'Predicted Sales'}, title='Predicted Sales for Next 6 Months')

    # Report summary
    report_summary = f"""
    Sales Prediction Analysis
    -------------------------
    Model Performance:
    - R² Score: {r2:.3f}
    - RMSE: ${rmse:,.2f}

    Predicted Sales:
    - Next Month: ${future_predictions[0]:,.2f}
    - Last Month in Forecast: ${future_predictions[-1]:,.2f}
    """

    return html.Div([
        html.H1("Sales Prediction"),
        dcc.Graph(figure=fig_prediction),
        html.H2("Future Sales Prediction"),
        dcc.Graph(figure=fig_future_sales),
        html.H2("Report Summary"),
        html.Pre(report_summary)
    ])

def render_data_explorer():
    if data is None:
        return html.Div("Unable to load data. Please check your database connection.")

    regions = data['region'].unique()
    categories = data['category'].unique()
    min_date = data['order_date'].min()
    max_date = data['order_date'].max()

    return html.Div([
        html.H1("Data Explorer"),
        html.Div([
            html.Label("Select Region"),
            dcc.Dropdown(
                id='region-filter',
                options=[{'label': region, 'value': region} for region in regions],
                multi=True
            ),
            html.Label("Select Category"),
            dcc.Dropdown(
                id='category-filter',
                options=[{'label': category, 'value': category} for category in categories],
                multi=True
            ),
            html.Label("Select Date Range"),
            dcc.DatePickerRange(
                id='date-range',
                start_date=min_date,
                end_date=max_date,
                display_format='YYYY-MM-DD'
            ),
            html.Button("Filter Data", id='filter-button', n_clicks=0)
        ]),
        html.Div(id='filtered-data-table')
    ])

@app.callback(
    Output('filtered-data-table', 'children'),
    Input('filter-button', 'n_clicks'),
    State('region-filter', 'value'),
    State('category-filter', 'value'),
    State('date-range', 'start_date'),
    State('date-range', 'end_date')
)
def update_data_table(n_clicks, selected_regions, selected_categories, start_date, end_date):
    if data is None:
        return html.Div("Unable to load data. Please check your database connection.")

    filtered_data = data.copy()

    if selected_regions:
        filtered_data = filtered_data[filtered_data['region'].isin(selected_regions)]
    if selected_categories:
        filtered_data = filtered_data[filtered_data['category'].isin(selected_categories)]
    if start_date and end_date:
        filtered_data = filtered_data[(filtered_data['order_date'] >= start_date) & (filtered_data['order_date'] <= end_date)]

    return html.Div([
        html.H2("Filtered Data"),
        dcc.Graph(
            figure=px.histogram(filtered_data, x='sales', nbins=50, title='Sales Distribution')
        ),
        dash_table.DataTable(
            data=filtered_data.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in filtered_data.columns],
            page_size=10,  # Display 10 rows per page
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'}
            ]
        )
    ])

if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_ui=True, dev_tools_props_check=True)
