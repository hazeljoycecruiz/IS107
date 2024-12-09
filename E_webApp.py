import dash
from dash import dcc, html, Input, Output, State, dash_table
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
import warnings
warnings.filterwarnings('ignore')

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Load data using RetailAnalytics class from D_DataMining.py
class RetailAnalytics:
    def __init__(self):
        self.engine = create_engine('postgresql://postgres:postgres@localhost/data_warehouse')

    def load_data(self):
        query = """
        SELECT 
            c.customer_id, 
            SUM(s.sales) as total_sales,
            AVG(s.sales) as avg_sales,
            COUNT(s.sales) as transaction_count,
            MIN(t.order_date) as first_purchase,
            MAX(t.order_date) as last_purchase,
            EXTRACT(MONTH FROM MIN(t.order_date)) as month,
            EXTRACT(YEAR FROM MIN(t.order_date)) as year
        FROM sales_fact s
        JOIN customer_dim c ON s.customer_id = c.customer_id
        JOIN time_dim t ON s.time_id = t.time_id
        GROUP BY c.customer_id
        """
        df = pd.read_sql(query, self.engine)
        df['customer_lifetime'] = (
            pd.to_datetime(df['last_purchase']) - 
            pd.to_datetime(df['first_purchase'])
        ).dt.days
        df['purchase_frequency'] = df['transaction_count'] / df['customer_lifetime'].replace(0, 1)
        return df

    def perform_customer_segmentation(self, df, n_clusters=3):
        features = ['total_sales', 'transaction_count']
        X = df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['segment'] = kmeans.fit_predict(X_scaled)
        return df, kmeans

    def perform_sales_prediction(self, df):
        X = df[['avg_sales', 'transaction_count', 'purchase_frequency', 'month', 'year']]
        y = df['total_sales']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {
            'model': model,
            'mse': mse,
            'r2': r2,
            'predicted': y_pred,
            'actual': y_test
        }

analytics = RetailAnalytics()
data = analytics.load_data()

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

    total_sales = data['total_sales'].sum()
    total_customers = data['customer_id'].nunique()
    avg_order_value = total_sales / total_customers

    daily_sales = data.groupby('first_purchase')['total_sales'].sum().reset_index()
    fig_sales_trend = px.line(daily_sales, x='first_purchase', y='total_sales', title='Daily Sales Trend')

    return html.Div([
        html.H1("Retail Analytics Dashboard"),
        html.Div([
            html.Div(f"Total Sales: ${total_sales:,.2f}", style={'width': '33%', 'display': 'inline-block'}),
            html.Div(f"Total Customers: {total_customers:,}", style={'width': '33%', 'display': 'inline-block'}),
            html.Div(f"Average Order Value: ${avg_order_value:,.2f}", style={'width': '33%', 'display': 'inline-block'}),
        ]),
        dcc.Graph(figure=fig_sales_trend)
    ])

def render_customer_segmentation():
    if data is None:
        return html.Div("Unable to load data. Please check your database connection.")

    customer_segments, _ = analytics.perform_customer_segmentation(data)
    fig_segments = px.scatter(customer_segments, x='total_sales', y='transaction_count', color='segment', title='Customer Segments')

    segment_analysis = customer_segments.groupby('segment').agg({
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
            page_size=5,
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

    prediction_results = analytics.perform_sales_prediction(data)
    y_test = prediction_results['actual']
    y_pred = prediction_results['predicted']

    fig_prediction = go.Figure()
    fig_prediction.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='markers', name='Actual Sales'))
    fig_prediction.add_trace(go.Scatter(x=y_test.index, y=y_pred, mode='lines', name='Predicted Sales'))

    report_summary = f"""
    Sales Prediction Analysis
    -------------------------
    Model Performance:
    - R² Score: {prediction_results['r2']:.3f}
    - RMSE: ${prediction_results['mse']:,.2f}
    """

    return html.Div([
        html.H1("Sales Prediction"),
        dcc.Graph(figure=fig_prediction),
        html.H2("Report Summary"),
        html.Pre(report_summary)
    ])

def render_data_explorer():
    if data is None:
        return html.Div("Unable to load data. Please check your database connection.")

    return html.Div([
        html.H1("Data Explorer"),
        dash_table.DataTable(
            data=data.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in data.columns],
            page_size=10,
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
