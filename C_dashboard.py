import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine
from D_DataMining import RetailAnalytics

# Database connection setup
db_name = "data_warehouse"
db_user = "postgres"
db_password = "postgres"
db_host = "localhost"
db_port = 5432
engine = create_engine(f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

# Load data from the database
sales_query = """
SELECT sf.customer_id, pd.description AS product_description, pd.unit_price, sf.quantity, sf.total_price, sf.invoice_no, sf.time_id,
       td.invoice_date, td.year, td.month, td.day, cd.country
FROM sales_fact sf
JOIN product_dimension pd ON sf.product_id = pd.product_id
JOIN time_dimension td ON sf.time_id = td.time_id
JOIN country_dimension cd ON sf.country_id = cd.country_id;
"""
data = pd.read_sql(sales_query, engine)

# Data preprocessing
data['invoice_date'] = pd.to_datetime(data['invoice_date']).dt.date
data['customer_id'] = data['customer_id'].astype(str)
data['time_id'] = data['time_id'].astype(str)
data['year'] = data['year'].astype(int)

    

# Navigation sidebar
st.sidebar.title("Choose a Visualization:")
selected_module = st.sidebar.radio("", ["Data Visualization", "Data Mining"])

# Conditionally display filter options only for "Data Visualization"
if selected_module == "Data Visualization":
    # Filter Options section
    st.sidebar.title("Filter Options")

    # Date Range Selection
    date_range = st.sidebar.date_input(
        "Select Date Range", 
        [data['invoice_date'].min(), data['invoice_date'].max()],  
        key="date_range"
    )

    # Dropdowns for Country, Products, and Customers
    country = ["All"] + list(data['country'].unique())
    selected_country = st.sidebar.selectbox("Select Country", country)

    product = ["All"] + list(data['product_description'].unique())
    selected_product = st.sidebar.selectbox("Select Product", product)

# Main content based on module selection
if selected_module == "Data Visualization":
    # Filter data based on selections
    filtered_data = data
    if selected_country != "All":
        filtered_data = filtered_data[filtered_data['country'] == selected_country]
    if selected_product != "All":
        filtered_data = filtered_data[filtered_data['product_description'] == selected_product]

    # Sort the filtered data by customer_id in ascending order
    filtered_data = filtered_data.sort_values(by='customer_id')

    # Streamlit app
    st.header("Retail Analytics Dashboard")

    # Display all data
    st.subheader("All Sales Data")
    st.dataframe(filtered_data)

    # Key metrics
    st.subheader("Key Metrics")

    # Center align the metrics by using an empty column on each side
    col1, col2, col3 = st.columns([4, 3, 3])

    with col1:
        st.metric("Total Sales", f"${filtered_data['total_price'].sum():,.2f}")
    with col2:
        st.metric("Total Orders", f"{filtered_data['invoice_no'].nunique():,}")
    with col3:
        st.metric("Unique Customers", f"{filtered_data['customer_id'].nunique():,}")

    # Add some vertical spacing
    st.markdown("<br>", unsafe_allow_html=True)

    # Sales by Region (Pie Chart) - Adjusted to prevent overlap
    sales_by_region_data = filtered_data.groupby('country')['total_price'].sum().reset_index()

    # Adjust text position, size, and explode small slices
    sales_by_region_fig = px.pie(
        sales_by_region_data,
        values='total_price',
        names='country',
        title="Sales by Country",
        hole=0.0  # Optional: Set a donut-style pie chart if preferred
    )

    # Update layout to adjust text and prevent overlap
    sales_by_region_fig.update_traces(
        textposition='inside',  # Place text inside the slices
        textinfo='percent+label',  # Show percentage and country name
        pull=[0.1 if val < sales_by_region_data['total_price'].sum() * 0.05 else 0 for val in sales_by_region_data['total_price']]
    )

    sales_by_region_fig.update_layout(
        margin=dict(t=50, b=50, l=50, r=50),  # Add margins to avoid clipping
        height=600, width=800  # Increase size for better spacing
    )

    # Display chart
    st.plotly_chart(sales_by_region_fig)
    st.markdown(
    """
    This pie chart shows the share of total sales by country. Each slice represents a country, sized by its sales contribution, making it easy to compare the country's performance.
    """
    )

    # Top Products
    top_products_fig = px.bar(
        filtered_data.groupby('product_description')['total_price'].sum().nlargest(10).reset_index(),
        x='product_description', y='total_price',
        title="Top 10 Products by Sales", labels={'total_price': 'Total Sales', 'product_description': 'Product'}
    )
    st.plotly_chart(top_products_fig)
    st.markdown(
    """
    This bar chart ranks the top 10 highest-selling products by total sales revenue. Each bar represents a product, highlighting top performers and customer preferences.
    """
    )

    # Sales Trend
    sales_trend_fig = px.line(
        filtered_data.groupby('invoice_date')['total_price'].sum().reset_index(),
        x='invoice_date', y='total_price',
        title="Sales Trend Over Time", labels={'total_price': 'Total Sales', 'invoice_date': 'Date'}
    )
    st.plotly_chart(sales_trend_fig)
    st.markdown(
    """
    This line graph shows total sales revenue over time, highlighting patterns like peaks, dips, and overall trends, helping identify sales performance and seasonal variations.
    """
    )

    # Top 10 Customers by Spending as a List
    top_customers = (
        filtered_data.groupby('customer_id')['total_price']
        .sum()
        .nlargest(10)
        .reset_index()
    )

    # Format total_price to .2lf
    top_customers['total_price'] = top_customers['total_price'].apply(lambda x: f"{x:.2f}")

    # Convert DataFrame to HTML with left-aligned customer_id
    top_customers_html = top_customers.to_html(index=False, justify='left')

    st.subheader("10 Top Spender Customers")
    st.markdown(top_customers_html, unsafe_allow_html=True)
    st.markdown(
    """
    This table lists the top 10 customers ranked by their total spending, providing insights into high-value customers and their contribution to revenue.
    """
    )

    # Calculate Customer Lifetime Value (CLV) over time
    clv_over_time = filtered_data.groupby(filtered_data['invoice_date'].apply(lambda x: pd.to_datetime(x).to_period('M'))).apply(
        lambda x: x['total_price'].groupby(x['customer_id']).sum().mean()
    ).reset_index(name='CLV')

    # Calculate Customer Retention Rate over time
    first_purchase = filtered_data.groupby('customer_id')['invoice_date'].min().reset_index()
    first_purchase['year_month'] = first_purchase['invoice_date'].apply(lambda x: pd.to_datetime(x).to_period('M'))
    retention_rate_over_time = first_purchase.groupby('year_month').apply(
        lambda x: (x['customer_id'].nunique() / first_purchase['customer_id'].nunique()) * 100
    ).reset_index(name='Retention Rate')

    # Merge CLV and Retention Rate data
    metrics_over_time = pd.merge(clv_over_time, retention_rate_over_time, left_on='invoice_date', right_on='year_month')

    # Dual-axis line chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=metrics_over_time['invoice_date'].astype(str), y=metrics_over_time['CLV'], name="CLV", mode='lines'),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=metrics_over_time['invoice_date'].astype(str), y=metrics_over_time['Retention Rate'], name="Retention Rate", mode='lines'),
        secondary_y=True,
    )
    fig.update_layout(title_text="CLV and Customer Retention Rate Over Time")
    fig.update_xaxes(title_text="Time (Month-Year)")
    fig.update_yaxes(title_text="Customer Lifetime Value (CLV)", secondary_y=False)
    fig.update_yaxes(title_text="Customer Retention Rate (%)", secondary_y=True)
    st.plotly_chart(fig)

    st.markdown(
    """
    This dual-axis chart displays how Customer Lifetime Value (CLV) and Retention Rate evolve over time. CLV represents the average revenue generated by a customer over their lifetime, while Retention Rate reflects the percentage of customers who continue to engage with the business. The chart provides a clear view of spending patterns and customer loyalty, helping to identify periods of growth or decline and evaluate the effectiveness of retention strategies.
    """
    )

elif selected_module == "Data Mining":
    # Visualizations from E_webApp.py and D_DataMining.py
    analytics = RetailAnalytics()
    df = analytics.load_data()

    st.subheader("Customer Segmentation Analysis")
    
    # RFM Analysis
    rfm_data, _ = analytics.perform_customer_segmentation(df)
    
    # Visualize segments
    fig_segments = px.scatter(
        rfm_data,
        x='total_price',
        y='transaction_count',
        color='segment',
        title='Customer Segments (RFM Analysis)'
    )
    st.plotly_chart(fig_segments)
    
    # Sales Prediction
    st.subheader("Sales Prediction")
    
    prediction_results = analytics.perform_sales_prediction(df)
    y_test = prediction_results['actual']
    y_pred = prediction_results['predicted']
    
    # Plot actual vs predicted
    fig_prediction = go.Figure()
    fig_prediction.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='markers', name='Actual Sales'))
    fig_prediction.add_trace(go.Scatter(x=y_test.index, y=y_pred, mode='lines', name='Predicted Sales'))
    fig_prediction.update_layout(title='Sales Prediction: Actual vs Predicted')
    
    st.plotly_chart(fig_prediction)
    
    # Model performance metrics
    r2 = prediction_results['r2']
    rmse = np.sqrt(prediction_results['mse'])
    
    st.write(f"Model Performance:")
    st.write(f"R² Score: {r2:.3f}")
    st.write(f"RMSE: ${rmse:,.2f}")

