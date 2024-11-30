import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime, timedelta

class RetailAnalyticsApp:
    def __init__(self):
        try:
            self.engine = create_engine('postgresql://postgres:postgres@localhost/data_warehouse')
            self.load_data()
        except Exception as e:
            st.error(f"Database connection error: {str(e)}")
            self.data = None

    def load_data(self):
        """Load data from PostgreSQL database"""
        try:
            self.data = pd.read_sql("""
                SELECT 
                    o.order_id, o.order_date, o.ship_date, o.ship_mode,
                    c.customer_id, c.customer_name, c.segment, c.region,
                    p.product_id, p.product_name, p.category, p.sub_category,
                    s.sales
                FROM sales s
                JOIN orders o ON s.order_id = o.order_id
                JOIN customers c ON s.customer_id = c.customer_id
                JOIN products p ON s.product_id = p.product_id
            """, self.engine)
            self.data['order_date'] = pd.to_datetime(self.data['order_date'])
        except Exception as e:
            st.error(f"Data loading error: {str(e)}")
            self.data = None

    def run(self):
        st.set_page_config(page_title="Retail Analytics Dashboard", layout="wide")
        
        if self.data is None:
            st.error("Unable to load data. Please check your database connection.")
            return

        # Sidebar
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Select Page", 
            ["Dashboard", "Customer Segmentation", "Sales Prediction", "Data Explorer"])

        if page == "Dashboard":
            self.show_dashboard()
        elif page == "Customer Segmentation":
            self.show_customer_segmentation()
        elif page == "Sales Prediction":
            self.show_sales_prediction()
        elif page == "Data Explorer":
            self.show_data_explorer()

    def show_dashboard(self):
        st.title("Retail Analytics Dashboard")

        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sales", f"${self.data['sales'].sum():,.2f}")
        with col2:
            st.metric("Total Orders", f"{self.data['order_id'].nunique():,}")
        with col3:
            st.metric("Total Customers", f"{self.data['customer_id'].nunique():,}")
        with col4:
            avg_order = self.data['sales'].sum() / self.data['order_id'].nunique()
            st.metric("Average Order Value", f"${avg_order:,.2f}")

        # Sales Trends
        st.subheader("Sales Trends")
        daily_sales = self.data.groupby('order_date')['sales'].sum().reset_index()
        fig = px.line(daily_sales, x='order_date', y='sales', 
                     title='Daily Sales Trend')
        st.plotly_chart(fig, use_container_width=True)

        # Regional Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sales by Region")
            region_sales = self.data.groupby('region')['sales'].sum().reset_index()
            fig = px.pie(region_sales, values='sales', names='region')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Sales by Category")
            category_sales = self.data.groupby('category')['sales'].sum().reset_index()
            fig = px.bar(category_sales, x='category', y='sales')
            st.plotly_chart(fig, use_container_width=True)

    def show_customer_segmentation(self):
        st.title("Customer Segmentation Analysis")

        try:
            # Prepare customer metrics
            customer_metrics = self.data.groupby('customer_id').agg({
                'sales': ['sum', 'mean', 'count'],
                'order_id': 'nunique'
            }).reset_index()
            customer_metrics.columns = ['customer_id', 'total_sales', 'avg_sales', 
                                      'transaction_count', 'order_count']

            # Perform clustering
            n_clusters = st.slider("Select number of clusters", 2, 6, 3)
            
            features = ['total_sales', 'transaction_count', 'order_count']
            X = customer_metrics[features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            customer_metrics['Segment'] = kmeans.fit_predict(X_scaled)

            # Visualize segments
            fig = px.scatter(customer_metrics, x='total_sales', y='transaction_count',
                            color='Segment', title='Customer Segments')
            st.plotly_chart(fig, use_container_width=True)

            # Segment Analysis
            st.subheader("Segment Analysis")
            segment_analysis = customer_metrics.groupby('Segment').agg({
                'customer_id': 'count',
                'total_sales': 'mean',
                'transaction_count': 'mean'
            }).round(2)
            
            # Display segment analysis as a simple table
            st.write(segment_analysis)

        except Exception as e:
            st.error(f"Error in customer segmentation: {str(e)}")

    def show_sales_prediction(self):
        st.title("Sales Prediction")

        try:
            # Prepare time series data
            monthly_sales = self.data.groupby(self.data['order_date'].dt.to_period('M'))['sales'].sum()
            X = np.arange(len(monthly_sales)).reshape(-1, 1)
            y = monthly_sales.values

            # Train model
            model = LinearRegression()
            model.fit(X, y)

            # Make future predictions
            months_to_predict = st.slider("Number of months to predict", 1, 12, 6)
            future_X = np.arange(len(X), len(X) + months_to_predict).reshape(-1, 1)
            future_predictions = model.predict(future_X)

            # Plot results
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly_sales.index.astype(str), 
                                    y=monthly_sales.values,
                                    name='Historical Sales'))
            fig.add_trace(go.Scatter(x=[str(monthly_sales.index[-1] + i + 1) 
                                       for i in range(months_to_predict)],
                                    y=future_predictions,
                                    name='Predicted Sales'))
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error in sales prediction: {str(e)}")

    def show_data_explorer(self):
        st.title("Data Explorer")

        try:
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_region = st.multiselect("Select Region", 
                                               options=self.data['region'].unique())
            with col2:
                selected_category = st.multiselect("Select Category", 
                                                 options=self.data['category'].unique())
            with col3:
                date_range = st.date_input("Select Date Range",
                                         [self.data['order_date'].min(),
                                          self.data['order_date'].max()])

            # Filter data
            filtered_data = self.data.copy()
            if selected_region:
                filtered_data = filtered_data[filtered_data['region'].isin(selected_region)]
            if selected_category:
                filtered_data = filtered_data[filtered_data['category'].isin(selected_category)]
            filtered_data = filtered_data[
                (filtered_data['order_date'].dt.date >= date_range[0]) &
                (filtered_data['order_date'].dt.date <= date_range[1])
            ]

            # Show filtered data
            st.write("Filtered Data Preview:")
            st.write(filtered_data.head(1000))  # Show only first 1000 rows

            # Download button
            if st.button("Download Filtered Data"):
                csv = filtered_data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="filtered_data.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error in data explorer: {str(e)}")

def main():
    app = RetailAnalyticsApp()
    app.run()

if __name__ == "__main__":
    main()
