# import pandas as pd
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load and prepare data
# def load_data():
#     df = pd.read_csv('cleaned_data_v3.csv')
#     df['Order Date'] = pd.to_datetime(df['Order Date'])
#     return df

# # Customer Segmentation Analysis
# def perform_customer_segmentation(df):
#     # Aggregate customer data - removed Quantity from aggregation
#     customer_metrics = df.groupby('Customer Name').agg({
#         'Sales': ['sum', 'mean', 'count'],
#         'Order ID': 'nunique'
#     }).reset_index()
    
#     # Flatten column names
#     customer_metrics.columns = ['Customer Name', 'Total Sales', 'Avg Sales', 
#                               'Transaction Count', 'Order Count']
    
#     # Prepare data for clustering
#     features = ['Total Sales', 'Transaction Count', 'Order Count']
#     X = customer_metrics[features]
    
#     # Standardize features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
    
#     # Perform K-means clustering
#     kmeans = KMeans(n_clusters=4, random_state=42)
#     customer_metrics['Segment'] = kmeans.fit_predict(X_scaled)
    
#     return customer_metrics

# # Sales Prediction Analysis
# def perform_sales_prediction(df):
#     # Prepare time series data
#     monthly_sales = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum()
#     monthly_sales.index = monthly_sales.index.astype(str)
    
#     # Create features for prediction
#     X = np.arange(len(monthly_sales)).reshape(-1, 1)
#     y = monthly_sales.values
    
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Train linear regression model
#     model = LinearRegression()
#     model.fit(X_train, y_train)
    
#     # Make predictions
#     y_pred = model.predict(X_test)
    
#     return {
#         'actual': y_test,
#         'predicted': y_pred,
#         'model': model,
#         'score': model.score(X_test, y_test)
#     }

# # Generate insights report
# def generate_insights_report(customer_segments, sales_prediction):
#     report = """
#     Data Mining Analysis Report
    
#     1. Customer Segmentation Analysis:
#     --------------------------------
#     - Number of customer segments: 4
#     - Segment distribution:
#     {}
    
#     2. Sales Prediction Analysis:
#     ---------------------------
#     - Model accuracy (R^2 score): {:.2f}
#     - The model suggests a {} trend in sales
    
#     3. Business Recommendations:
#     --------------------------
#     a) Customer Strategy:
#     - Focus on high-value customers in Segment 0
#     - Develop retention programs for Segment 1
#     - Create upgrade paths for Segments 2 and 3
    
#     b) Sales Strategy:
#     - Use predictive model for inventory planning
#     - Adjust marketing spend based on predicted trends
#     - Monitor seasonal patterns for better forecasting
    
#     4. Key Metrics:
#     -------------
#     - Average customer value: ${:.2f}
#     - Total number of customers: {}
#     """
    
#     segment_dist = customer_segments['Segment'].value_counts().to_dict()
#     trend = "positive" if sales_prediction['model'].coef_[0] > 0 else "negative"
#     avg_value = customer_segments['Total Sales'].mean()
#     total_customers = len(customer_segments)
    
#     return report.format(
#         segment_dist, 
#         sales_prediction['score'], 
#         trend,
#         avg_value,
#         total_customers
#     )

# def main():
#     # Load data
#     df = load_data()
    
#     # Perform analyses
#     customer_segments = perform_customer_segmentation(df)
#     sales_prediction = perform_sales_prediction(df)
    
#     # Generate report
#     report = generate_insights_report(customer_segments, sales_prediction)
    
#     # Save results
#     customer_segments.to_csv('customer_segments.csv', index=False)
#     print(report)
    
#     # Visualize results
#     plt.figure(figsize=(12, 5))
    
#     # Customer segments plot
#     plt.subplot(1, 2, 1)
#     sns.scatterplot(data=customer_segments, x='Total Sales', y='Order Count', 
#                    hue='Segment', palette='deep')
#     plt.title('Customer Segments')
    
#     # Sales prediction plot
#     plt.subplot(1, 2, 2)
#     plt.plot(sales_prediction['actual'], label='Actual')
#     plt.plot(sales_prediction['predicted'], label='Predicted')
#     plt.title('Sales Prediction')
#     plt.legend()
    
#     plt.tight_layout()
#     plt.savefig('data_mining_results.png')
#     plt.close()

# if __name__ == "__main__":
#     main()


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RetailAnalytics:
    def __init__(self):
        self.engine = create_engine('postgresql://postgres:postgres@localhost/data_warehouse')
        
    def load_data(self):
        """Load data from PostgreSQL database"""
        query = """
        SELECT 
            c.customer_id, c.customer_name, c.segment, c.region,
            o.order_date, o.ship_date,
            s.sales,
            p.category, p.sub_category
        FROM sales s
        JOIN customers c ON s.customer_id = c.customer_id
        JOIN orders o ON s.order_id = o.order_id
        JOIN products p ON s.product_id = p.product_id
        """
        return pd.read_sql(query, self.engine)

    def prepare_customer_features(self, df):
        """Prepare customer features for segmentation"""
        customer_features = df.groupby('customer_id').agg({
            'sales': ['sum', 'mean', 'count'],
            'order_date': ['min', 'max']
        }).reset_index()
        
        customer_features.columns = [
            'customer_id', 'total_sales', 'avg_sales', 
            'transaction_count', 'first_purchase', 'last_purchase'
        ]
        
        # Calculate customer lifetime value and purchase frequency
        customer_features['customer_lifetime'] = (
            pd.to_datetime(customer_features['last_purchase']) - 
            pd.to_datetime(customer_features['first_purchase'])
        ).dt.days
        customer_features['purchase_frequency'] = customer_features['transaction_count'] / \
                                                customer_features['customer_lifetime'].replace(0, 1)
        
        return customer_features

    def perform_customer_segmentation(self, df, n_clusters=4):
        """Perform customer segmentation using K-means clustering"""
        # Prepare features
        customer_features = self.prepare_customer_features(df)
        features_for_clustering = ['total_sales', 'avg_sales', 'transaction_count', 
                                 'purchase_frequency']
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(customer_features[features_for_clustering])
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        customer_features['segment'] = kmeans.fit_predict(features_scaled)
        
        # Add segment descriptions
        segment_descriptions = {
            0: 'High-Value Loyal Customers',
            1: 'Mid-Tier Regular Customers',
            2: 'Low-Value Occasional Customers',
            3: 'New Potential Customers'
        }
        customer_features['segment_description'] = customer_features['segment'].map(segment_descriptions)
        
        return customer_features, kmeans.cluster_centers_

    def prepare_time_series_data(self, df):
        """Prepare time series data for sales prediction"""
        df['order_date'] = pd.to_datetime(df['order_date'])
        monthly_sales = df.groupby(df['order_date'].dt.to_period('M'))['sales'].sum().reset_index()
        monthly_sales['order_date'] = monthly_sales['order_date'].astype(str)
        
        # Add time-based features
        monthly_sales['time_index'] = np.arange(len(monthly_sales))
        monthly_sales['month'] = pd.to_datetime(monthly_sales['order_date']).dt.month
        monthly_sales['year'] = pd.to_datetime(monthly_sales['order_date']).dt.year
        
        return monthly_sales

    def perform_sales_prediction(self, df):
        """Perform sales prediction using linear regression"""
        monthly_sales = self.prepare_time_series_data(df)
        
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
        
        return {
            'model': model,
            'actual': y_test,
            'predicted': y_pred,
            'r2_score': r2,
            'rmse': rmse,
            'monthly_sales': monthly_sales
        }

    def generate_visualizations(self, customer_segments, sales_prediction):
        """Generate visualizations for the analysis"""
        # Create a figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Customer Segmentation Plot
        plt.subplot(2, 2, 1)
        sns.scatterplot(data=customer_segments, 
                       x='total_sales', 
                       y='transaction_count',
                       hue='segment_description',
                       palette='deep')
        plt.title('Customer Segments')
        plt.xlabel('Total Sales')
        plt.ylabel('Transaction Count')
        plt.xticks(rotation=45)
        
        # Sales Prediction Plot
        plt.subplot(2, 2, 2)
        monthly_sales = sales_prediction['monthly_sales']
        plt.plot(monthly_sales['order_date'], monthly_sales['sales'], label='Actual Sales')
        plt.plot(monthly_sales['order_date'].iloc[-len(sales_prediction['predicted']):],
                sales_prediction['predicted'], 
                label='Predicted Sales',
                linestyle='--')
        plt.title('Sales Prediction')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.xticks(rotation=45)
        plt.legend()
        
        # Segment Distribution Plot
        plt.subplot(2, 2, 3)
        segment_dist = customer_segments['segment_description'].value_counts()
        sns.barplot(x=segment_dist.index, y=segment_dist.values)
        plt.title('Customer Segment Distribution')
        plt.xticks(rotation=45)
        
        # Sales Trend Plot
        plt.subplot(2, 2, 4)
        monthly_sales['sales'].plot(kind='line')
        plt.title('Monthly Sales Trend')
        plt.xlabel('Month')
        plt.ylabel('Sales')
        
        plt.tight_layout()
        plt.savefig('retail_analytics_results.png')
        plt.close()

    def generate_report(self, customer_segments, sales_prediction):
        """Generate a comprehensive analysis report"""
        report = f"""
        Retail Analytics Report
        =====================

        1. Customer Segmentation Analysis
        -------------------------------
        Total Customers: {len(customer_segments)}
        
        Segment Distribution:
        {customer_segments['segment_description'].value_counts().to_string()}
        
        Average Customer Metrics:
        - Average Total Sales: ${customer_segments['total_sales'].mean():,.2f}
        - Average Transaction Count: {customer_segments['transaction_count'].mean():.2f}
        - Average Purchase Frequency: {customer_segments['purchase_frequency'].mean():.2f}

        2. Sales Prediction Analysis
        -------------------------
        Model Performance:
        - R² Score: {sales_prediction['r2_score']:.3f}
        - RMSE: ${sales_prediction['rmse']:,.2f}
        
        3. Business Recommendations
        ------------------------
        Customer Strategy:
        1. Focus on high-value customers with personalized services
        2. Develop retention programs for mid-tier customers
        3. Create engagement campaigns for occasional customers
        4. Design acquisition strategies for potential customers
        
        Sales Strategy:
        1. Use predictive models for inventory planning
        2. Implement seasonal marketing campaigns
        3. Monitor and adjust pricing strategies
        4. Develop targeted promotions for each customer segment
        """
        
        with open('retail_analytics_report.txt', 'w') as f:
            f.write(report)
        
        return report

def main():
    try:
        # Initialize analytics
        analytics = RetailAnalytics()
        
        # Load data
        print("Loading data...")
        df = analytics.load_data()
        
        # Perform customer segmentation
        print("Performing customer segmentation...")
        customer_segments, _ = analytics.perform_customer_segmentation(df)
        
        # Perform sales prediction
        print("Performing sales prediction...")
        sales_prediction = analytics.perform_sales_prediction(df)
        
        # Generate visualizations
        print("Generating visualizations...")
        analytics.generate_visualizations(customer_segments, sales_prediction)
        
        # Generate report
        print("Generating report...")
        report = analytics.generate_report(customer_segments, sales_prediction)
        
        print("\nAnalysis completed successfully!")
        print("\nReport Summary:")
        print(report)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
