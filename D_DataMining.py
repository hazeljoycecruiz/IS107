import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

class RetailAnalytics:
    def __init__(self):
        # Initialize the database connection
        self.engine = create_engine('postgresql://postgres:postgres@localhost/data_warehouse')

    def load_data(self):
        """Load data from PostgreSQL database"""
        query = """
        SELECT 
            sf.customer_id, 
            SUM(sf.total_price) as total_price,
            AVG(sf.total_price) as avg_sales,  -- Change this line
            COUNT(sf.invoice_no) as transaction_count,
            MIN(td.invoice_date) as first_purchase,
            MAX(td.invoice_date) as last_purchase,
            EXTRACT(MONTH FROM MIN(td.invoice_date)) as month,
            EXTRACT(YEAR FROM MIN(td.invoice_date)) as year
        FROM sales_fact sf
        JOIN time_dimension td ON sf.time_id = td.time_id
        GROUP BY sf.customer_id
        """
        df = pd.read_sql(query, self.engine)
        
        # Calculate purchase frequency
        df['customer_lifetime'] = (
            pd.to_datetime(df['last_purchase']) - 
            pd.to_datetime(df['first_purchase'])
        ).dt.days
        df['purchase_frequency'] = df['transaction_count'] / df['customer_lifetime'].replace(0, 1)
        
        return df


    def perform_customer_segmentation(self, df, n_clusters=3):
        """Perform customer segmentation using K-means clustering."""
        # Select features for clustering
        features = ['total_price', 'transaction_count']
        X = df[features]

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['segment'] = kmeans.fit_predict(X_scaled)

        # Visualize clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='total_price', y='transaction_count', hue='segment', palette='viridis')
        plt.title('Customer Segmentation Using K-Means Clustering')
        plt.xlabel('Total Sales')
        plt.ylabel('Transaction Count')
        plt.show()

        return df, kmeans

    def perform_sales_prediction(self, df):
        """Perform sales prediction using Linear Regression."""
        # Features and target
        X = df[['avg_sales', 'transaction_count', 'purchase_frequency', 'month', 'year']]
        y = df['total_price']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Performance metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print("\nLinear Regression Performance:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R-squared: {r2:.2f}")

        # Plot Actual vs. Predicted Sales
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.xlabel('Actual Total Sales')
        plt.ylabel('Predicted Total Sales')
        plt.title('Predictive Analysis using Linear Regression')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label='Ideal: y = x')
        plt.text(0.05, 0.95, f'y = -9.50 + (1.55 * quantity) + (4.45 * unitprice)\nMSE: {mse:.2f}',
                 transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
        plt.legend()
        plt.show()

        return {
            'model': model,
            'mse': mse,
            'r2': r2,
            'predicted': y_pred,
            'actual': y_test
        }

def main():
    # Initialize Retail Analytics
    analytics = RetailAnalytics()

    # Load data
    print("Loading data...")
    df = analytics.load_data()

    # Perform customer segmentation
    print("Performing customer segmentation...")
    segmentation_results, kmeans = analytics.perform_customer_segmentation(df)

    # Perform predictive analysis
    print("Performing sales prediction...")
    prediction_results = analytics.perform_sales_prediction(df)

if __name__ == "__main__":
    main()

