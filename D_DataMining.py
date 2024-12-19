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
        """Load data from PostgreSQL database."""
        query = """
        SELECT 
            sf.customer_id, 
            SUM(sf.total_price) AS total_price,
            AVG(sf.total_price) AS avg_sales,
            COUNT(sf.invoice_no) AS transaction_count,
            MIN(td.invoice_date) AS first_purchase,
            MAX(td.invoice_date) AS last_purchase,
            EXTRACT(MONTH FROM MIN(td.invoice_date)) AS month,
            EXTRACT(YEAR FROM MIN(td.invoice_date)) AS year
        FROM sales_fact sf
        JOIN time_dimension td ON sf.time_id = td.time_id
        GROUP BY sf.customer_id
        """
        df = pd.read_sql(query, self.engine)
        
        # Calculate purchase frequency and handle division issues
        df['customer_lifetime'] = (
            pd.to_datetime(df['last_purchase']) - 
            pd.to_datetime(df['first_purchase'])
        ).dt.days
        df['customer_lifetime'] = df['customer_lifetime'].replace(0, 1)  # Avoid zero division
        df['purchase_frequency'] = df['transaction_count'] / df['customer_lifetime']
        
        return df

    def optimal_clusters(self, X_scaled):
        """Determine the optimal number of clusters using the Elbow Method."""
        inertia = []
        for k in range(1, 10):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertia.append(kmeans.inertia_)

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, 10), inertia, marker='o')
        plt.title("Elbow Method for Optimal Clusters")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Inertia")
        plt.show()

    def perform_customer_segmentation(self, df, n_clusters=3):
        """Perform customer segmentation using K-means clustering."""
        # Select features for clustering
        features = ['total_price', 'transaction_count']
        X = df[features]

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply K-Means clustering
        self.optimal_clusters(X_scaled)  # Visualize Elbow Method
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['segment'] = kmeans.fit_predict(X_scaled)

        # Visualize clusters
        sns.scatterplot(data=df, x='total_price', y='transaction_count', hue='segment', palette='viridis')
        plt.title('Customer Segmentation Using K-Means Clustering')
        plt.xlabel('Total Sales')
        plt.ylabel('Transaction Count')
        plt.show()

        return df, kmeans

    def perform_sales_prediction(self, df):
        """Perform sales prediction using Linear Regression."""
        # Combine month and year into a single feature
        df['time'] = df['year'] + (df['month'] - 1) / 12

        # Features and target
        X = df[['avg_sales', 'transaction_count', 'purchase_frequency', 'time']]
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

        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R-squared: {r2:.2f}")

        # Visualize Actual vs Predicted
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
        plt.xlabel('Actual Total Sales')
        plt.ylabel('Predicted Total Sales')
        plt.title('Actual vs. Predicted Sales')
        plt.show()

        return {'model': model, 'mse': mse, 'r2': r2, 'y_test': y_test, 'y_pred': y_pred}

    def generate_report(self, segmentation_results, prediction_results):
        """Generate a summary report of findings."""
        report = f"""
        ### Customer Segmentation Insights:
        - Total clusters: {segmentation_results['segment'].nunique()}
        - Cluster distribution:
        {segmentation_results['segment'].value_counts().to_string()}
        
        ### Sales Prediction Insights:
        - Mean Squared Error (MSE): {prediction_results['mse']:.2f}
        - R-squared: {prediction_results['r2']:.2f}
        """
        print(report)

def main():
    analytics = RetailAnalytics()
    df = analytics.load_data()
    segmented_df, kmeans = analytics.perform_customer_segmentation(df)
    prediction_results = analytics.perform_sales_prediction(df)
    analytics.generate_report(segmented_df, prediction_results)

if __name__ == "__main__":
    main()
