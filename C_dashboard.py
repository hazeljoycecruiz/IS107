# D_DataMining.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from A_etl import load_data, clean_data

# Load and clean data
raw_data = load_data('C:/Users/USER/Downloads/IS107/train.csv')
cleaned_data = clean_data(raw_data)

# Check the columns in the cleaned data
print("Columns in cleaned data:", cleaned_data.columns)

# Customer Segmentation using KMeans Clustering
def customer_segmentation(data):
    """
    Perform customer segmentation using KMeans clustering.

    Parameters:
    - data: DataFrame, the cleaned data to be used for clustering.

    Returns:
    - DataFrame with an additional 'Cluster' column indicating the cluster assignment.
    """
    print("Performing customer segmentation using KMeans clustering...")

    # Adjust the feature selection based on available columns
    # Use 'Sales' and any other available numeric columns
    feature_columns = ['Sales']
    available_features = [col for col in feature_columns if col in data.columns]
    
    if len(available_features) < 1:
        print("Available columns for clustering:", available_features)
        raise ValueError("Not enough features available for clustering. Check your dataset.")

    # Select features for clustering
    features = data[available_features]
    
    # Normalize the data
    features_normalized = (features - features.mean()) / features.std()
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['Cluster'] = kmeans.fit_predict(features_normalized)
    
    # Visualize the clusters
    sns.pairplot(data, hue='Cluster', vars=available_features)
    plt.title('Customer Segmentation')
    plt.show()

    print("Customer segmentation completed. Clusters have been visualized.")

# Predictive Analysis using Naive Bayes
def predictive_analysis(data):
    """
    Perform predictive analysis using Naive Bayes.

    Parameters:
    - data: DataFrame, the cleaned data to be used for prediction.

    Returns:
    - None
    """
    print("Performing predictive analysis using Naive Bayes...")

    # Prepare the data
    data['Order Date'] = pd.to_datetime(data['Order Date'])
    data['Year'] = data['Order Date'].dt.year
    # Use 'Sales' and 'Year' for prediction
    features = data[['Sales', 'Year']]
    target = data['Segment']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    
    # Train Naive Bayes model
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Classification Report:\n", classification_report(y_test, predictions))

    print("Predictive analysis completed. Results have been printed.")

# Main function
if __name__ == "__main__":
    try:
        # Perform customer segmentation
        customer_segmentation(cleaned_data)
    except ValueError as e:
        print(e)
    
    # Perform predictive analysis
    predictive_analysis(cleaned_data)

    # Insights and Report
    print("\nReport:")
    print("1. Customer Segmentation: Customers are segmented into 3 clusters based on available features.")
    print("   This helps in identifying different customer groups for targeted marketing strategies.")
    print("2. Predictive Analysis: The Naive Bayes model predicts customer segments with an accuracy score.")
    print("   This can be valuable for forecasting future sales and understanding customer behavior.")
