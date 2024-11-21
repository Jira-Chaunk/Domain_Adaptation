import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load your source and target datasets
# Example: 'source.csv' contains global data; 'target.csv' contains Odisha data
source_data = pd.read_csv("source.csv")
target_data = pd.read_csv("target.csv")

# Preprocess the data
def preprocess_data(data):
    features = data.drop(columns=["target"])
    labels = data["target"]
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return features, labels

source_features, source_labels = preprocess_data(source_data)
target_features, target_labels = preprocess_data(target_data)

# Split target data into training and testing
target_train_features, target_test_features, target_train_labels, target_test_labels = train_test_split(
    target_features, target_labels, test_size=0.2, random_state=42
)
