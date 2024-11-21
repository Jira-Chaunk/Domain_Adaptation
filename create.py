import pandas as pd
import numpy as np

# Simulating the source domain dataset
np.random.seed(42)
source_data = pd.DataFrame({
    "SST": np.random.uniform(26, 30, 1000),  # Sea Surface Temperature
    "WindSpeed": np.random.uniform(10, 50, 1000),  # Wind Speed (m/s)
    "Pressure": np.random.uniform(950, 1005, 1000),  # Atmospheric Pressure (hPa)
    "Humidity": np.random.uniform(70, 100, 1000),  # Humidity (%)
    "target": np.random.choice([0, 1], 1000, p=[0.7, 0.3])  # Cyclone severity: 0=Low, 1=High
})

# Simulating the target domain dataset (Odisha)
target_data = pd.DataFrame({
    "SST": np.random.uniform(27, 29, 200),  # Slightly narrower range for SST
    "WindSpeed": np.random.uniform(20, 40, 200),
    "Pressure": np.random.uniform(960, 990, 200),
    "Humidity": np.random.uniform(75, 95, 200),
    "target": np.random.choice([0, 1], 200, p=[0.8, 0.2])
})

# Save the data to CSV files (optional)
source_data.to_csv("source.csv", index=False)
target_data.to_csv("target.csv", index=False)

print("Source Data Example:\n", source_data.head())
print("\nTarget Data Example:\n", target_data.head())
