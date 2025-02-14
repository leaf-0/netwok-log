import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('/home/kaderavan/Desktop/base_log/combined_cicids_dataset.csv')

# Remove leading/trailing spaces from column names
df.columns = df.columns.str.strip()

print(df.shape)
print("Data Loaded Successfully!")

# Basic Information about the dataset
print("\nDataset Info:")
print(df.info())
print("\nBasic Statistics:")
print(df.describe())

# Handling missing values: Filling with 0 for simplicity
df.fillna(0, inplace=True)

# Check for remaining missing values
print("\nMissing Values After Filling:")
print(df.isnull().sum())

# Create new features from 'Timestamp' if available
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['hour'] = df['Timestamp'].dt.hour
    df['day_of_week'] = df['Timestamp'].dt.dayofweek
    df['minute'] = df['Timestamp'].dt.minute
else:
    # If 'Timestamp' doesn't exist, skip creating time features
    print("\nNo 'Timestamp' column found, skipping time-based feature creation.")

# Feature engineering: adding new features
df['packet_ratio'] = df['Flow Duration'] / (df['Total Fwd Packets'] + 1)  # Example feature creation

# Visualize the distribution of the 'Flow Duration' feature
plt.figure(figsize=(10, 6))
sns.histplot(df['Flow Duration'], bins=50, kde=True)
plt.title('Distribution of Flow Duration')
plt.xlabel('Flow Duration')
plt.ylabel('Frequency')
plt.show()

# Select numeric features that exist in the dataset
numeric_columns = ['Flow Duration', 'Total Fwd Packets', 'packet_ratio']

# If time-based features exist, add them to numeric columns
if 'hour' in df.columns and 'minute' in df.columns and 'day_of_week' in df.columns:
    numeric_columns += ['hour', 'minute', 'day_of_week']

# Visualize correlation among numeric features
numeric_df = df[numeric_columns]
corr_matrix = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numeric Features')
plt.show()

# Feature Scaling: Scaling numeric features to normalize them
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Anomaly detection using Isolation Forest for zero-day attacks
model = IsolationForest(contamination=0.01, random_state=42)
df['anomaly'] = model.fit_predict(df[numeric_columns])

# Replace -1 with 1 (anomalies) and 1 with 0 (normal traffic)
df['anomaly'] = df['anomaly'].apply(lambda x: 1 if x == -1 else 0)

# Visualize anomalies
plt.figure(figsize=(10, 6))
sns.countplot(x='anomaly', data=df)
plt.title('Anomaly Count (0 = Normal, 1 = Anomaly)')
plt.show()

# Splitting data into training and testing for further modeling
X = df.drop('anomaly', axis=1)
y = df['anomaly']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Summary of final features
print("\nFinal Features for Model Training:")
print(X.columns)

# Save the cleaned and processed dataset for model training
df.to_csv('/home/kaderavan/Desktop/base_log/prepared_network_data.csv', index=False)
print("\nPreprocessed data saved as 'prepared_network_data.csv'. Ready for model training.")
