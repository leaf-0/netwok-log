import pandas as pd
import os

# Path to the CICIDS datasets
cicids_dataset_path = '/home/kaderavan/Desktop/base_log/archive/'

# Load all CSV files from the archive folder
dataframes = []
for file_name in os.listdir(cicids_dataset_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(cicids_dataset_path, file_name)
        print(f"Loading file: {file_path}")
        df = pd.read_csv(file_path)
        dataframes.append(df)

# Combine all dataframes into one
if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
    print("All datasets loaded successfully.")
else:
    print("No CSV files found in the specified directory.")

# Perform any data processing or analysis here
print(combined_df.head())

# Save combined dataset to a CSV file for further use
combined_df.to_csv('combined_cicids_dataset.csv', index=False)
