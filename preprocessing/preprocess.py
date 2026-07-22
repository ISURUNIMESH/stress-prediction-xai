import pandas as pd
import os

# ---------------------------------------
# File Paths
# ---------------------------------------

input_file = "dataset/raw/Stress Prediction Form.csv"
output_folder = "dataset/processed"
output_file = os.path.join(output_folder, "cleaned_stress_dataset.csv")

# ---------------------------------------
# Create Output Folder (if it doesn't exist)
# ---------------------------------------

os.makedirs(output_folder, exist_ok=True)

# ---------------------------------------
# Load Dataset
# ---------------------------------------

try:
    df = pd.read_csv(input_file)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: File not found -> {input_file}")
    exit()

print(f"Original Dataset Shape: {df.shape}")

# ---------------------------------------
# Remove Duplicate Records
# ---------------------------------------

duplicate_count = df.duplicated().sum()
print(f"Duplicate Records: {duplicate_count}")

df = df.drop_duplicates()

# ---------------------------------------
# Handle Missing Values
# ---------------------------------------

missing_values = df.isnull().sum().sum()
print(f"Missing Values: {missing_values}")

df = df.dropna()

# ---------------------------------------
# Reset Index
# ---------------------------------------

df.reset_index(drop=True, inplace=True)

# ---------------------------------------
# Dataset Information
# ---------------------------------------

print("\nDataset Information")
print(df.info())

# ---------------------------------------
# Save Clean Dataset
# ---------------------------------------

df.to_csv(output_file, index=False)

print("\n----------------------------------------")
print("Preprocessing Completed Successfully!")
print("----------------------------------------")
print(f"Clean Dataset Shape : {df.shape}")
print(f"Saved File          : {output_file}")