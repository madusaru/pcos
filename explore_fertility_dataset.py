import pandas as pd

# Load the dataset
df = pd.read_csv("female_fertility_dataset_realistic.csv")

# Quick look
print(df.head())
print(df.describe())
print(df['Fertility_label'].value_counts())

