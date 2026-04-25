import pandas as pd

data = pd.read_csv("Sperm_Analysis_Clinical_Study_Data.csv")

print(data.head())      # shows first 5 rows
print(data.columns)     # shows column names
print(data.shape)       # shows rows and columns

