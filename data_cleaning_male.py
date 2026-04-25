import pandas as pd

# Load dataset
data = pd.read_csv("Sperm_Analysis_Clinical_Study_Data.csv")

# Select important columns
data = data[['Age','Volume','Period of Abstinence',
             'Total Spermatozoa Count','Active Motile',
             'Weakly Motile','Non Motile',
             'Morphology Normal','Morphology Abnormal',
             'Comment']]

# Clean numeric values (remove text like ml, days etc.)
data['Volume'] = data['Volume'].astype(str).str.extract('(\d+\.?\d*)').astype(float)
data['Period of Abstinence'] = data['Period of Abstinence'].astype(str).str.extract('(\d+)').astype(float)
data['Total Spermatozoa Count'] = data['Total Spermatozoa Count'].astype(str).str.extract('(\d+\.?\d*)').astype(float)
data['Active Motile'] = data['Active Motile'].astype(str).str.extract('(\d+\.?\d*)').astype(float)
data['Weakly Motile'] = data['Weakly Motile'].astype(str).str.extract('(\d+\.?\d*)').astype(float)
data['Non Motile'] = data['Non Motile'].astype(str).str.extract('(\d+\.?\d*)').astype(float)
data['Morphology Normal'] = data['Morphology Normal'].astype(str).str.extract('(\d+\.?\d*)').astype(float)
data['Morphology Abnormal'] = data['Morphology Abnormal'].astype(str).str.extract('(\d+\.?\d*)').astype(float)

# Show cleaned data
print(data.head())

# Show dataset info
print(data.info())

