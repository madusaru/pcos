import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

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

# Fill missing values
data = data.fillna(data.mean(numeric_only=True))

# Show cleaned data
print(data.head())

# Show dataset info
print(data.info())

# Features (X) and target (y)
X = data.drop("Comment", axis=1)
y = data["Comment"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train machine learning model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Model accuracy
accuracy = accuracy_score(y_test, predictions)
print("\nModel Accuracy:", accuracy)

# Save trained model
with open("male_infertility_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully as male_infertility_model.pkl")

