import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv('data/diabetes.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, 'model/model.pkl')
print(" Diabetes disease model trained.")
