import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv('diabetes')
X = df.drop('target', axis=1)
y = df['target']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, 'model/model.pkl')
print(" Heart disease model trained.")
