import pandas as pd
import joblib

# Load model
model = joblib.load('model/model.pkl')

# Sample patient
sample = pd.DataFrame({
    'Pregnancies': [10],
    'Glucose': [98],
    'BloodPressure': [50],
    'SkinThickness': [46],
    'Insulin': [27],
    'BMI': [30],
    'DiabetesPedigreeFunction': [0.672],
    'Age': [54],

})

# Predict
prediction = model.predict(sample)[0]
risk = "Diabetic" if prediction == 1 else "No Diabetic"
print(f" Prediction: {prediction} ({Diabetic})")
