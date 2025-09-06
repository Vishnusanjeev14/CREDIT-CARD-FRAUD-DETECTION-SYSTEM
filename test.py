import joblib
import pandas as pd

# Load the trained model
model = joblib.load('credit_fraud_xgb_model.pkl')

# Example new transaction (replace with actual feature values)
new_transaction = {
    'Time': 0,
    'V1': -1.359807,
    'V2': -0.072781,
    'V3': 2.536347,
    'V4': 1.378155,
    'V5': -0.338321,
    'V6': 0.0,
    'V7': 0.0,
    'V8': 0.0,
    'V9': 0.0,
    'V10': 0.0,
    'V11': 0.0,
    'V12': 0.0,
    'V13': 0.0,
    'V14': 0.0,
    'V15': 0.0,
    'V16': 0.0,
    'V17': 0.0,
    'V18': 0.0,
    'V19': 0.0,
    'V20': 0.0,
    'V21': 0.0,
    'V22': 0.0,
    'V23': 0.0,
    'V24': 0.0,
    'V25': 0.0,
    'V26': 0.0,
    'V27': 0.0,
    'V28': 0.0,
    'Amount': 100.0,
    'Transaction_Hour': 0
}

# Convert to DataFrame
df = pd.DataFrame([new_transaction])

# Predict
prediction = model.predict(df)
print("Fraud Prediction:", "Fraud" if prediction[0] == 1 else "Not Fraud")
