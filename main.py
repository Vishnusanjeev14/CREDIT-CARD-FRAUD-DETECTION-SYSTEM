

# Step 1: Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import shap 
from flask import Flask, request, jsonify

# Step 2: Load the dataset
data = pd.read_csv("creditcard.csv")

# Step 3: Feature Engineering
data['Transaction_Hour'] = (data['Time'] // 3600) % 24

# Optional: drop 'Time' if not needed
# data = data.drop(['Time'], axis=1)

# Step 4: Data exploration
print(data.head())
print(data.describe())

# Step 5: Class distribution
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlier_fraction = len(fraud) / float(len(valid))
print('Outlier fraction:', outlier_fraction)
print('Fraudulent Transactions:', len(fraud))
print('Valid Transactions:', len(valid))

# Step 6: Visualization
plt.figure(figsize=(12, 9))
sns.heatmap(data.corr(), vmax=.8, square=True)
plt.show()

# Step 7: Prepare data
X = data.drop(['Class'], axis=1)
Y = data['Class']

# Step 8: Balance dataset using SMOTE
sm = SMOTE(random_state=42)
X_resampled, Y_resampled = sm.fit_resample(X, Y)

# Step 9: Train-test split
x_train, x_test, y_train, y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2, random_state=42)

# Step 10: Model training with hyperparameter tuning (XGBoost)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1]
}

grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='f1', cv=3, verbose=2)
grid_search.fit(x_train, y_train)

best_model = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)

# Step 11: Evaluate model
y_pred = best_model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Confusion matrix plot
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()

# Step 12: Cross-validation
cv_scores = cross_val_score(best_model, X_resampled, Y_resampled, cv=5, scoring='f1')
print("Cross-validated F1 scores:", cv_scores)
print("Mean F1 Score:", cv_scores.mean())

# Step 13: Save the trained model
joblib.dump(best_model, 'credit_fraud_xgb_model.pkl')

# Step 14: Explain model predictions with SHAP
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_resampled)
shap.summary_plot(shap_values, X_resampled)

