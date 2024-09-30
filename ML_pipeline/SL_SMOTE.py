#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, RocCurveDisplay, precision_recall_curve
from sklearn.model_selection import cross_val_score, train_test_split
import pickle
import os
from imblearn.over_sampling import SMOTE

# Step 1: Load the dataset from a Parquet file
print("Loading data...")
df = pd.read_parquet('/home/ndo/vardict_ML/processed_data.parquet')

df['VD'] = df['VD'].fillna(df['VD'].mean())

# Check if any NaN values remain
print("Remaining NaN values in the DataFrame after filling:")
print(df.isnull().sum())

# Step 2: Data Preparation
X = df.drop(columns=['labels'])  # Features
y = df['labels']  # Labels

# Dictionary to store model results
model_results = {}

# Function to train, evaluate, and store results of each model
def evaluate_model(model, model_name):
    print(f"Evaluating {model_name}...")
    
    # 3-fold Cross-Validation
    cv_scores = cross_val_score(model, X, y, cv=3)
    print(f"3-fold Cross-Validation Scores for {model_name}: {cv_scores}")
    print(f"Mean: {np.mean(cv_scores)}, Standard deviation: {np.std(cv_scores)}")
    
    # Train/Test Split Validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Applying SMOTE
    sm = SMOTE(random_state=27)
    X_train, y_train = sm.fit_resample(X_train, y_train)    
    
    # Train the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate Precision and Recall
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred)
    
    print(f"Validation using train/test split for {model_name}")
    print(f"Precision: {precision}, Recall: {recall}")
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    # Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, y_pred)
    plt.plot(rec, prec, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.show()
    
    # Store results
    model_results[model_name] = {
        'model': model,
        'cv_scores': cv_scores,
        'cv_mean': np.mean(cv_scores),
        'cv_std': np.std(cv_scores),
        'precision': precision,
        'recall': recall,
        'confusion_matrix': conf_matrix
    }

# Step 3: Evaluate Multiple Models
# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
evaluate_model(rf_model, "RandomForestClassifier")

# Logistic Regression Classifier
lr_model = LogisticRegression(random_state=42)
evaluate_model(lr_model, "LogisticRegression")

# Stochastic Gradient Descent Classifier
sgd_model = SGDClassifier(random_state=42)
evaluate_model(sgd_model, "SGDClassifier")

# Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
evaluate_model(dt_model, "DecisionTreeClassifier")

# Step 4: Plot ROC Curve Comparison
print("Plotting ROC Curve Comparison...")
fig, ax = plt.subplots()

# ROC Curve for Logistic Regression
RocCurveDisplay.from_estimator(lr_model, X, y, ax=ax, name='Logistic Regression')
# ROC Curve for Random Forest
RocCurveDisplay.from_estimator(rf_model, X, y, ax=ax, name='Random Forest')
# ROC Curve for SGD Classifier
RocCurveDisplay.from_estimator(sgd_model, X, y, ax=ax, name='SGD Classifier')
# ROC Curve for Decision Tree Classifier
RocCurveDisplay.from_estimator(dt_model, X, y, ax=ax, name='Decision Tree Classifier')

plt.title("ROC Curve Comparison")
plt.show()

# Step 5: Save model results to a file for later retrieval
print("Saving model results for later retrieval...")
file_path = '/home/ndo/vardict_ML/models_results.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(model_results, file)

print(f"Model results saved successfully to {file_path}!")

# Step 6: Function to load saved model results
def load_model_results(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    else:
        print(f"File {file_path} not found.")
        return None

# Uncomment the following to test loading the saved results
# loaded_results = load_model_results(file_path)
# print(loaded_results)
