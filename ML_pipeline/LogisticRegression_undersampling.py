#!/usr/bin/env python3

# Re-train the Decision Tree Classifier
import pandas as pd
# from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve
import numpy as np


# Load the data again
df = pd.read_parquet('/home/ndo/vardict_ML/processed_data.parquet')
df['VD'] = df['VD'].fillna(df['VD'].mean())
X = df.drop(columns=['labels'])
y = df['labels']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#combine them back for resampling
train_data = pd.concat([X_train, y_train], axis=1)
# separate minority and majority classes
negative = train_data[train_data.labels==0]
positive = train_data[train_data.labels==1]
# downsampling minority
pos_downsampled = resample(positive,
 replace=True, # sample with replacement
 n_samples=len(negative), # match number in majority class
 random_state=27) # reproducible results
# combine majority and upsampled minority
downsampled = pd.concat([negative, pos_downsampled])

print(downsampled.labels.value_counts())
# Split the training data again
X_train = downsampled.drop(columns=['labels'])
y_train = downsampled['labels']

print("finish resampling, start training models")
# Train a new DecisionTreeClassifier
dt_model = LogisticRegression(random_state=42)
dt_model.fit(X_train, y_train)

# Predict on the test set
y_pred = dt_model.predict(X_test)
y_pred_prob = dt_model.predict_proba(X_test)[:, 1]

# Calculate accuracy, F1 score, and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Calculate Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)

# Save metrics to a text file
metrics_file_path = '/home/ndo/vardict_ML/models_output/decision_tree__downsampling_metrics.txt'
with open(metrics_file_path, 'w') as file:
    file.write(f'Accuracy: {accuracy:.4f}\n')
    file.write(f'F1 Score: {f1:.4f}\n')
    file.write(f'Confusion Matrix:\n{conf_matrix}\n')
    file.write(f'ROC AUC: {roc_auc:.4f}\n')
    file.write(f'Precision-Recall Curve:\n')
    file.write(f'Precision: {np.array2string(precision, precision=4)}\n')
    file.write(f'Recall: {np.array2string(recall, precision=4)}\n')

# Save the model to a new file
model_file_path = '/home/ndo/vardict_ML/models_output/decision_tree__downsampling_model.pkl'
with open(model_file_path, 'wb') as model_file:
    pickle.dump(dt_model, model_file)

print("DecisionTreeClassifier model re-saved successfully.")
print(f"Model metrics written to {metrics_file_path}.")

