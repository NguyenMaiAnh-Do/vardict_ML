#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
# Import your classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load your dataset and drop the VD column
df = pd.read_parquet('/home/ndo/vardict_ML/processed_data.parquet')  # Load your dataset
df['VD'] = df['VD'].fillna(df['VD'].mean())  # Handle missing values in column 'VD'
X = df.drop(columns=['labels'])  # Features
y = df['labels']  # Target labels
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Combine them back for resampling
train_data = pd.concat([X_train, y_train], axis=1)

# Separate minority and majority classes
negative = train_data[train_data.labels == 0]
positive = train_data[train_data.labels == 1]

# Downsample the minority class
pos_downsampled = resample(positive,
                           replace=True,  # Sample with replacement
                           n_samples=len(negative),  # Match number in majority class
                           random_state=27)  # Reproducible results

# Combine majority and downsampled minority
downsampled = pd.concat([negative, pos_downsampled])

print(downsampled.labels.value_counts())

# Split the downsampled data into features and labels
X_train = downsampled.drop(columns=['labels'])
y_train = downsampled['labels']

print("Finished resampling, starting to train the model")

# Logistic Regression with L1 regularization (Lasso)
logreg_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)  # C is the inverse of regularization strength
logreg_l1.fit(X_train, y_train)
y_pred_l1 = logreg_l1.predict(X_test)

# Logistic Regression with L2 regularization (Ridge)
logreg_l2 = LogisticRegression(penalty='l2', solver='liblinear', C=1.0)
logreg_l2.fit(X_train, y_train)
y_pred_l2 = logreg_l2.predict(X_test)

# Logistic Regression with both L1 and L2 (Elastic Net-like regularization)
logreg_l1_l2 = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=1.0, max_iter=10000)
logreg_l1_l2.fit(X_train, y_train)
y_pred_l1_l2 = logreg_l1_l2.predict(X_test)

# Evaluate the models
print(f"L1 Regularization Accuracy: {accuracy_score(y_test, y_pred_l1)}")
print(f"L2 Regularization Accuracy: {accuracy_score(y_test, y_pred_l2)}")
print(f"L1 + L2 Regularization Accuracy: {accuracy_score(y_test, y_pred_l1_l2)}")

# Save models to a pickle file
with open('/home/ndo/vardict_ML/models_output/LR_downsam_L1_L2', 'wb') as pickle_file:
    pickle.dump({
        'L1': logreg_l1,
        'L2': logreg_l2,
        'L1_L2': logreg_l1_l2
    }, pickle_file)

print("Models saved to /home/ndo/vardict_ML/models_output/LR_downsam_L1_L2.pkl")