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
df = pd.read_parquet('/home/ndo/vardict_ML/train_test_shuffle_data/train_scaled_dataset.parquet')  # Load your dataset

X_train = df.drop(columns=['labels'])  # Features
y_train = df['labels']  # Target labels

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

##### TODO: use standardization and normalization data, not this one

# Logistic Regression with L1 regularization (Lasso)
logreg_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)  # C is the inverse of regularization strength
logreg_l1.fit(X_train, y_train)


# Logistic Regression with L2 regularization (Ridge)
logreg_l2 = LogisticRegression(penalty='l2', solver='liblinear', C=1.0)
logreg_l2.fit(X_train, y_train)


# Logistic Regression with both L1 and L2 (Elastic Net-like regularization)
logreg_l1_l2 = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=1.0, max_iter=10000)
logreg_l1_l2.fit(X_train, y_train)



# Save models to a pickle file
with open('/home/ndo/vardict_ML/models_output/LR_downsam_stand_norm_L1_L2', 'wb') as pickle_file:
    pickle.dump({
        'L1': logreg_l1,
        'L2': logreg_l2,
        'L1_L2': logreg_l1_l2
    }, pickle_file)

print("Models saved to /home/ndo/vardict_ML/models_output/LR_downsam_stand_norm_L1_L2.pkl")