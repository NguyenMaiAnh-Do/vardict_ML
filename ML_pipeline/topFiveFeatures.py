#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, PrecisionRecallDisplay
from sklearn.utils import resample
# Import your classifiers
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, Perceptron, RidgeClassifier, RidgeClassifierCV, SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from itertools import combinations

# Load your dataset 
df = pd.read_parquet('/home/ndo/vardict_ML/process_train_normalized_dataset.parquet')  # Load your dataset
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

# Load test set:
df = pd.read_parquet('/home/ndo/vardict_ML/process_test_106_normalized_dataset.parquet')
df.dropna(subset=['labels'], inplace=True)
# Define X (features) and y (target)
X_test = df.drop(columns=['labels'])  # Features
y_test = df['labels']  # Target labels

print("Finish loading test set")

# Function to load the saved model
def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

model_filename = '/home/ndo/vardict_ML/models_output/SL_19_models_normalized_crossVal.pkl'
models = load_model(model_filename)
model = models['AdaBoostClassifier']['model']

selected_features = [
    'DP', 'VD', 'AF', 'PMEAN', 'PSTD', 'QUAL', 'QSTD', 'SBF', 'ODDRATIO',
    'MQ', 'SN', 'HIAF', 'ADJAF', 'SHIFT3', 'MSI', 'MSILEN', 'NM', 'HICNT',
    'HICOV', 'DUPRATE', 'SPLITREAD', 'SPANPAIR'
]

results_hashmap = {}

def train_and_evaluate(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return accuracy, precision, recall

# Iterate through each selected feature, keep only that feature, and evaluate the model
for feature in selected_features:
    # Keep only the selected feature in both train and test datasets
    X_train_selected = X_train[[feature]]
    X_test_selected = X_test[[feature]]
    
    # Train the model and evaluate metrics
    accuracy, precision, recall = train_and_evaluate(X_train_selected, y_train, X_test_selected, y_test, model)
    
    # Store the results in the hashmap
    results_hashmap[feature] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall
    }

# Save the results hashmap to a pickle file
with open('/home/ndo/vardict_ML/models_output/topFive_features_results.pkl', 'wb') as f:
    pickle.dump(results_hashmap, f)

print("Results stored in topFive_features_results.pkl")


