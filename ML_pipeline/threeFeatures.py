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

def train_and_evaluate(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

results = []
# Iterate over combinations of 3 features
for feature_subset in combinations(X_train.columns, 3):
    # Select the subset of features
    X_train_subset = X_train[list(feature_subset)]
    X_test_subset = X_test[list(feature_subset)]
 
    accuracy = train_and_evaluate(X_train=X_train_subset, y_train=y_train, X_test=X_test_subset, y_test=y_test, model=model)
    results.append((feature_subset, accuracy))

# Save the results to a text file
with open("/home/ndo/vardict_ML/models_output/three_features_results_norm_SRR106_AdaBoostClassifier.txt", "w") as f:
    for subset, acc in results:
        f.write(f"Features: {subset}, Accuracy: {acc}\n")


