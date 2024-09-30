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


# Load your dataset and drop the VD column
df = pd.read_parquet('/home/ndo/vardict_ML/train_test_shuffle_data/test_scaled_dataset.parquet')  # Load your dataset
df.dropna(subset=['labels'], inplace=True)
X_test = df.drop(columns=['labels'])  # Features
y_test = df['labels']  # Target labels

# Load the model
def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

# Load your AdaBoostClassifier model from the pickle file
model_filename = '/home/ndo/vardict_ML/models_output/SL_21_models_stand_norm_downsampling_shuffle.pkl'
models = load_model(model_filename)

excluded_names = ['KNeighborsClassifier','NearestCentroid' ]
results = {}

# Loop through each classifier and evaluate
for name, model in models.items():
    if name in excluded_names:
        continue
    try:  
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Store results
        results[name] = {'Accuracy': acc, 'Precision': precision, 'Recall': recall, 'F1-score': f1}

        # Print metrics
        print(f"{name}: Accuracy = {acc:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1-score = {f1:.4f}")
        

    except Exception as e:
        print(f"Failed to run {name}: {e}")


# Optional: Save the results to a CSV or log file
results_df = pd.DataFrame(results).T  # Transpose to get models as rows
results_df.to_csv('/home/ndo/vardict_ML/models_output/models_perf_stand_norm_19.csv', index=True)
