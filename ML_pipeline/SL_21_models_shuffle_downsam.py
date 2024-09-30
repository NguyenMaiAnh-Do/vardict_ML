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

# Custom classifier list
clf_list = [
    ('AdaBoostClassifier', AdaBoostClassifier),
    ('BaggingClassifier', BaggingClassifier),
    ('BernoulliNB', BernoulliNB),
    ('CalibratedClassifierCV', CalibratedClassifierCV),
    ('CategoricalNB', CategoricalNB),
    ('DecisionTreeClassifier', DecisionTreeClassifier),
    ('DummyClassifier', DummyClassifier),
    ('ExtraTreeClassifier', ExtraTreeClassifier),
    ('ExtraTreesClassifier', ExtraTreesClassifier),
    ('GaussianNB', GaussianNB),
    ('KNeighborsClassifier', KNeighborsClassifier),
    ('LabelPropagation', LabelPropagation),
    ('LabelSpreading', LabelSpreading),
    ('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis),
    ('LinearSVC', LinearSVC),
    ('LogisticRegression', LogisticRegression),
    ('NearestCentroid', NearestCentroid),
    ('PassiveAggressiveClassifier', PassiveAggressiveClassifier),
    ('Perceptron', Perceptron),
    ('QuadraticDiscriminantAnalysis', QuadraticDiscriminantAnalysis),
    ('RandomForestClassifier', RandomForestClassifier),
    ('RidgeClassifier', RidgeClassifier),
    ('RidgeClassifierCV', RidgeClassifierCV),
    ('SGDClassifier', SGDClassifier),
    ('StackingClassifier', StackingClassifier),
    ('XGBClassifier', XGBClassifier),
    ('LGBMClassifier', LGBMClassifier)
]

# Load your dataset and drop the VD column
df = pd.read_parquet('/home/ndo/vardict_ML/train_test_shuffle_data/train_dataset.parquet')  # Load your dataset

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

all_models = {}

# Function to save all models into one pickle file
def save_all_models(models, filename='all_models.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(models, file)

# Loop through each classifier and evaluate
for name, clf in clf_list:
    try:
        model = clf()
        model.fit(X_train, y_train)
        # Store model
        all_models[name] = model
        
    except Exception as e:
        print(f"Failed to run {name}: {e}")

# Save all models into one pickle file
save_all_models(all_models, '/home/ndo/vardict_ML/models_output/SL_21_models_downsampling_shuffle.pkl')


