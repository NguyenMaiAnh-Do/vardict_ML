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
    # ('AdaBoostClassifier', AdaBoostClassifier),
    # ('BaggingClassifier', BaggingClassifier),
    # ('BernoulliNB', BernoulliNB),
    # ('CalibratedClassifierCV', CalibratedClassifierCV),
    # ('CategoricalNB', CategoricalNB),
    # ('DecisionTreeClassifier', DecisionTreeClassifier),
    # ('DummyClassifier', DummyClassifier),
    # ('ExtraTreeClassifier', ExtraTreeClassifier),
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
# # Combine them back for resampling
# train_data = pd.concat([X_train, y_train], axis=1)

# # Separate minority and majority classes
# negative = train_data[train_data.labels == 0]
# positive = train_data[train_data.labels == 1]

# # Downsample the minority class
# pos_downsampled = resample(positive,
#                            replace=True,  # Sample with replacement
#                            n_samples=len(negative),  # Match number in majority class
#                            random_state=27)  # Reproducible results

# # Combine majority and downsampled minority
# downsampled = pd.concat([negative, pos_downsampled])

# print(downsampled.labels.value_counts())

# # Split the downsampled data into features and labels
# X_train = downsampled.drop(columns=['labels'])
# y_train = downsampled['labels']

# print("Finished resampling, starting to train the model")

# Dictionary to store model performance and trained models
results = {}
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
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Store results
        results[name] = {'Accuracy': acc, 'Precision': precision, 'Recall': recall, 'F1-score': f1}
        
        # Store model
        all_models[name] = model
        
        # Print metrics
        print(f"{name}: Accuracy = {acc:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1-score = {f1:.4f}")
        
        # Plot Precision-Recall curve
        # if hasattr(model, "predict_proba"):
        #     display = PrecisionRecallDisplay.from_estimator(model, X_test, y_test, name=name)
        #     display.ax_.set_title(f"Precision-Recall Curve: {name}")
        #     plt.show()

    except Exception as e:
        print(f"Failed to run {name}: {e}")

# Save all models into one pickle file
save_all_models(all_models, '/home/ndo/vardict_ML/models_output/Extra_to_LGBMC_downsampling.pkl')

# Optional: Save the results to a CSV or log file
results_df = pd.DataFrame(results).T  # Transpose to get models as rows
results_df.to_csv('/home/ndo/vardict_ML/models_output/models_perf_Extra_to_LGBMC_downsampling.csv', index=True)
