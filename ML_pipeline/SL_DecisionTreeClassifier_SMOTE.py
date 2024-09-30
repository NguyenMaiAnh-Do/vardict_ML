#!/usr/bin/env python3

# Re-train the Decision Tree Classifier
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle
from imblearn.over_sampling import SMOTE
# Load the data again
df = pd.read_parquet('/home/ndo/vardict_ML/processed_data.parquet')
df['VD'] = df['VD'].fillna(df['VD'].mean())
X = df.drop(columns=['labels'])
y = df['labels']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sm = SMOTE(random_state=27)
X_train, y_train = sm.fit_resample(X_train, y_train)
# Train a new DecisionTreeClassifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Save the model to a new file
with open('/home/ndo/vardict_ML/models_output/decistion_tree_smote', 'wb') as file:
    pickle.dump(dt_model, file)

print("DecisionTreeClassifier model re-saved successfully.")