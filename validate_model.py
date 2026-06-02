import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score,precision_score,roc_auc_score
from xgboost import XGBClassifier
import joblib
import sys

print("Starting the model Validation")

df = pd.read_csv("sample_dataset.csv")
X_val = df.drop('target', axis=1)
y_true = df['target']


print(df.columns.to_list())

pipe = joblib.load('saved_model.pkl')

predictions = pipe.predict(X_val)
prob = pipe.predict_proba(X_val)[:,1]

recall = recall_score(y_true,predictions)
precision = precision_score(y_true,predictions)
roc_auc = roc_auc_score(y_true,prob)

print(f"Recall is: {recall:.4f}")
print(f"Precision is: {precision:.4f}")
print(f"Roc_auc is: {roc_auc:.4f}")

RECALL_THRESHOLD = 0.40
AUC_THRESHOLD = 0.65

if recall < RECALL_THRESHOLD:
    print(f"FAILED RECALL SCORE IS BELOW THRESHOLD!!!, SCORE WAS {recall:.4f} ")
    sys.exit(1)

if roc_auc<AUC_THRESHOLD:
    print(f"FAILED ROC_AUC SCORE IS BELOW THRESHOLD!!!, AUC SCORE WAS {roc_auc:.4f}")
    sys.exit(1)

with open('trained_sample_model.pkl', 'wb') as f:
    joblib.dump(pipe,f)

print("PASSED: ALL METRICS WERE ABOVE THRESHOLD")
print("MODEL SAVED AS trained_sample_model.pkl")


