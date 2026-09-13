# Gaussian Naive Bayes

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, roc_auc_score)

# load the dataset
df = pd.read_csv("data\ChurnGuard_processed.csv")
df.info()

X = df.drop(columns="churn")
y = df.churn

# Data Preprocessing & Gaussian Naive Bayes Pipeline
num_var = X.select_dtypes(exclude="object").columns
cat_var = X.select_dtypes(include="object").columns

cat_tran = Pipeline([("encode", OneHotEncoder(drop="first", handle_unknown="ignore"))])
num_tran = Pipeline([("num", StandardScaler())])
preprocessing = ColumnTransformer([("num", num_tran, num_var), ("cat", cat_tran, cat_var)])
model = Pipeline([("preprocessing", preprocessing), ("classifier", GaussianNB())])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=47)

# Define GridSearch parameters
param_grid = {
    "classifier__var_smoothing": np.logspace(-12, -6, 20)
}

# GridSearchCV
grid_nb = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1, scoring="accuracy")
grid_nb.fit(X_train, y_train)

# Best parameters
print("Best Parameters:")
print(grid_nb.best_params_)

print("\nBest CV Accuracy:")
print(grid_nb.best_score_)

# Model Performance Evaluation
best_nb = grid_nb.best_estimator_

y_pred = best_nb.predict(X_test)
y_pred_prob = best_nb.predict_proba(X_test)[:, 1]

accuracy_score(y_test, y_pred)
roc_auc_score(y_test, y_pred_prob)
print(classification_report(y_test, y_pred))
