# Gradient Boosting

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, roc_auc_score)

# Load the dataset
# load the dataset
df = pd.read_csv("data\ChurnGuard_processed.csv")
df.info()

X = df.drop(columns="churn")
y = df.churn

# Data Preprocessing & Gradient Boosting Pipeline
num_var = X.select_dtypes(exclude="object").columns
cat_var = X.select_dtypes(include="object").columns

preprocessing = ColumnTransformer([("num", "passthrough", num_var), ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_var)])
model = Pipeline([("preprocessing", preprocessing), ("classifier", GradientBoostingClassifier(random_state=42))])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.27, random_state=57)

# Define GridSearch parameters
param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__learning_rate": [0.01, 0.05, 0.1],
    "classifier__max_depth": [3, 5, 7],
    "classifier__subsample": [0.8, 1.0],
    "classifier__colsample_bytree": [0.8, 1.0]
}

# GridSearchCV
grid_gb = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1, return_train_score=True)
grid_gb.fit(X_train, y_train)

# Best Parameters
print("Best Parameters:")
print(grid_gb.best_params_)

print("\nBest CV Accuracy:")
print(grid_gb.best_score_)

# Model Performance Evaluation
best_gb = grid_gb.best_estimator_

y_pred = best_gb.predict(X_test)
y_pred_prob = best_gb.predict_proba(X_test)[:, 1]

confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
roc_auc_score(y_test, y_pred_prob)
print(classification_report(y_test, y_pred))
