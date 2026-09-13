# CatBoost

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from catboost import CatBoostClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, roc_auc_score)

# Load the dataset
# load the dataset
df = pd.read_csv("data\ChurnGuard_processed.csv")
df.info()

X = df.drop(columns="churn")
y = df.churn

# Data Preprocessing & CatBoost Pipeline
num_var = X.select_dtypes(exclude="object").columns
cat_var = X.select_dtypes(include="object").columns

preprocessing = ColumnTransformer([("num", "passthrough", num_var), ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_var)])
model = Pipeline([("preprocessing", preprocessing), ("classifier", CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", random_seed=42, verbose=0, thread_count=-1))])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.27, random_state=57)

# Define GridSearch parameters
param_grid = {
    "classifier__depth": [4, 6, 8],
    "classifier__learning_rate": [0.03, 0.05, 0.1],
    "classifier__iterations": [100, 200],
    "classifier__l2_leaf_reg": [1, 3, 5]
}

# GridSearchCV
grid_cb = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1, return_train_score=True)
grid_cb.fit(X_train, y_train)

# Best Parameters
print("Best Parameters:")
print(grid_cb.best_params_)

print("\nBest CV Accuracy:")
print(grid_cb.best_score_)

# Model Performance Evaluation
best_cb = grid_cb.best_estimator_

y_pred = best_cb.predict(X_test)
y_pred_prob = best_cb.predict_proba(X_test)[:, 1]

confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
roc_auc_score(y_test, y_pred_prob)
print(classification_report(y_test, y_pred))
