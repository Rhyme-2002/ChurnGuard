# Random Forest

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, roc_auc_score)

# load the dataset
df = pd.read_csv("data\ChurnGuard_processed.csv")
df.info()

X = df.drop(columns="churn")
y = df.churn

# Data Preprocessing & Random Forest Pipeline
num_var = X.select_dtypes(exclude="object").columns
cat_var = X.select_dtypes(include="object").columns

cat_tran = Pipeline([("encode", OneHotEncoder(drop="first", handle_unknown="ignore"))])
preprocessing = ColumnTransformer([("num", "passthrough", num_var), ("cat", cat_tran, cat_var)])
model = Pipeline([("preprocessing", preprocessing), ("classifier", RandomForestClassifier(random_state=38, n_jobs=-1))])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=47)

# Define GridSearch parameters
rf_params = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__criterion": ["gini", "entropy"],
    "classifier__max_depth": [None, 5, 10, 15],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 5],
    "classifier__max_features": ["sqrt", "log2"]
}

# GridSearchCV
grid_rf = GridSearchCV(estimator=model, param_grid=rf_params, scoring="accuracy", cv=5, n_jobs=-1, verbose=1)
grid_rf.fit(X_train, y_train)

# Best parameters
print("Best Parameters:")
print(grid_rf.best_params_)

print("\nBest CV Accuracy:")
print(grid_rf.best_score_)

# Model Performance Evaluation
best_dt = grid_rf.best_estimator_
y_pred = best_dt.predict(X_test)
y_pred_prob = best_dt.predict_proba(X_test)[:, 1]
confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
roc_auc_score(y_test, y_pred_prob)
print(classification_report(y_test, y_pred))
