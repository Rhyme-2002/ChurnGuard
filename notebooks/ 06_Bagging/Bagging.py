# Bagging

import  pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, roc_auc_score)

# load the dataset
df = pd.read_csv("data\ChurnGuard_processed.csv")
df.info()

X = df.drop(columns="churn")
y = df.churn

# Data Preprocessing & Bagging Pipeline
num_var = X.select_dtypes(exclude="object").columns
cat_var = X.select_dtypes(include="object").columns

cat_tran = Pipeline([("encode", OneHotEncoder(drop="first", handle_unknown="ignore"))])
preprocessing = ColumnTransformer([("num", "passthrough", num_var), ("cat", cat_tran, cat_var)])
model = Pipeline([("preprocessing", preprocessing), ("classifier", BaggingClassifier(estimator=DecisionTreeClassifier(random_state=14), random_state=17, n_jobs=-1))])

# Define GridSearch parameters
bagging_params = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_samples": [0.7, 1.0],
    "classifier__max_features": [0.7, 1.0],
    "classifier__bootstrap": [True, False],
    "classifier__estimator__max_depth": [None, 5, 10],
    "classifier__estimator__min_samples_split": [2, 5],
    "classifier__estimator__min_samples_leaf": [1, 2]
}

# GridSearchCV
grid_bd = GridSearchCV(estimator=model, param_grid=bagging_params, cv=5, n_jobs=-1, verbose=1, scoring="accuracy")
grid_bd.fit(X_train, y_train)

# Best parameters
print("Best Parameters:")
print(grid_dt.best_params_)

print("\nBest CV Accuracy:")
print(grid_dt.best_score_)

# Model Performance Evaluation
best_dt = grid_bd.best_estimator_
y_pred = best_dt.predict(X_test)
y_pred_prob = best_dt.predict_proba(X_test)[:, 1]
accuracy_score(y_test, y_pred)
roc_auc_score(y_test, y_pred_prob)
print(classification_report(y_test, y_pred))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=47)
