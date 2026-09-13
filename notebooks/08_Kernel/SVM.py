# Support Vector Machine (SVM)

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, roc_auc_score)

# load the dataset
df = pd.read_csv("data\ChurnGuard_processed.csv")
df.info()

X = df.drop(columns="churn")
y = df.churn

# Data Preprocessing & SVM Pipeline
num_var = X.select_dtypes(exclude="object").columns
cat_var = X.select_dtypes(include="object").columns

num_tran = Pipeline([("scale", StandardScaler())])
cat_tran = Pipeline([("encode", OneHotEncoder(drop="first", handle_unknown="ignore"))])

preprocessing = ColumnTransformer([("num", num_tran, num_var), ("cat", cat_tran, cat_var)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.27, random_state=57)
model = Pipeline([("Preprocessing", preprocessing), ("classifier", SVC(random_state=46, probability=True))])

# Define GridSearch parameters
param_grid_svm = {
    "classifier__C": [0.1, 1, 10, 100],
    "classifier__kernel": ["linear", "poly", "rbf", "sigmoid"],
    "classifier__gamma": ["scale", "auto"]
}

# GridSearchCV
grid_svm = GridSearchCV(estimator=model, param_grid=param_grid_svm, cv=5, scoring="accuracy", n_jobs=-1)
grid_svm.fit(X_train, y_train)

# Best parameters
print("Best Parameters:")
print(grid_svm.best_params_)

print("\nBest CV Accuracy:")
print(grid_svm.best_score_)

# Model Performance Evaluation
best_svm = grid_svm.best_estimator_
y_pred = best_svm.predict(X_test)
y_pred_prob = best_svm.predict_proba(X_test)[:, 1]
confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
roc_auc_score(y_test, y_pred_prob)
print(classification_report(y_test, y_pred))

model = Pipeline([("Preprocessing", preprocessing), ("classifier", SVC(random_state=46, probability=True))])
