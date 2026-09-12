# Logistic Regression

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, roc_auc_score)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# load the dataset
df = pd.read_csv("data\ChurnGuard_processed.csv")
df.info()

X = df.drop(columns="churn")
y = df.churn

# Data Preprocessing & Logistic Regression Pipeline
num_var = X.select_dtypes(exclude="object").columns
cat_var = X.select_dtypes(include="object").columns

num_tran = Pipeline([("scale", StandardScaler())])
cat_tran = Pipeline([("endoce", OneHotEncoder(drop="first", handle_unknown="ignore"))])

preprocessing = ColumnTransformer([("num", num_tran, num_var), ("cat", cat_tran, cat_var)])
model = Pipeline([("preprocessing", preprocessing), ("classifer", LogisticRegression(max_iter=1000))])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=47)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_prod = model.predict_proba(X_train)

# Model Performance Evaluation
confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
roc_auc_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
