# LDA

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, roc_auc_score)

# Load the dataset
df = pd.read_csv("data\ChurnGuard_processed.csv")
df.info()
X = df.drop(columns="churn")
y = df.churn

# Data Preprocessing & Linear Discriminant Analysis Pipeline
num_var = X.select_dtypes(exclude="object").columns
cat_var = X.select_dtypes(include="object").columns

cat_tran = Pipeline([("encode", OneHotEncoder(drop="first", handle_unknown="ignore"))])
num_tran = Pipeline([("num", StandardScaler())])
preprocessing = ColumnTransformer([("num", num_tran, num_var), ("cat", cat_tran, cat_var)])
model = Pipeline([("preprocessing", preprocessing), ("classifier", LinearDiscriminantAnalysis())])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=47)

model.fit(X_train, y_train)

# Model Performance Evaluation
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

accuracy_score(y_test, y_pred)
roc_auc_score(y_test, y_pred_prob)
print(classification_report(y_test, y_pred))
