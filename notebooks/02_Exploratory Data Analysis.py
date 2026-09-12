import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("data\ChurnGuard_processed.csv")

# Basic inspection
df.head()
df.tail()
df.shape
df.columns

# Summary statistics
df.describe()
df.describe(include="object")
df.describe(include="all")

# Check data types
df.dtypes
df.select_dtypes(include="object").columns
df.select_dtypes(include=np.number).columns

# Check unique values
for col in df.columns:
    print(f"\n{col}")
    print(df[col].nunique())
    print(df[col].unique()[:10])

# Target variable analysis
df["churn"].value_counts()
df["churn"].value_counts(normalize=True) * 100

sns.countplot(data=df, x="churn")
plt.title("Customer Churn Distribution")
plt.show()

# Numerical variable distribution
df.hist(figsize=(15, 12), bins=30)
plt.tight_layout()
plt.show()

# Correlation
corr = df.select_dtypes(include=np.number).corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f")

plt.title("Correlation Matrix")
plt.show()

