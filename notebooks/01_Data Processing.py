import pandas as pd

df = pd.read_csv("data\ChurnGuard_raw.csv")
df.info()
pd.isna(df).sum()

ChurnGuard = df.dropna()
pd.isna(ChurnGuard).sum()

ChurnGuard.drop(columns="customer_id", inplace=True)
ChurnGuard.duplicated().sum()

ChurnGuard.to_csv("data\ChurnGuard_processed.csv", index=False)
