import os
import pandas as pd
from sklearn.model_selection import train_test_split
"""
数据预处理：
1. 数据切分
2. 组织成fastnlp需要的格式
"""
 
df = pd.read_excel("./dataset/dataset0929.xlsx")
df = df.dropna()
# 去除内容的换行符、多余空格
df["content"] = df["content"].apply(lambda x: x.replace("\n", " ").strip())
print(df["label"].unique())
df = df[df["content"] != ""]
print(df["label"].unique())
X, y = df["content"].values, df["label"].values
# 数据切分 train:80% test: 10% valid: 10%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5)

train_df = pd.DataFrame(data={"content": X_train, "label": y_train})
train_df["type"] = "train"
test_df = pd.DataFrame(data={"content": X_test, "label": y_test})
test_df["type"] = "test"
valid_df = pd.DataFrame(data={"content": X_valid, "label": y_valid})
valid_df["type"] = "valid"
df = pd.concat([train_df, test_df, valid_df])
df = df.to_csv("./dataset/processed0929.csv", index=False)