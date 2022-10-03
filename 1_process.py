import os
import pandas as pd
from sklearn.model_selection import train_test_split
"""
数据预处理：
1. 数据切分
2. 组织成fastnlp需要的格式
"""
 
df = pd.read_excel("./dataset/dataset0930.xlsx")
df = df.dropna()
# 去除内容的换行符、多余空格
df["content"] = df["content"].apply(lambda x: x.replace("\n", " ").strip())
df = df[df["content"] != ""]
X, y = df["content"].values, df["label"]
# 数据切分 train:80% test: 10% valid: 10%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5)

train_df = pd.DataFrame(data={"content": X_train, "label": y_train})
train_df["type"] = "train"

# size = train_df.shape[0]
# to_replace_size = int(size * 0.05)
# train_df.loc[train_df[train_df['label'] == 1].sample(to_replace_size).index, 'label'] = 0
# train_df.loc[train_df[train_df['label'] == 0].sample(to_replace_size).index, 'label'] = 1

test_df = pd.DataFrame(data={"content": X_test, "label": y_test})
test_df["type"] = "test"

# size = test_df.shape[0]
# to_replace_size = int(size * 0.03)
# test_df.loc[test_df[test_df['label'] == 1].sample(to_replace_size).index, 'label'] = 0
# test_df.loc[test_df[test_df['label'] == 0].sample(to_replace_size).index, 'label'] = 1

valid_df = pd.DataFrame(data={"content": X_valid, "label": y_valid})
valid_df["type"] = "valid"

# size = valid_df.shape[0]
# to_replace_size = int(size * 0.05)
# valid_df.loc[valid_df[valid_df['label'] == 1].sample(to_replace_size).index, 'label'] = 0
# valid_df.loc[valid_df[valid_df['label'] == 0].sample(to_replace_size).index, 'label'] = 1


df = pd.concat([train_df, test_df, valid_df])
df = df.to_csv("./dataset/processed0930.csv", index=False)