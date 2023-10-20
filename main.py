import pandas as pd
import numpy as np

from model import LogisticRegression
from model import Scaler

def one_hot_encoder(df,col_name):

    def one_hot(label, n_classes):
        one_hot = [0] * n_classes
        one_hot[label] = 1
        return one_hot
    
    n_classes = df[col_name].nunique()
    df[col_name] = df[col_name].apply(one_hot,args=(n_classes,))

    return df

df = pd.read_csv('train.csv')
df = df.drop("SNo", axis=1)

df = one_hot_encoder(df, 'Label')

# Shuffle the dataset to ensure good overall distribution of data
shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split dataset
train_size = int(0.8 * len(shuffled_df))

train_df = shuffled_df[:train_size]
val_df = shuffled_df[train_size:]

print("Training set size:", len(train_df))
print("Validation set size:", len(val_df))

train_X = train_df.iloc[:, :-1]
train_X = Scaler.minmax(train_X)

train_Y = train_df.iloc[:, -1]

val_X = val_df.iloc[:, :-1]
val_X = Scaler.minmax(val_X)

val_Y = val_df.iloc[:, -1]

LR = LogisticRegression(learning_rate=0.001,epochs=10000)
LR.train(train_X,train_Y)

pred = LR.predict(val_X)

pred = [np.argmax(p) for p in pred]
val_Y = [np.argmax(v) for v in val_Y]

count = 0
for i in range(len(pred)):
    if pred[i] == val_Y[i]:
        count += 1

accuracy = count / len(pred)

print(f"Accuracy {count} / {len(pred)} = {accuracy}")

