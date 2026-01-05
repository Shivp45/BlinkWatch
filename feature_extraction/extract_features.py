import pandas as pd
import numpy as np

df = pd.read_csv("dataset/features.csv")
df["label"] = ((df["EAR"] < 0.23) | (df["Head_Tilt"] < -10)).astype(int)

SEQ_LEN = 30
seqs, labels = [], []

for i in range(len(df) - SEQ_LEN):
    window = df["EAR"].iloc[i:i+SEQ_LEN].values
    seqs.append(window)
    labels.append(df["label"].iloc[i+SEQ_LEN])

X = np.array(seqs).reshape(-1, SEQ_LEN, 1)
y = np.array(labels)

np.save("dataset/X.npy", X)
np.save("dataset/y.npy", y)

print("[INFO] Feature sequences generated and saved for model training!")
