import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset/features.csv")
df["label"] = ((df["EAR"] < 0.23) | (df["Head_Tilt"] < -10)).astype(int)

X = df[["EAR", "Head_Tilt"]]
y = df["label"]

Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=120).fit(Xt, yt)

joblib.dump(model, "models/ml_model.pkl")
print("[INFO] ML baseline model trained and saved!")
print("[INFO] Validation accuracy:", model.score(Xv, yv))
