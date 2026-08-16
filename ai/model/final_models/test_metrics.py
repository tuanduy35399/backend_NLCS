from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    log_loss,
    top_k_accuracy_score,
)

BACKEND_DIR = Path(__file__).resolve().parents[3]

data_path = (
    BACKEND_DIR
    / "ai"
    / "data"
    / "mixed"
    / "dataset_balanced_holland_mixed.csv"
)

model_path = (
    BACKEND_DIR
    / "ai"
    / "model"
    / "final_models"
    / "mixed"
    / "best_model_randomforest_mixed.pkl"
)

data = pd.read_csv(data_path)

X = data.drop(columns=["NhomNganh"])
y = data["NhomNganh"]

model = joblib.load(model_path)

# Ma trận xác suất: số mẫu × số lớp
y_probability = model.predict_proba(X)

# Chuyển nhãn thật sang one-hot theo đúng thứ tự model.classes_
y_onehot = label_binarize(y, classes=model.classes_)

# sklearn MSE/MAE: trung bình trên cả mẫu và lớp
mse = mean_squared_error(y_onehot, y_probability)
mae = mean_absolute_error(y_onehot, y_probability)

# Multiclass Brier Score: cộng sai số của các lớp trong từng mẫu
brier = np.mean(
    np.sum((y_onehot - y_probability) ** 2, axis=1)
)

logloss = log_loss(
    y,
    y_probability,
    labels=model.classes_,
)

top3_accuracy = top_k_accuracy_score(
    y,
    y_probability,
    k=3,
    labels=model.classes_,
)

print(f"MSE:             {mse:.6f}")
print(f"MAE:             {mae:.6f}")
print(f"Brier Score:     {brier:.6f}")
print(f"Log Loss:        {logloss:.6f}")
print(f"Top-3 Accuracy:  {top3_accuracy:.6f}")