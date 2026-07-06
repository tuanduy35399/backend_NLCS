import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('ai/model/final_models/dataset.csv')

x = data.drop('NhomNganh', axis=1)
y = data["NhomNganh"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42 )


# x_test = data[["MaToHop", "DiemToHop", "NhomTinhCach"]]
# y_true = data["NhomNganh"]
model = joblib.load('ai/model/final_models/best_model_randomforest.pkl')

y_pred = model.predict(x_test)

cm = confusion_matrix(y_test, y_pred, normalize="true")

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True,  cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()