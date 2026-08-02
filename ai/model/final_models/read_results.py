import joblib
import pandas as pd

results = joblib.load(r"C:\Users\Duy\Documents\MyProject\nienLuanCoSo\backend_NLCS\ai\model\final_models\2025\cv_results_2025.pkl")
df = pd.DataFrame(results)

best_row = df.loc[df["rank_test_f1"].idxmin()]

print("Best parameters:", best_row["params"])
print("Accuracy:", best_row["mean_test_accuracy"])
print("F1:", best_row["mean_test_f1"])
print("Precision:", best_row["mean_test_precision"])
print("Recall:", best_row["mean_test_recall"])