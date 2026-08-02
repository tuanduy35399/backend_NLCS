import argparse
from pathlib import Path

import pandas as pd
import numpy as np 
import joblib 

from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer #link other pipelines together
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Tien xu ly data, lam data clean hon
from sklearn.impute import SimpleImputer #Xu ly bat khi du lieu bi missing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier #Chon thuat toan
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score #test model sau khi train xong #nay 
from sklearn.metrics import accuracy_score, classification_report
# Loading data.  Select the school-year dataset explicitly so models trained
# on 2025 and 2026 cannot accidentally overwrite or mix with each other.
parser = argparse.ArgumentParser()

parser.add_argument("--year", choices=["2025", "2026", "mixed"], default="2025")
parser.add_argument("--dataset", type=Path, default=None)
args = parser.parse_args()

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
dataset_path = args.dataset or (DATA_DIR / args.year / f"dataset_balanced_holland_{args.year}.csv")
if not dataset_path.exists():
    raise FileNotFoundError(
        f"Không tìm thấy dataset: {dataset_path}. Hãy chạy gen_dataset.py --year {args.year} trước."
    )
data = pd.read_csv(dataset_path)
#Create feature and target variable
target= "NhomNganh"
x= data.drop(target, axis= 1) #bo cot label
y= data[target] #label

#Split
numerical_cols = ["DiemToHop"]
categorial_cols = ["MaToHop", "NhomTinhCach"]

#Start PipeLine w/ Encoding
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
])
categorial_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore')),   
])

#Join the pipelines 
preprocessor= ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cate',categorial_pipeline, categorial_cols),
])

#combine the pipelines
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('select', SelectKBest(k=10)),
    ('model', GradientBoostingClassifier()),
])

#Train model

#Gom cac model lai
# models = {
#     "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
#     "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
#     "GradientBoosting": GradientBoostingClassifier(random_state=42),
#     "SVM": SVC()
# }
results = {}
param_grid = {
    'model__n_estimators': [100, 200],
    "model__learning_rate": [0.03, 0.05],
    'model__max_depth': [2,3],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1, 2],
    'select__k': [5, 10],
}

grid  = RandomizedSearchCV(
    pipe,
    param_distributions=param_grid,
    n_iter=30, 
    cv=5,
    scoring={
        'accuracy': 'accuracy',
        'f1': 'f1_weighted',
        'precision': 'precision_weighted',
        'recall': 'recall_weighted'
    },
    refit='f1', #tieu chi chinh de chon best model
    n_jobs=-1,
    verbose=2,
    random_state=42,
    return_train_score=True,
)
grid.fit(x, y) #tien hanh train model

print(grid.best_params_)
print(grid.cv_results_)
MODEL_DIR = Path(__file__).resolve().parents[1] / "final_models" / args.year
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(grid.best_estimator_, MODEL_DIR / f"best_model_gbc_{args.year}.pkl")
joblib.dump(grid.cv_results_, MODEL_DIR / f"cv_results_{args.year}.pkl")



# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# for name, model in models.items():
#     pipe.set_params(model= model)
#     print(f"\nĐang đánh giá {name} trên dataset {args.year}...")
#     result = cross_validate(
#         pipe,
#         x,
#         y,
#         cv=cv,
#         scoring=['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'],
#         n_jobs=1,
#     )
#     results[name] = {
#         "accuracy_mean": result['test_accuracy'].mean(),
#         "accuracy_std": result['test_accuracy'].std(),
#         "f1_mean": result['test_f1_weighted'].mean(),
#         "f1_std": result['test_f1_weighted'].std(),
#         "precision_mean": result['test_precision_weighted'].mean(),
#         "precision_std": result['test_precision_weighted'].std(),
#         "recall_mean": result['test_recall_weighted'].mean(),
#         "recall_std": result['test_recall_weighted'].std()
#     }
# df = pd.DataFrame(results).T
# df = df.sort_values("accuracy_mean", ascending=False)
# print("\nKết quả xếp theo accuracy_mean:")
# print(df.to_string(float_format=lambda value: f"{value:.6f}"))

# # Benchmark only: save the comparison table, not a model.  The best algorithm
# # can be tuned and trained separately after reviewing this file.
# RESULT_DIR = Path(__file__).resolve().parents[1] / "benchmark_results"
# RESULT_DIR.mkdir(parents=True, exist_ok=True)
# result_path = RESULT_DIR / f"comparison_{args.year}.csv"
# df.to_csv(result_path, encoding="utf-8-sig")
# print(f"\nĐã lưu bảng so sánh: {result_path}")




# Lệnh để chạy
# python ai/model/train_model/train.py --year 2025
# python ai/model/train_model/train.py --year 2026
# python ai/model/train_model/train.py --year mixed

