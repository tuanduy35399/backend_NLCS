import pandas as pd
import numpy as np 
import joblib 

from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
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
#Loading data
data = pd.read_csv(r'C:\Users\Duy\Documents\MyProject\nienLuanCoSo\backend_NLCS\ai\data\dataset.csv') 

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
    ('model', RandomForestClassifier()),
])

#Train model

#Gom cac model lai
# models = {
#     "LogisticRegression": LogisticRegression(),
#     "RandomForest": RandomForestClassifier(),
#     "GradientBoosting": GradientBoostingClassifier(),
#     "SVM": SVC()
# }
results = {}
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 20],
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
joblib.dump(grid.best_estimator_, 'best_model_randomforest.pkl')
joblib.dump(grid.cv_results_, 'cv_results.pkl')



# for name, model in models.items():
#     pipe.set_params(model= model)
#     result = cross_validate(pipe, x,y, cv=5, scoring=['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'])
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
# print(df)
# Cross Validation (chia fold)
#     ↓
# Trong mỗi fold:
#     ↓
#     SelectKBest (chọn feature trên train fold)
#     ↓
#     Train model
#     ↓
#     Validate


#Evaluate model accuarcy


