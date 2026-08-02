import joblib
import pandas as pd 


model = joblib.load(r'C:\Users\Duy\Documents\MyProject\nienLuanCoSo\backend_NLCS\ai\model\train_model\best_model_randomforest_mbti.pkl')

# print(type(data))
# print(data)

ma= input("Nhap vao ma to hop:").strip()
diem= input("Nhap vao diem thi:").strip()
tinh_cach= input("Nhap vao tinh cach:").strip()
para = pd.DataFrame([{
    "MaToHop": ma,
    "DiemToHop":diem,
    "NhomTinhCach": tinh_cach,
}])
pred = model.predict(para)

print(type(pred))
for i in pred:
    print('Nhom nganh phu hop: ', i)