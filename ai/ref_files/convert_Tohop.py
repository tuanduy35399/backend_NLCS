
import pandas as pd
from itertools import combinations


user_subjects = ["Toan", "Vat li", "Hoa hoc", "Tieng Anh"]

subject_scores = {
    "Toan": 9.0,
    "Vat li": 8.5,
    "Hoa hoc": 7.0,
    "Tieng Anh": 6.5
}

def tinh_to_hop(ds_mon, ds_diem, to_hop):
    ds_mon= set(ds_mon)
    results = []
    
    for ma_to_hop, mon_hoc in to_hop.items():
        if mon_hoc.issubset(ds_mon):
            tong_diem = sum(ds_diem[m] for m in mon_hoc)
            
            results.append({
                "MaToHop":ma_to_hop,
                "MonHoc":list(mon_hoc),
                "DiemToHop": tong_diem
            })
    return results

df = pd.read_csv("to_hop_mon.csv", encoding="utf-8")

TO_HOP = df.groupby("MaToHop")["MonHoc"].apply(set).to_dict()

df_out = pd.DataFrame.from_dict(TO_HOP, orient="index")
# print(df_out)

print(tinh_to_hop(user_subjects,subject_scores,TO_HOP ))