import pandas as pd


# Đọc danh sách tổ hợp

df_tohop = pd.read_csv("to_hop_mon.csv", encoding="utf-8")

TO_HOP = (
    df_tohop.groupby("MaToHop")["MonHoc"]
    .apply(set)
    .to_dict()
)

# Hàm tính điểm tổ hợp


def tinh_to_hop(subject_scores, to_hop):

    ket_qua = []

    ds_mon = set(subject_scores.keys())

    for ma_to_hop, mon_hoc in to_hop.items():

        if mon_hoc.issubset(ds_mon):

            tong_diem = sum(subject_scores[m] for m in mon_hoc)

            ket_qua.append({
                "MaToHop": ma_to_hop,
                "DiemToHop": round(tong_diem, 2)
            })

    return ket_qua


# Đọc file điểm thi


df_diem = pd.read_excel("diemthiTHPT_tuoitre.xlsx")

tat_ca_ket_qua = []

for _, row in df_diem.iterrows():

    subject_scores = {}

    # bỏ 2 cột đầu
    for mon in df_diem.columns[2:]:

        diem = row[mon]

        if pd.notna(diem):
            subject_scores[mon] = float(diem)

    ket_qua = tinh_to_hop(subject_scores, TO_HOP)

    for kq in ket_qua:

        tat_ca_ket_qua.append({
            "SoBaoDanh": row["SOBAODANH"],
            "MaToHop": kq["MaToHop"],
            "DiemToHop": kq["DiemToHop"]
        })

# Xuất kết quả

df_output = pd.DataFrame(tat_ca_ket_qua)

df_output.to_excel("TongHopDiemThi.xlsx", index=False)

print(df_output.head())

print(f"Tổng số dòng sinh ra: {len(df_output)}")