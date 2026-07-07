import random
import pandas as pd


path_nganh = "ThongKeByDuy.xlsx"
path_holland = "Holland_ThongKe_Mapping.xlsx"
path_diem = "TongHopDiemThi.xlsx"
path_output = "dataset.csv"

random.seed(42)


# Đọc dữ liệu
df_nganh = pd.read_excel(path_nganh)

df_holland = pd.read_excel(path_holland)

df_diem = pd.read_excel(path_diem)

print("Đã đọc dữ liệu")

# Chuẩn hóa bảng ngành
ds_nganh = []

for _, row in df_nganh.iterrows():

    tohop = [
        x.strip()
        for x in str(row["ToHopXetTuyenChung"]).split(",")
        if x.strip()
    ]

    ds_nganh.append(
        {
            "label": row["Label"],
            "tohop": set(tohop),
            "diem_tb": float(row["DiemTB"]),
            "diem_min": float(row["DiemMin"]),
        }
    )

print("Số nhóm ngành:", len(ds_nganh))

# Chuẩn hóa Holland
mapping_holland = {}

for _, row in df_holland.iterrows():

    tinh_cach = row["MaTinhCach"]

    ds = [
        x.strip()
        for x in str(row["NhomNganhNghe"]).split(";")
        
        if x.strip()
    ]

    for nganh in ds:

        mapping_holland.setdefault(nganh, []).append(tinh_cach)

print("Đã đọc Holland")

# Sinh dataset


dataset = []

for _, row in df_diem.iterrows():

    ma_to_hop = str(row["MaToHop"]).strip()

    diem = float(row["DiemToHop"])

    # tất cả ngành phù hợp
    nganh_hop_le = []

    for nganh in ds_nganh:

        if ma_to_hop not in nganh["tohop"]:
            continue

        if diem < nganh["diem_min"]:
            continue

        nganh_hop_le.append(nganh["label"])

    if len(nganh_hop_le) == 0:
        continue

    # Xuất mỗi ngành thành 1 dòng
    for label in nganh_hop_le:

        ds_tc = mapping_holland.get(label, [])

        tinh_cach = random.choice(ds_tc) if ds_tc else ""

        dataset.append(
            {
                "MaToHop": ma_to_hop,
                "DiemToHop": diem,
                "NhomTinhCach": tinh_cach,
                "NhomNganh": label,
            }
        )

# Xuất CSV

df_output = pd.DataFrame(dataset)

MAX_SAMPLE = 10000

balanced = []

for label, group in df_output.groupby("NhomNganh"):

    if len(group) > MAX_SAMPLE:
        group = group.sample(n=MAX_SAMPLE, random_state=42)

    balanced.append(group)

df_output = pd.concat(balanced, ignore_index=True)

df_output = df_output.sample(frac=1, random_state=42).reset_index(drop=True)

print(df_output["NhomNganh"].value_counts())
df_output.to_csv(
    path_output,
    index=False,
    encoding="utf-8-sig"
)