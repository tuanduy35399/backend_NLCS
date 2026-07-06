from fastapi import FastAPI
import joblib
import pandas as pd 
from pydantic import BaseModel
from typing import List, Dict
from rag.app.google_model.rag import ask
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
app = FastAPI()

model = joblib.load("ai/model/final_models/best_model_randomforest.pkl")
classes = model.classes_
df = pd.read_csv("to_hop_mon.csv", encoding="utf-8")
to_hop = df.groupby("MaToHop")["MonHoc"].apply(set).to_dict()
origins = [
    "http://localhost:5173",  
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def tinh_to_hop(ds_mon, ds_diem):

    ds_mon= set(ds_mon)
    results = []
    
    for ma_to_hop, mon_hoc in to_hop.items():
        if mon_hoc.issubset(ds_mon):
            tong_diem = sum(ds_diem[m] for m in mon_hoc)
            
            results.append({
                "MaToHop":ma_to_hop,
                "DiemToHop": tong_diem
            })
    return results

def call_model(list_results, nhom_tc):
    kq = []

    for item in list_results:

        df = pd.DataFrame([{
            "MaToHop": item["MaToHop"],
            "DiemToHop": item["DiemToHop"],
            "NhomTinhCach": nhom_tc
        }])

        probs = model.predict_proba(df)[0]

        top3_idx = np.argsort(probs)[::-1][:3]

        top3 = [
            {
                "NhomNganh": classes[i],
                "XacSuat": round(float(probs[i]), 4)
            }
            for i in top3_idx
        ]

        kq.append({
            "MaToHop": item["MaToHop"],
            "DiemToHop": item["DiemToHop"],
            "Top3": top3
        })

    return kq


class PredictRequest(BaseModel):
    subjects: List[str]
    scores: Dict[str, float]
    holland: str
    
class Question(BaseModel):
    group_major:str
    describe:str



@app.get("/")
def home():

    return {
        "status":"running"
    }


@app.post("/predict")
def predict_group_major(data: PredictRequest):

    subjects = data.subjects    
    scores = data.scores        
    nhom_tc = data.holland      

    list_to_hop = tinh_to_hop(subjects, scores)

    result = call_model(list_to_hop, nhom_tc)

    return {
        "input": data,
        "recommendations": result
    }


@app.post("/chat")
def chat(q:Question):
    
    answer=ask(
        q.group_major, q.describe
    )
    return {
        "answer":answer
    }
    