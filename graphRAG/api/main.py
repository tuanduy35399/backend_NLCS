from fastapi import FastAPI
import joblib
import pandas as pd 
from pydantic import BaseModel
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import os
from pathlib import Path
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.property_graph import PropertyGraphIndex

from graphRAG.indexing.embedding.bge_m3 import BGE_M3_Embedding
from graphRAG.indexing.graph.graph_store import GraphStore
from graphRAG.indexing.vector.chroma import ChromaStore
from graphRAG.llm.answer_generator import AnswerGenerator
from graphRAG.llm.gemini import GeminiLLM
from graphRAG.retrieval.retrieve import Retriever

app = FastAPI()
rag_pipeline = None

BACKEND_DIR = Path(__file__).resolve().parents[2]
# MODEL_YEAR = os.getenv("MODEL_YEAR", "2025")
MODEL_YEAR = "2025"
MODEL_FILES = {
    "2025": "best_model_gbc_2025.pkl",
    "2026": "best_model_gbc_2026.pkl",
    "mixed": "best_model_randomforest_mixed.pkl",
}

if MODEL_YEAR not in MODEL_FILES:
    raise ValueError("MODEL_YEAR phải là 2025, 2026 hoặc mixed")

model_path = (
    BACKEND_DIR / "ai" / "model" / "final_models"
    / MODEL_YEAR / MODEL_FILES[MODEL_YEAR]
)
model = joblib.load(model_path)
classes = model.classes_
df = pd.read_csv(BACKEND_DIR / "ai" / "data" / "to_hop_mon.csv", encoding="utf-8")
to_hop = df.groupby("MaToHop")["MonHoc"].apply(set).to_dict()
origins = [
    "http://localhost:5173",  
    "http://127.0.0.1:5173",
    "https://guessyourjob.vercel.app",
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


def get_rag_pipeline():
    global rag_pipeline

    if rag_pipeline is None:
        llm = GeminiLLM().get_llm()
        embed_model = BGE_M3_Embedding().get_model()

        chroma_store = ChromaStore()
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store=chroma_store.vector_store,
            embed_model=embed_model,
        )

        graph_store = GraphStore().get_store()
        graph_index = PropertyGraphIndex.from_existing(
            property_graph_store=graph_store,
            llm=llm,
            embed_model=embed_model,
        )

        rag_pipeline = {
            "retriever": Retriever(
                llm=llm,
                vector_index=vector_index,
                graph_index=graph_index,
            ),
            "answer_generator": AnswerGenerator(llm=llm),
        }

    return rag_pipeline



@app.get("/")
def home():

    return {
        "status":"running",
        "model_year": MODEL_YEAR,
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
    question = f"{q.group_major}\n{q.describe}"
    pipeline = get_rag_pipeline()
    contexts = pipeline["retriever"].retrieve(question)
    answer = pipeline["answer_generator"].generate(question, contexts)

    return {
        "answer": answer
    }
    
