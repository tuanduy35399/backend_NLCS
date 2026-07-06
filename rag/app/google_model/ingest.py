import os
from langchain_community.vectorstores import Chroma #nhap vao vectorDB
from langchain_google_genai import GoogleGenerativeAIEmbeddings  #nhung Google AI Studio vao
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter #phan chia cac text thanh chunk
from langchain_core.documents import Document
import json
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()


with open("ctu_majors.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

docs = []

for item in documents:

    docs.append(
        Document(
            page_content=item["noi_dung"],
            metadata={
                "ten_nganh": item["ten_nganh"],
                "url": item["url"]
            }
        )
    )
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)


#Nhung Googel AI  (het quota roi nen bo khong xai)
# embedding = GoogleGenerativeAIEmbeddings(
#     model="models/gemini-embedding-2" #model embedding chuyen doc pdf 
#     #chi tiet model o day https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2?hl=vi
# )
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)

#tao vectorDB
db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="vector_db"
)

db.persist()

print("Đa tao thanh cong vector database",len(chunks))
