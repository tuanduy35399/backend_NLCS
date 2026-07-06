import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import (ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings)
from pydantic import BaseModel, Field
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name ="BAAI/bge-m3"
)
db= Chroma(
    persist_directory="vector_db",
    embedding_function=embedding
)
retriever = db.as_retriever(
    search_kwargs={"k":3}
)
class MajorAdvice(BaseModel):
    ten_nganh: str = Field(description="Tên ngành được đề xuất")
    mo_ta_nganh: str = Field(description="Mô tả ngắn gọn về ngành")
    ly_do_phu_hop: str = Field(description="Giải thích vì sao ngành phù hợp với người dùng")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0 #giam do sang tao de model dua ra kq chinh xac nhat, khong bi ao giac
)

def ask(gr_major,describe):
    question = "Nhóm ngành phù hợp với tôi là " + gr_major + " và sở thích, tính cách của tôi là " + describe
    print(f"[{datetime.now()}] Đã nhận request")
    docs = retriever.invoke(question) #goi model
    print(f"[{datetime.now()}] Đã retrieve xong")
    context="\n\n".join(
        [
            d.page_content
            for d in docs
        ]
    )



    prompt=f"""

        Bạn là chuyên gia về tư vấn ngành nghề dựa trên nhóm ngành và mô tả tính cách từ học sinh. 
        Từ những thông tin như nhóm ngành kèm mô tả tính cách của người đó, hãy đưa ra 1 ngành phù hợp nhất thuộc nhóm ngành của người dùng nhập vào
        Đọc mô tả ngành 1 cách kỹ lưỡng để đảm bảo nhóm ngành đó phù hợp với tính cách và năng lực của người dùng. 
        Chỉ trả lời dựa trên dữ liệu bên dưới. 
        Nếu dữ liệu không có, hãy nói không tìm thấy thông tin.
        
        Hãy trả lời bằng cách:
        - Đưa ra lời tư vấn rõ ràng, chi tiết
        - giải thích dễ hiểu
        - không chép nguyên văn, nên sáng tạo câu từ cho nó chuyên nghiệp và mạch lạc nhưng vẫn đảm bảo thông tin chính xác
        Chỉ sử dụng context.

        CONTEXT:
        {context}
        CÂU HỎI:
        {question}

    """

    structured_llm = llm.with_structured_output(MajorAdvice)
    response=structured_llm.invoke(prompt)
    print(f"[{datetime.now()}] Gemini trả lời xong")

    return response.model_dump()
