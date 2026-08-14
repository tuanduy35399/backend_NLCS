# Backend NLCS

**Ngày bắt đầu**: 17/05/2026

**Ngày hoàn thành**: 11/08/2026

Backend gồm hai chức năng đang chạy:

- `POST /predict`: dự đoán nhóm ngành từ điểm và mã Holland.
- `POST /chat`: dùng RAG hiện tại để chọn một ngành cụ thể.

## Yêu cầu:

Máy cần cài đặt **Python 3.12.4** là tốt nhất, khuyên không nên dùng phiên bản mới nhất

vì sẽ xảy ra xung đột do một số thư viện chưa khả dụng trên Python mới nhất.

Link tải **Python 3.12.4**: https://www.python.org/downloads/release/python-3124/

Cài đặt **Neo4J**: https://neo4j.com/download/

## Clone dự án về để chung thư mục với client_NLCS

Link repo **client_NLCS**: https://github.com/tuanduy35399/client_NLCS.git

## Từ thư mục `backend_NLCS`:

Cần chạy lần lượt các lệnh sau:

### Tạo môi trường vnev

```powershell
python -m venv venv
```

### Kích hoạt môi trường venv

```powershell
\venv\Scripts\activate
```

### Cài các thư viện cần thiết với requirements.txt

```powershell
pip install -r requirements.txt
```

### Upload dữ liệu Graph lên Neo4J Desktop

Bước 1: Chạy Neo4J Desktop tạo instance rồi tạo luôn 1 database

Bước 2: Nhấp vào dấu 3 chấm rồi chọn **Load database from file** 

Bước 3: Dẫn vào folder graphRAG/database/neo4j_db_file_dump_backup/neo4j-2026-08-14T11-45-39.dump

Bước 4: Bấm vào nút run để chạy Database

Bước 5: Vào 3 chấm > Plugin > Cài APOC

Bước 6: Chạy Database vừa tạo

Bước 7: Vào file **.env** nhập những thông tin cần thiết vô là chạy được

### Chạy API

```powershell
uvicorn graphRAG.api.main:app --reload
```

Khởi động lúc đầu sẽ hơi lâu....

Mặc định API dùng model năm **2025**. Có thể đổi ở file _graphRAG/api/main_

**Lưu ý: Muốn sử dụng được LLM thì cần có khóa `NVIDIA_API_KEY` đặt trong `.env`.**

Hiện tại do em không có thẻ Visa để đăng ký render, railway host backend FastAPI nên không deploy được.

Mong cô thông cảm nha cô.

Thư mục `graphRAG` là phiên bản Hybrid RAG mới nhất.

Thư mục `old_rag` là phiên bản RAG classic và đã không còn dùng nữa.
