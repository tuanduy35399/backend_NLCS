# Backend NLCS

Backend gồm hai chức năng đang chạy:

- `POST /predict`: dự đoán nhóm ngành từ điểm và mã Holland.
- `POST /chat`: dùng RAG hiện tại để chọn một ngành cụ thể.

## Chạy API

Từ thư mục `backend_NLCS`:

```powershell
..\venv\Scripts\python.exe -m uvicorn graphRAG.api.main:app --reload
```

Mặc định API dùng model năm 2025. Có thể đổi trước khi chạy:

```powershell
$env:MODEL_YEAR="2026"  # 2025, 2026 hoặc mixed
```

Khóa `GOOGLE_API_KEY` đặt trong `.env`. Nếu cài môi trường mới, cài thêm
`requirements-rag-runtime.txt` để chạy `/chat`.

Thư mục `graphRAG` là phiên bản Hybrid RAG đang phát triển. Nó đã dùng dữ liệu
`ctu_majors.json`, nhưng chưa được nối vào `/chat` vì còn cần Neo4j và bước build index.
