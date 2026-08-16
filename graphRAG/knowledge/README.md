# Làm mới dữ liệu ngành CTU

Crawler chỉ đọc danh sách và bài giới thiệu ngành từ website tuyển sinh chính
thức của Đại học Cần Thơ. Chạy kiểm tra nhanh, không ghi file:

```powershell
python graphRAG/knowledge/update_ctu_majors.py --dry-run --max-pages 3
```

Chạy cập nhật thủ công:

```powershell
python graphRAG/knowledge/update_ctu_majors.py
```

Workflow `update-ctu-major-knowledge.yml` chạy lúc 09:00 mỗi thứ Hai theo giờ
Việt Nam và chỉ commit khi JSON thực sự thay đổi. Crawler có retry, kiểm tra số
trang tối thiểu, ghi file nguyên tử và giữ lại bản ghi cũ nếu một trang đơn lẻ
tạm thời không tải được.

Sau khi JSON thay đổi, cần chạy lại quy trình indexing Chroma/Neo4j của dự án
trước khi RAG sử dụng nội dung mới. Nên xem diff của JSON trước khi re-index để
phát hiện trường hợp website CTU thay đổi cấu trúc HTML.
