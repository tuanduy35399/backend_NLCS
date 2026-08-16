# Đánh giá Hybrid RAG bằng LlamaIndex

Thư mục này đánh giá độc lập pipeline RAG hiện tại và không thay đổi luồng chạy
FastAPI. Các thành phần được dùng lại gồm Chroma, Neo4j Property Graph, RRF,
BGE reranker, query rewriting, HyDE và answer generator.

## Chỉ số

Retrieval:

- `hit_rate`: có ít nhất một node đúng trong Top-K hay không.
- `mrr`: nghịch đảo thứ hạng của node đúng đầu tiên.
- `precision`: tỷ lệ node đúng trong Top-K.
- `recall`: tỷ lệ node đúng được tìm thấy.
- `ndcg`: chất lượng thứ tự xếp hạng các node đúng.

Generation:

- `faithfulness`: các khẳng định có được context hỗ trợ không.
- `answer_relevancy`: câu trả lời có đúng trọng tâm câu hỏi không.
- `correctness`: mức tương đồng về tính đúng đắn với câu trả lời chuẩn; chỉ chạy
  khi test case có `reference_answer`.

## Các cấu hình so sánh

- `vector`: vector search trực tiếp bằng câu hỏi.
- `graph`: graph search trực tiếp bằng câu hỏi.
- `hybrid`: vector + graph rồi hợp nhất bằng RRF.
- `full`: query rewrite + HyDE + vector + graph + RRF + BGE reranker.

Việc so sánh bốn cấu hình trên là một ablation study, giúp xác định thành phần
nào thực sự cải thiện retrieval và câu trả lời.

## 1. Chuẩn bị test set

File `test_cases.json` có 60 câu hỏi cân bằng theo lĩnh vực và độ khó, được gán
nhãn bằng URL nguồn CTU. Mỗi
test case có dạng:

```json
{
  "id": "curriculum_001",
  "query": "Ngành Kỹ thuật phần mềm học những nội dung gì?",
  "group_major": "Công nghệ thông tin và Truyền thông",
  "expected_urls": ["https://tuyensinh.ctu.edu.vn/gioi-thieu-nganh/..."],
  "reference_answer": "Câu trả lời chuẩn từ nguồn chính thức.",
  "difficulty": "easy",
  "category": "program_overview",
  "unanswerable": false
}
```

Mỗi câu cần có `expected_urls` hoặc `expected_ids`. URL được ưu tiên vì ổn định
hơn UUID của chunk sau khi re-index. `group_major` và `reference_answer` có thể
để trống. Riêng câu ngoài phạm vi có `unanswerable: true` nên không bắt buộc có
nguồn; các retrieval metric của những câu này được ghi là `null` và không được
tính vào điểm retrieval trung bình.

## 2. Xem node ID để gán nhãn

Khởi động Neo4j trước, bảo đảm Chroma đã được index và LLM endpoint trong
`graphRAG/llm/custom.py` đang khả dụng. Từ thư mục `backend_NLCS`, chạy:

```powershell
python -m evaluate_rag.evaluate inspect `
  --query "Ngành Kỹ thuật phần mềm học những nội dung gì?" `
  --config full `
  --top-k 5
```

Lệnh in ra `node_id`, score, metadata và phần đầu nội dung. Người đánh giá đọc
nội dung rồi đưa những node đúng vào `expected_ids`. Việc gán nhãn cần dựa trên
nguồn chính thức, không chọn node chỉ vì hệ thống đã retrieve nó.

## 3. Chạy retrieval evaluation trước

```powershell
python -m evaluate_rag.evaluate run `
  --dataset evaluate_rag/test_cases.json `
  --config all `
  --top-k 5 `
  --retrieval-only
```

Chế độ này không sinh câu trả lời và không dùng LLM làm judge. Nên dùng nó để
kiểm tra test set và node ID trước vì chạy nhanh hơn.

Nếu BGE-M3 đã được tải về máy và muốn ngăn Hugging Face kiểm tra mạng, có thể
chạy offline:

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
python -m evaluate_rag.evaluate run `
  --dataset evaluate_rag/test_cases.json `
  --config vector --top-k 5 --retrieval-only
```

`vector` chỉ cần Chroma; `graph`, `hybrid` và `full` cần Neo4j đang chạy.

## 4. Chạy đánh giá đầy đủ

```powershell
python -m evaluate_rag.evaluate run `
  --dataset evaluate_rag/test_cases.json `
  --config all `
  --top-k 5
```

Đánh giá đầy đủ gọi LLM nhiều lần cho mỗi test case. Hãy cố định model, endpoint
và cấu hình sinh khi so sánh các lần chạy.

## Kết quả

Mỗi lần chạy tạo ba file trong `evaluate_rag/results`:

- `rag_evaluation_*.json`: kết quả từng câu, node đã retrieve, feedback của
  evaluator và latency.
- `rag_evaluation_summary_*.csv`: điểm trung bình theo từng cấu hình.
- `rag_evaluation_summary_*.md`: bảng Markdown có thể dán vào báo cáo.

Lần chạy retrieval-only bằng Chroma trên bộ 10 câu cũ đã tạo bảng
`results/rag_evaluation_summary_20260815_095256.md`. Kết quả vector là Hit Rate
1.0000, MRR 0.8500, Precision 0.2683, Recall 1.0000 và NDCG 0.8893. Đây mới là
kết quả baseline vector; chỉ so sánh mô hình sau khi chạy thêm cùng test set cho
`graph`, `hybrid` và `full` trong cùng một môi trường.

Chỉ các test case chạy thành công được đưa vào hai file kết quả. Lỗi kết nối hoặc
lỗi test case được in ra terminal để sửa trước khi chạy lại.

## Lưu ý phương pháp

- Nên có ít nhất 30 câu; 50-100 câu sẽ có ý nghĩa hơn.
- Chia câu hỏi theo mô tả ngành, chương trình học, kỹ năng, nghề nghiệp, so sánh,
  thiếu thông tin và ngoài phạm vi.
- LLM-as-a-judge không hoàn toàn khách quan. Nên kiểm tra thủ công một phần kết
  quả và ghi rõ judge model được sử dụng.
- `faithfulness` cao không có nghĩa câu trả lời đúng với thế giới thật; nó chỉ
  cho biết câu trả lời bám vào context. Vì vậy cần kiểm chứng chất lượng nguồn và
  `reference_answer` riêng.
